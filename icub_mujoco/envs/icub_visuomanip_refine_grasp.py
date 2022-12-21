from icub_mujoco.envs.icub_visuomanip import ICubEnv
import numpy as np
from icub_mujoco.utils.pcd_utils import pcd_from_depth, points_in_world_coord
from icub_mujoco.utils.superquadrics_utils import SuperquadricEstimator
from icub_mujoco.utils.gaze_controller import GazeController
from dm_control.utils import inverse_kinematics as ik
import random
from pyquaternion import Quaternion
from icub_mujoco.utils.idyntree_ik import IDynTreeIK
from icub_mujoco.utils.dm_robotics_ik import DMRoboticsIK
from icub_mujoco.utils.ikin_ik import IKinIK


class ICubEnvRefineGrasp(ICubEnv):

    def __init__(self, **kwargs):
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')

        super().__init__(**kwargs)

        self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
        self.superquadric_estimator = SuperquadricEstimator(self.pregrasp_distance_from_grasp_pose)

        if self.control_gaze:
            self.gaze_controller = GazeController()
            self.fixation_point = self.objects_positions[0]

        self.prev_obj_zpos = None
        self.reward_obj_height = True
        self.already_touched_with_2_fingers = False
        self.already_touched_with_5_fingers = False

        self.prev_dist_superq_center = None
        self.prev_dist_line_superq_center = None
        self.prev_dist_superq_grasp_position = None

        self.superq_pose = None
        self.superq_position = None
        self.target_ik = None

        if self.ik_solver == 'idyntree':
            self.ik_idyntree_reduced_model = False
            self.ik_idyntree = IDynTreeIK(joints_to_control=self.joints_to_control_ik_sorted,
                                          joints_icub=self.joint_names_icub,
                                          eef_frame='r_hand_dh_frame',
                                          reduced_model=self.ik_idyntree_reduced_model)
        elif self.ik_solver == 'dm_robotics':
            self.ik_dm_robotics = DMRoboticsIK(self.world_entity.mjcf_model,
                                               joints_to_control=self.joints_to_control_ik_sorted)
        elif self.ik_solver == 'ikin':
            self.ik_ikin = IKinIK(self.joints_to_control_ik)

        self.lfd_stage = 'close_hand' if not self.lfd_with_approach else 'approach_object'
        self.lfd_approach_object_step = 0
        if self.pregrasp_distance_from_grasp_pose < 0.1:
            self.lfd_approach_object_max_steps = int(100 * (1 - self.pregrasp_distance_from_grasp_pose / 0.1))
        else:
            self.lfd_approach_object_max_steps = 100
        self.lfd_close_hand_step = 0
        self.lfd_close_hand_max_steps = 500
        self.lfd_approach_position = None
        self.close_hand_action_fingers = np.zeros(len(self.actuators_to_control_fingers_ids))
        self.lfd_steps = 0
        self.pre_approach_object_steps = 0
        if self.pregrasp_distance_from_grasp_pose < 0.1:
            self.pre_approach_object_max_steps = 100 - self.lfd_approach_object_max_steps
        else:
            self.pre_approach_object_max_steps = 100

        self.r_hand_to_r_hand_dh_frame = ([[-9.65925844e-01, -2.58818979e-01, 3.88881728e-17, -0.05765429784284365],
                                           [1.40347148e-18, -1.84721605e-16, -1.00000000e+00, -0.005556799999999987],
                                           [2.58818979e-01, -9.65925844e-01, 1.37057067e-16, 0.013693832308330513],
                                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.000000000e+00]])
        self.inv_r_hand_to_r_hand_dh_frame = np.linalg.inv(np.array(self.r_hand_to_r_hand_dh_frame))

    def step(self, action, increase_steps=True, pre_approach_phase=False):
        if self.control_gaze:
            neck_qpos = self.gaze_controller.gaze_control(self.fixation_point,
                                                          self.env.physics.named.data.qpos['neck_pitch'],
                                                          self.env.physics.named.data.qpos['neck_roll'],
                                                          self.env.physics.named.data.qpos['neck_yaw'],
                                                          self.env.physics.named.data.xpos['chest'],
                                                          self.env.physics.named.data.xmat['chest'])
        if not pre_approach_phase:
            if self.learning_from_demonstration or \
                    ((self.approach_in_reset_model or self.curriculum_learning_approach_object)
                     and self.lfd_stage == 'approach_object'):
                if self.lfd_steps <= self.learning_from_demonstration_max_steps or \
                        (self.approach_in_reset_model and self.lfd_stage == 'approach_object'):
                    if increase_steps:
                        self.lfd_steps += 1
                    action = self.collect_demonstrations()
                    action_lfd = action
                else:
                    self.learning_from_demonstration = False
        # If the hand is touching the object, remove constraints on fingers actuators
        if self.number_of_contacts == 0:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = np.clip(action, self.action_space_with_touch.low, self.action_space_with_touch.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        named_qpos = self.env.physics.named.data.qpos
        if 'cartesian' in self.icub_observation_space and not self.use_only_right_hand_model:
            self.target_ik[self.cartesian_ids] = self.target_ik[self.cartesian_ids] + action[:len(self.cartesian_ids)]
            done_ik = False
            if self.cartesian_orientation == 'ypr':
                qy = Quaternion(axis=[0, 0, 1], angle=self.target_ik[3])
                qp = Quaternion(axis=[0, 1, 0], angle=self.target_ik[4])
                qr = Quaternion(axis=[1, 0, 0], angle=self.target_ik[5])
                # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L1018
                target_ik_pyquaternion = qr * qp * qy
                target_ik_quaternion = target_ik_pyquaternion.q
            else:
                target_ik_quaternion = self.target_ik[3:7]

            if self.ik_solver == 'idyntree':
                ik_sol, solved = self.ik_idyntree.solve_ik(eef_pos=self.target_ik[:3],
                                                           eef_quat=target_ik_quaternion,
                                                           current_qpos=self.env.physics.named.data.qpos,
                                                           desired_configuration=None)
                if solved:
                    if not self.ik_idyntree_reduced_model:
                        qpos_ik = ik_sol[np.array(self.joints_to_control_ik_ids, dtype=np.int32)]
                    else:
                        qpos_ik = ik_sol
                else:
                    done_ik = True
            elif self.ik_solver == 'dm_robotics':
                ik_sol, solved = self.ik_dm_robotics.solve_ik(eef_pos=self.target_ik[:3],
                                                              eef_quat=target_ik_quaternion,
                                                              current_qpos=None)
                if solved:
                    qpos_ik = ik_sol
                else:
                    done_ik = True
            elif self.ik_solver == 'ikin':
                target_ik_pyquaternion = Quaternion(target_ik_quaternion)
                target_ik_axis_angle = np.append(target_ik_pyquaternion.axis, target_ik_pyquaternion.angle)
                ik_sol, solved = self.ik_ikin.solve_ik(eef_pos=self.target_ik[:3],
                                                       eef_axis_angle=target_ik_axis_angle,
                                                       current_qpos=self.env.physics.named.data.qpos,
                                                       joints_to_control_ik_ids=self.joints_to_control_ik_ids)
                if solved:
                    qpos_ik = ik_sol
                else:
                    done_ik = True
            else:
                qpos_ik_result = ik.qpos_from_site_pose(physics=self.env.physics,
                                                        site_name='r_hand_dh_frame_site',
                                                        target_pos=self.target_ik[:3],
                                                        target_quat=target_ik_quaternion,
                                                        joint_names=self.joints_to_control_ik)
                if qpos_ik_result.success:
                    qpos_ik = qpos_ik_result.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int32)]
                else:
                    done_ik = True
            # Use as action only the offsets for the joints to control (e.g. hands)
            action = action[len(self.cartesian_ids):]
        elif 'cartesian' in self.icub_observation_space and self.use_only_right_hand_model:
            self.target_ik[self.cartesian_ids] = self.target_ik[self.cartesian_ids] + action[:len(self.cartesian_ids)]
            done_ik = False
            if self.cartesian_orientation == 'ypr':
                qy = Quaternion(axis=[0, 0, 1], angle=self.target_ik[3])
                qp = Quaternion(axis=[0, 1, 0], angle=self.target_ik[4])
                qr = Quaternion(axis=[1, 0, 0], angle=self.target_ik[5])
                # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L1018
                target_ik_pyquaternion = qr * qp * qy
                target_ik_quaternion = target_ik_pyquaternion.q
            else:
                target_ik_quaternion = self.target_ik[3:7]

            superq_pose_transformation_matrix = Quaternion(target_ik_quaternion).transformation_matrix
            superq_pose_transformation_matrix[:3, 3] = self.target_ik[:3]

            superq_pose_r_hand = np.matmul(superq_pose_transformation_matrix, self.inv_r_hand_to_r_hand_dh_frame)
            self.env.physics.named.data.mocap_pos['icub_r_hand_welding'] = superq_pose_r_hand[:3, 3]
            self.env.physics.named.data.mocap_quat['icub_r_hand_welding'] = Quaternion(matrix=superq_pose_r_hand).q
            # Use as action only the offsets for the joints to control (e.g. hands)
            action = action[len(self.cartesian_ids):]
            if superq_pose_r_hand[0, 3] <= -0.5 or superq_pose_r_hand[0, 3] >= 0 or \
                    superq_pose_r_hand[1, 3] <= -0.3 or superq_pose_r_hand[1, 3] >= 0.5 or \
                    superq_pose_r_hand[2, 3] <= 1.0 or superq_pose_r_hand[2, 3] >= 1.6:
                done_ik = True
        qpos_jnt_tendons = np.empty([0, ], dtype=np.float32)
        for actuator in self.actuators_to_control_dict:
            qpos_jnt_tendons = np.append(qpos_jnt_tendons,
                                         np.sum(named_qpos[actuator['jnt']] * actuator['coeff']))
        action += qpos_jnt_tendons
        action -= self.init_icub_act_after_superquadrics[self.actuators_to_control_ids]
        null_action = np.zeros(len(self.init_icub_act_after_superquadrics))
        np.put(null_action, self.actuators_to_control_ids, action)
        action = null_action
        target = np.clip(np.add(self.init_icub_act_after_superquadrics, action),
                         self.actuators_space.low + self.actuators_margin,
                         self.actuators_space.high - self.actuators_margin)
        if self.control_gaze:
            target[self.actuators_to_control_gaze_controller_ids] = np.reshape(neck_qpos.x, (3,))
        if 'cartesian' in self.icub_observation_space and not self.use_only_right_hand_model:
            if not done_ik:
                target[self.actuators_to_control_ik_ids] = qpos_ik
        self.prev_obj_zpos = self.env.physics.data.qpos[self.joint_ids_objects[2]]
        self.do_simulation(target, self.frame_skip, increase_steps=increase_steps)
        if self.done_if_joints_out_of_limits:
            done_limits = len(self.joints_out_of_range()) > 0
        else:
            done_limits = False
        done_goal = self.goal_reached()
        observation = self._get_obs()
        if self.number_of_contacts >= 2:
            self.already_touched_with_2_fingers = True
        if self.number_of_contacts == 5:
            self.already_touched_with_5_fingers = True
        done_timesteps = self.steps >= self._max_episode_steps
        done_moved_object = self.moved_object()
        if self.do_not_consider_done_z_pos:
            done_z_pos = False
        else:
            done_z_pos = self.done_z_r_hand()
        reward = self._get_reward(done_limits, done_goal, done_timesteps, done_moved_object, done_z_pos)
        done_object_falling = self.falling_object() and self.use_table
        done = done_limits or done_goal or done_timesteps or done_object_falling or done_moved_object or done_z_pos
        info = {'Steps': self.steps,
                'Done': {'timesteps': done_timesteps,
                         'goal_reached': done_goal,
                         'limits exceeded': self.joints_out_of_range(),
                         'object falling from the table': done_object_falling,
                         'moved object': done_moved_object,
                         'done z pos': done_z_pos},
                'is_success': done_goal}
        if self.learning_from_demonstration and not pre_approach_phase:
            info['learning from demonstration action'] = action_lfd
        if 'cartesian' in self.icub_observation_space:
            done = done or done_ik
            info['Done']['done IK'] = done_ik
        if self.learning_from_demonstration and done:
            self.lfd_stage = 'close_hand' if not self.lfd_with_approach else 'approach_object'
            self.lfd_close_hand_step = 0
            self.lfd_approach_object_step = 0
        if done and self.print_done_info:
            print(info)
        # Remove self.steps from self.lfd_steps to remove unsuccessful episode steps if required
        if done and self.lfd_keep_only_successful_episodes and not info['is_success'] and \
                self.lfd_steps <= self.learning_from_demonstration_max_steps:
            self.lfd_steps -= self.steps
        return observation, reward, done, info

    def _get_reward(self, done_limits, done_goal, done_timesteps, done_moved_object, done_z_pos, done_ik=None):
        if done_limits:
            return self.reward_out_of_joints
        if done_timesteps:
            return self.reward_end_timesteps
        if done_moved_object:
            if not self.high_negative_reward_approach_failures:
                return -1
            else:
                if self.already_touched_with_2_fingers:
                    return -1
                else:
                    return -100
        if done_ik or done_z_pos:
            return -1
        reward = 0
        reward += self.diff_num_contacts()
        if self.reward_obj_height:
            rew_height = (self.env.physics.data.qpos[self.joint_ids_objects[2]] - self.prev_obj_zpos) * 1000
            # Add positive reward only if all fingers are in contact, add negative reward in any case
            if (rew_height > 0 and self.number_of_contacts >= self.min_fingers_touching_object) or \
                    (rew_height < 0 and self.already_touched_with_5_fingers):
                reward += rew_height
        if self.reward_dist_superq_center and not self.already_touched_with_2_fingers:
            if self.rotated_dist_superq_center:
                superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(
                    self.superq_pose['superq_center'], site_name='r_hand_dh_frame_site_rotated')
            else:
                superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(self.superq_pose['superq_center'])
            current_dist_superq_center = np.linalg.norm(superq_center_in_dh_frame[:2])
            delta_dist_superq_center = self.prev_dist_superq_center - current_dist_superq_center
            self.prev_dist_superq_center = current_dist_superq_center
            reward += delta_dist_superq_center * 100
        if self.reward_line_pregrasp_superq_center and not self.already_touched_with_2_fingers:
            current_dist_line_superq_center = \
                self.point_to_line_distance(self.env.physics.named.data.xpos['r_hand_dh_frame'],
                                            self.superq_pose['superq_center'],
                                            self.superq_pose['distanced_grasp_position'])
            delta_dist_line_superq_center = self.prev_dist_line_superq_center - current_dist_line_superq_center
            self.prev_dist_line_superq_center = current_dist_line_superq_center
            reward += delta_dist_line_superq_center * 100
        if self.reward_dist_original_superq_grasp_position and not self.already_touched_with_2_fingers:
            current_dist_superq_grasp_position = np.linalg.norm(
                self.env.physics.named.data.xpos['r_hand_dh_frame'] - self.superq_pose['original_position'])
            delta_dist_superq_grasp_position = self.prev_dist_superq_grasp_position - current_dist_superq_grasp_position
            self.prev_dist_superq_grasp_position = current_dist_superq_grasp_position
            reward += delta_dist_superq_grasp_position * 100
        if done_goal:
            reward += self.reward_goal
        return reward

    def goal_reached(self):
        if self.goal_reached_only_with_lift_refine_grasp:
            return self.lifted_object()
        else:
            return self.lifted_object() and self.number_of_contacts == 5

    def reset_model(self):
        grasp_found = False
        while not grasp_found:
            super().reset_model()
            if hasattr(self, 'superquadric_estimator'):
                self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
                img = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera)
                depth = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera, depth=True)
                pcd = pcd_from_depth(depth)
                pcd[:, 2] = -pcd[:, 2]
                cam_id = self.env.physics.model.name2id(self.superquadrics_camera, 'camera')
                pcd = points_in_world_coord(pcd,
                                            cam_xpos=self.env.physics.named.data.cam_xpos[cam_id, :],
                                            cam_xmat=np.reshape(self.env.physics.named.data.cam_xmat[cam_id, :],
                                                                (3, 3)))
                pcd[:, 2] -= 1.0
                segm = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera,
                                               segmentation=True)
                ids = np.where(np.reshape(segm[:, :, 0], (segm[:, :, 0].size,)) ==
                               self.env.physics.model.name2id(self.object_visual_mesh_name, 'geom'))
                pcd_colors = np.concatenate((pcd, np.reshape(img, (int(img.size / 3), 3))), axis=1)[ids]
                self.superq_pose = self.superquadric_estimator.compute_grasp_pose_superquadrics(pcd_colors)
                if self.superq_pose['position'][0] == 0.00:
                    print('Grasp pose not found. Resetting the environment.')
                    continue
                self.superq_position = self.superq_pose['position'].copy()
                # Use distanced superq pose if required and if not in the learning from demonstration phase
                if self.distanced_superq_grasp_pose and (
                        not self.learning_from_demonstration or self.lfd_with_approach):
                    self.superq_pose['original_position'] = self.superq_pose['position'].copy()
                    if self.pregrasp_distance_from_grasp_pose >= 0.1:
                        self.superq_pose['position'] = self.superq_pose['distanced_grasp_position'].copy()
                    else:
                        self.superq_pose['position'] = self.superq_pose['distanced_grasp_position_10_cm'].copy()
                if self.cartesian_orientation == 'ypr':
                    self.superq_pose['ypr'] = np.array(Quaternion(self.superq_pose['quaternion']).yaw_pitch_roll)
                    self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['ypr']),
                                                    dtype=np.float64)
                else:
                    self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['quaternion']),
                                                    dtype=np.float64)

                if self.use_only_right_hand_model:
                    superq_pose_transformation_matrix = Quaternion(self.superq_pose['quaternion']).transformation_matrix
                    superq_pose_transformation_matrix[:3, 3] = self.superq_pose['position']
                    superq_pose_r_hand = np.matmul(superq_pose_transformation_matrix,
                                                   self.inv_r_hand_to_r_hand_dh_frame)
                    self.env.physics.named.data.mocap_pos['icub_r_hand_welding'] = superq_pose_r_hand[:3, 3]
                    self.env.physics.named.data.mocap_quat['icub_r_hand_welding'] = Quaternion(
                        matrix=superq_pose_r_hand).q
                    qpos_after_superq_with_mocap = self.init_qpos.copy()
                    qpos_after_superq_with_mocap[self.joint_ids_icub_free] = np.concatenate(
                        (self.env.physics.named.data.mocap_pos['icub_r_hand_welding'],
                         self.env.physics.named.data.mocap_quat['icub_r_hand_welding']))
                    self.set_state(
                        np.concatenate([qpos_after_superq_with_mocap,
                                        self.init_qvel.copy(),
                                        self.env.physics.data.act]))
                    self.env.physics.forward()
                    grasp_found = True
                else:
                    if self.ik_solver == 'idyntree':
                        ik_sol, solved = self.ik_idyntree.solve_ik(eef_pos=self.superq_pose['position'],
                                                                   eef_quat=self.superq_pose['quaternion'],
                                                                   current_qpos=self.env.physics.named.data.qpos,
                                                                   desired_configuration=None)
                        if solved:
                            if not self.ik_idyntree_reduced_model:
                                qpos_sol_final_qpos = ik_sol
                            else:
                                qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                                qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                            grasp_found = True
                    elif self.ik_solver == 'dm_robotics':
                        ik_sol, solved = self.ik_dm_robotics.solve_ik(eef_pos=self.superq_pose['position'],
                                                                      eef_quat=self.superq_pose['quaternion'],
                                                                      current_qpos=None)
                        if solved:
                            qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                            qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                            grasp_found = True
                    elif self.ik_solver == 'ikin':
                        target_ik_pyquaternion = Quaternion(self.superq_pose['quaternion'])
                        target_ik_axis_angle = np.append(target_ik_pyquaternion.axis, target_ik_pyquaternion.angle)
                        ik_sol, solved = self.ik_ikin.solve_ik(eef_pos=self.superq_pose['position'],
                                                               eef_axis_angle=target_ik_axis_angle,
                                                               current_qpos=self.env.physics.named.data.qpos,
                                                               joints_to_control_ik_ids=self.joints_to_control_ik_ids,
                                                               on_step=False)
                        if solved:
                            qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                            qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                            grasp_found = True
                    else:
                        qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                                site_name='r_hand_dh_frame_site',
                                                                target_pos=self.superq_pose['position'],
                                                                target_quat=self.superq_pose['quaternion'],
                                                                joint_names=self.joints_to_control_ik)
                        if qpos_sol_final.success:
                            qpos_sol_final_qpos = qpos_sol_final.qpos
                            grasp_found = True

                    if not grasp_found:
                        max_delta_position_perturb = 1
                        while max_delta_position_perturb < 10:
                            print('Solution not found for superquadric grasp pose, perturb position randomly adding an '
                                  'offset in the range [{:.2f}, {:.2f}]'.format(-0.01 * max_delta_position_perturb,
                                                                                0.01 * max_delta_position_perturb))
                            tmp_superq_position = self.superq_pose['position'].copy()
                            tmp_superq_position[0] += random.uniform(-0.01 * max_delta_position_perturb,
                                                                     0.01 * max_delta_position_perturb)
                            tmp_superq_position[1] += random.uniform(-0.01 * max_delta_position_perturb,
                                                                     0.01 * max_delta_position_perturb)
                            tmp_superq_position[2] += random.uniform(-0.01 * max_delta_position_perturb,
                                                                     0.01 * max_delta_position_perturb)
                            if self.ik_solver == 'idyntree':
                                ik_sol, solved = self.ik_idyntree.solve_ik(eef_pos=self.superq_pose['position'],
                                                                           eef_quat=self.superq_pose['quaternion'],
                                                                           current_qpos=self.env.physics.named.data.qpos,
                                                                           desired_configuration=None)
                                if solved:
                                    if not self.ik_idyntree_reduced_model:
                                        qpos_sol_final_qpos = ik_sol
                                    else:
                                        qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                                        qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                                    grasp_found = True
                            elif self.ik_solver == 'dm_robotics':
                                ik_sol, solved = self.ik_dm_robotics.solve_ik(eef_pos=self.superq_pose['position'],
                                                                              eef_quat=self.superq_pose['quaternion'],
                                                                              current_qpos=None)
                                if solved:
                                    qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                                    qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                                    grasp_found = True
                            elif self.ik_solver == 'ikin':
                                target_ik_pyquaternion = Quaternion(self.superq_pose['quaternion'])
                                target_ik_axis_angle = np.append(target_ik_pyquaternion.axis,
                                                                 target_ik_pyquaternion.angle)
                                ik_sol, solved = self.ik_ikin.solve_ik(eef_pos=self.superq_pose['position'],
                                                                       eef_axis_angle=target_ik_axis_angle,
                                                                       current_qpos=self.env.physics.named.data.qpos,
                                                                       joints_to_control_ik_ids=self.joints_to_control_ik_ids,
                                                                       on_step=False)
                                if solved:
                                    qpos_sol_final_qpos = np.zeros(len(self.joint_ids_icub))
                                    qpos_sol_final_qpos[self.joints_to_control_ik_ids] = ik_sol
                                    grasp_found = True
                            else:
                                qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                                        site_name='r_hand_dh_frame_site',
                                                                        target_pos=self.superq_pose['position'],
                                                                        target_quat=self.superq_pose['quaternion'],
                                                                        joint_names=self.joints_to_control_ik)
                                if qpos_sol_final.success:
                                    qpos_sol_final_qpos = qpos_sol_final.qpos
                                    grasp_found = True
                            if grasp_found:
                                break
                            else:
                                max_delta_position_perturb += 0.1
                        if max_delta_position_perturb >= 10:
                            print('Solution not found after superquadric perturbation. Resetting the environment.')
                            continue

                    self.update_init_qpos_act_after_superquadrics(qpos_sol_final_qpos)
                    target = self.init_icub_act_after_superquadrics.copy()
                    num_steps_initial_movement = 50
                    initial_qpos = self.env.physics.data.qpos[:len(self.joint_ids_icub)].copy()
                    for i in range(num_steps_initial_movement):
                        qpos_i = self.go_to(initial_qpos,
                                            qpos_sol_final_qpos[:len(self.joint_ids_icub)],
                                            i,
                                            num_steps_initial_movement)
                        for actuator_id, actuator_dict in enumerate(self.actuators_dict):
                            if actuator_id in self.actuators_to_control_ik_ids:
                                target[actuator_id] = 0.0
                                for j in range(len(actuator_dict['jnt'])):
                                    target[actuator_id] += qpos_i[self.joint_ids_icub_dict[actuator_dict['jnt'][j]]] * \
                                                           actuator_dict['coeff'][j]

                        self.do_simulation(target, self.frame_skip, increase_steps=False)
                        self._get_obs()

                self.already_touched_with_2_fingers = False
                self.already_touched_with_5_fingers = False
                self.previous_number_of_contacts = self.compute_num_fingers_touching_object()
                if self.reward_dist_superq_center:
                    if self.rotated_dist_superq_center:
                        superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(
                            self.superq_pose['superq_center'], site_name='r_hand_dh_frame_site_rotated')
                    else:
                        superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(self.superq_pose['superq_center'])
                    self.prev_dist_superq_center = np.linalg.norm(superq_center_in_dh_frame[:2])
                if self.reward_line_pregrasp_superq_center:
                    self.prev_dist_line_superq_center = \
                        self.point_to_line_distance(self.env.physics.named.data.xpos['r_hand_dh_frame'],
                                                    self.superq_pose['superq_center'],
                                                    self.superq_pose['distanced_grasp_position'])
                if self.reward_dist_original_superq_grasp_position:
                    self.prev_dist_superq_grasp_position = np.linalg.norm(
                        self.env.physics.named.data.xpos['r_hand_dh_frame'] - self.superq_pose['original_position'])
                if self.pregrasp_distance_from_grasp_pose < 0.1:
                    done = False
                    while self.pre_approach_object_steps < self.pre_approach_object_max_steps and not done:
                        action = np.concatenate([self.pre_approach_object(),
                                                 np.zeros(len(self.actuators_to_control_fingers_ids))])
                        _, _, done, _ = self.step(action=action, increase_steps=False, pre_approach_phase=True)
                        self._get_obs()
                    self.pre_approach_object_steps = 0
                if self.approach_in_reset_model or self.curriculum_learning_approach_object:
                    self.lfd_stage = 'approach_object'
                    done = False
                    while self.lfd_stage == 'approach_object' and not done:
                        if self.curriculum_learning_approach_object and \
                                self.total_steps < self.curriculum_learning_approach_object_start_step:
                            if self.lfd_approach_object_step == self.lfd_approach_object_max_steps:
                                self.lfd_stage = 'close_hand'
                                self.lfd_approach_object_step = 0
                                self.lfd_approach_position = None
                                break
                        elif self.curriculum_learning_approach_object and \
                                self.total_steps < self.curriculum_learning_approach_object_end_step:
                            if self.lfd_approach_object_step / self.lfd_approach_object_max_steps > \
                                    1 - (self.total_steps - self.curriculum_learning_approach_object_start_step) / \
                                    (self.curriculum_learning_approach_object_end_step
                                     - self.curriculum_learning_approach_object_start_step):
                                self.lfd_stage = 'close_hand'
                                self.lfd_approach_object_step = 0
                                self.lfd_approach_position = None
                                break
                        elif self.curriculum_learning_approach_object and \
                                self.total_steps >= self.curriculum_learning_approach_object_end_step:
                            self.lfd_stage = 'close_hand'
                            self.lfd_approach_object_step = 0
                            self.lfd_approach_position = None
                            break
                        _, _, done, _ = self.step(action=None, increase_steps=False)
            else:
                # Initial reset, just need to return the observation
                break
        return self._get_obs()

    def update_init_qpos_act_from_current_state(self, current_state):
        for actuator_id, actuator_dict in enumerate(self.actuators_dict):
            self.init_icub_act_after_superquadrics[actuator_id] = 0.0
            for i in range(len(actuator_dict['jnt'])):
                self.init_icub_act_after_superquadrics[actuator_id] += \
                    current_state[self.joint_ids_icub_dict[actuator_dict['jnt'][i]]] * actuator_dict['coeff'][i]

    def update_init_qpos_act_after_superquadrics(self, superq_qpos):
        self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
        for actuator_id, actuator_dict in enumerate(self.actuators_dict):
            if actuator_id in self.actuators_to_control_ik_ids:
                self.init_icub_act_after_superquadrics[actuator_id] = 0.0
                for i in range(len(actuator_dict['jnt'])):
                    self.init_icub_act_after_superquadrics[actuator_id] += \
                        superq_qpos[self.joint_ids_icub_dict[actuator_dict['jnt'][i]]] * actuator_dict['coeff'][i]

    def diff_num_contacts(self):
        diff = self.number_of_contacts - self.previous_number_of_contacts
        self.previous_number_of_contacts = self.number_of_contacts
        return diff

    def moved_object(self):
        if self.random_mujoco_scanned_object:
            # Only one object can be used for this task, so there is no need to loop
            current_obj_rotation = Quaternion(self.env.physics.data.qpos[self.joint_ids_objects[3:7]])
            axis_angle_curr_rot = current_obj_rotation.axis * current_obj_rotation.angle
            # Object is moved if it is rotated of more than 90 degrees around the x or y axes
            if axis_angle_curr_rot[0] >= np.pi / 2 or axis_angle_curr_rot[0] <= -np.pi / 2 or \
                axis_angle_curr_rot[1] >= np.pi / 2 or axis_angle_curr_rot[1] <= -np.pi / 2:
                return True
        else:
            for list_id, joint_id in enumerate(self.joint_ids_objects):
                # Check only the z component
                if list_id % 7 != 2:
                    continue
                else:
                    if self.env.physics.data.qpos[joint_id] < self.moved_object_height:
                        return True
        return False

    def lifted_object(self):
        # Check only the z component
        if self.env.physics.data.qpos[self.joint_ids_objects[2]] > self.lift_object_height:
            return True
        return False

    def done_z_r_hand(self):
        if self.prev_obj_zpos > self.env.physics.named.data.xpos['r_hand_dh_frame'][2]:
            return True
        return False

    def collect_demonstrations(self):
        if self.lfd_stage == 'close_hand':
            # Close hand
            action_fingers = self.close_hand()
        elif self.lfd_stage == 'approach_object':
            action_fingers = np.zeros(len(self.actuators_to_control_fingers_ids))
        elif self.lfd_stage == 'lift_object':
            action_fingers = self.close_hand_action_fingers
        if self.lfd_stage == 'lift_object':
            # Lift
            action_ik = self.lift_object()
        elif self.lfd_stage == 'approach_object':
            # Approach
            action_ik = self.approach_object()
        else:
            action_ik = np.zeros(len(self.cartesian_ids))
        return np.concatenate((action_ik, action_fingers))

    def close_hand(self):
        self.lfd_close_hand_step += 1
        named_qpos = self.env.physics.named.data.qpos
        target_fingers = np.empty([0, ], dtype=np.float32)
        actions_fingers = np.empty([0, ], dtype=np.float32)
        for actuator in self.actuators_to_control_dict:
            if actuator['name'] in self.actuators_to_control_no_fingers or not actuator['name'].startswith('r_'):
                continue
            else:
                qpos_act = np.sum(named_qpos[actuator['jnt']] * actuator['coeff'])
                act_delta = actuator['close_value'] - actuator['open_value']
                target_i = min(qpos_act + act_delta / 500 * 20,
                               actuator['open_value'] + act_delta * self.lfd_close_hand_step / 500)
                target_fingers = np.append(target_fingers, target_i)
                actions_fingers = np.append(actions_fingers, target_i - qpos_act)
        if self.lfd_close_hand_step == self.lfd_close_hand_max_steps:
            self.lfd_stage = 'lift_object'
            self.lfd_close_hand_step = 0
            self.close_hand_action_fingers = actions_fingers.copy()
        return actions_fingers

    def lift_object(self):
        action_ik = np.zeros(len(self.cartesian_ids))
        action_ik[self.cartesian_ids.index(2)] = 0.002
        return action_ik

    def approach_object(self):
        self.lfd_approach_object_step += 1
        action_ik = np.zeros(len(self.cartesian_ids))
        if self.lfd_approach_position is None:
            self.lfd_approach_position = self.superq_pose['position'].copy()
        action_ik[self.cartesian_ids.index(0)] = (self.superq_position[0] - self.lfd_approach_position[0]) \
                                                 / self.lfd_approach_object_max_steps
        action_ik[self.cartesian_ids.index(1)] = (self.superq_position[1] - self.lfd_approach_position[1]) \
                                                 / self.lfd_approach_object_max_steps
        action_ik[self.cartesian_ids.index(2)] = (self.superq_position[2] - self.lfd_approach_position[2]) \
                                                 / self.lfd_approach_object_max_steps
        if self.lfd_approach_object_step == self.lfd_approach_object_max_steps:
            self.lfd_stage = 'close_hand'
            self.lfd_approach_object_step = 0
            self.lfd_approach_position = None
        return action_ik

    def pre_approach_object(self, ):
        self.pre_approach_object_steps += 1
        final_pose = self.superq_pose['distanced_grasp_position'].copy()
        initial_pose = self.superq_pose['distanced_grasp_position_10_cm'].copy()
        action_ik = np.zeros(len(self.cartesian_ids))
        action_ik[self.cartesian_ids.index(0)] = (final_pose[0] - initial_pose[0]) / self.pre_approach_object_max_steps
        action_ik[self.cartesian_ids.index(1)] = (final_pose[1] - initial_pose[1]) / self.pre_approach_object_max_steps
        action_ik[self.cartesian_ids.index(2)] = (final_pose[2] - initial_pose[2]) / self.pre_approach_object_max_steps
        return action_ik

    # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    @staticmethod
    def point_to_line_distance(x0, x1, x2):
        distance = np.linalg.norm(np.cross(x2-x1, x1-x0))/np.linalg.norm(x2-x1)
        return distance
