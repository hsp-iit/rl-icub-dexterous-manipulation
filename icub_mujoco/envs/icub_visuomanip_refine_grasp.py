from icub_mujoco.envs.icub_visuomanip import ICubEnv
import numpy as np
from icub_mujoco.utils.pcd_utils import pcd_from_depth, points_in_world_coord
from icub_mujoco.utils.superquadrics_utils import SuperquadricEstimator
from dm_control.utils import inverse_kinematics as ik
import random
from pyquaternion import Quaternion


class ICubEnvRefineGrasp(ICubEnv):

    def __init__(self, **kwargs):
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')

        super().__init__(**kwargs)

        self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
        self.superquadric_estimator = SuperquadricEstimator()

        self.prev_obj_zpos = None
        self.reward_obj_height = True
        self.already_touched_with_5_fingers = False

        self.superq_pose = None
        self.target_ik = None

        self.lfd_stage = 'close_hand'
        self.lfd_close_hand_step = 0
        self.lfd_close_hand_max_steps = 500
        self.close_hand_action_fingers = np.zeros(len(self.actuators_to_control_fingers_ids))
        self.lfd_steps = 0

    def step(self, action):
        if self.learning_from_demonstration:
            if self.lfd_steps <= self.learning_from_demonstration_max_steps:
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
        if 'cartesian' in self.icub_observation_space:
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
        if 'cartesian' in self.icub_observation_space:
            if not done_ik:
                target[self.actuators_to_control_ik_ids] = qpos_ik
        self.prev_obj_zpos = self.env.physics.data.qpos[self.joint_ids_objects[2]]
        self.do_simulation(target, self.frame_skip)
        if self.done_if_joints_out_of_limits:
            done_limits = len(self.joints_out_of_range()) > 0
        else:
            done_limits = False
        done_goal = self.goal_reached()
        observation = self._get_obs()
        if self.number_of_contacts == 5:
            self.already_touched_with_5_fingers = True
        done_timesteps = self.steps >= self._max_episode_steps
        done_moved_object = self.moved_object()
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
                         'done z pos': done_z_pos}}
        if self.learning_from_demonstration:
            info['learning from demonstration action'] = action_lfd
        if 'cartesian' in self.icub_observation_space:
            done = done or done_ik
            info['Done']['done IK'] = done_ik
        if self.learning_from_demonstration and done:
            self.lfd_stage = 'close_hand'
            self.lfd_close_hand_step = 0
        if done and self.print_done_info:
            print(info)
        return observation, reward, done, info

    def _get_reward(self, done_limits, done_goal, done_timesteps, done_moved_object, done_z_pos, done_ik=None):
        if done_limits:
            return self.reward_out_of_joints
        if done_timesteps:
            return self.reward_end_timesteps
        if done_moved_object or done_z_pos:
            return -1
        if done_ik:
            return -1
        reward = 0
        reward += self.diff_num_contacts()
        if self.reward_obj_height:
            rew_height = (self.env.physics.data.qpos[self.joint_ids_objects[2]] - self.prev_obj_zpos) * 1000
            # Add positive reward only if all fingers are in contact, add negative reward in any case
            if (rew_height > 0 and self.number_of_contacts == 5) or \
                    (rew_height < 0 and self.already_touched_with_5_fingers):
                reward += rew_height
        if done_goal:
            reward += self.reward_goal
        return reward

    def goal_reached(self):
        return self.lifted_object() and self.number_of_contacts == 5

    def reset_model(self):
        super().reset_model()
        if hasattr(self, 'superquadric_estimator'):
            img = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera)
            depth = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera, depth=True)
            pcd = pcd_from_depth(depth)
            pcd[:, 2] = -pcd[:, 2]
            cam_id = self.env.physics.model.name2id(self.superquadrics_camera, 'camera')
            pcd = points_in_world_coord(pcd,
                                        cam_xpos=self.env.physics.named.data.cam_xpos[cam_id, :],
                                        cam_xmat=np.reshape(self.env.physics.named.data.cam_xmat[cam_id, :], (3, 3)))
            pcd[:, 2] -= 0.95
            segm = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera,
                                           segmentation=True)
            ids = np.where(np.reshape(segm[:, :, 0], (segm[:, :, 0].size,)) ==
                           self.env.physics.model.name2id(self.objects[0] + "/mesh_" + self.objects[0] + "_00_visual",
                                                          'geom'))
            pcd_colors = np.concatenate((pcd, np.reshape(img, (int(img.size / 3), 3))), axis=1)[ids]
            self.superq_pose = self.superquadric_estimator.compute_grasp_pose_superquadrics(pcd_colors)
            if self.superq_pose['position'][0] == 0.00:
                print('Grasp pose not found. Resetting the environment.')
                self.reset_model()
            if self.cartesian_orientation == 'ypr':
                self.superq_pose['ypr'] = np.array(Quaternion(self.superq_pose['quaternion']).yaw_pitch_roll)
                self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['ypr']),
                                                dtype=np.float64)
            else:
                self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['quaternion']),
                                                dtype=np.float64)
            qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                    site_name='r_hand_dh_frame_site',
                                                    target_pos=self.superq_pose['position'],
                                                    target_quat=self.superq_pose['quaternion'],
                                                    joint_names=self.joints_to_control_ik)
            if qpos_sol_final.success:
                qpos_sol_final = qpos_sol_final.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
            else:
                max_delta_position_perturb = 1
                while max_delta_position_perturb < 10:
                    print('Solution not found for superquadric grasp pose, perturb position randomly adding an '
                          'offset in the range [{:.2f}, {:.2f}]'. format(-0.01*max_delta_position_perturb,
                                                                         0.01*max_delta_position_perturb))
                    tmp_superq_position = self.superq_pose['position'].copy()
                    tmp_superq_position[0] = tmp_superq_position[0] + random.uniform(-0.01*max_delta_position_perturb,
                                                                                     0.01*max_delta_position_perturb)
                    tmp_superq_position[1] = tmp_superq_position[1] + random.uniform(-0.01 * max_delta_position_perturb,
                                                                                     0.01 * max_delta_position_perturb)
                    tmp_superq_position[2] = tmp_superq_position[2] + random.uniform(-0.01 * max_delta_position_perturb,
                                                                                     0.01 * max_delta_position_perturb)
                    qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                            site_name='r_hand_dh_frame_site',
                                                            target_pos=tmp_superq_position,
                                                            target_quat=self.superq_pose['quaternion'],
                                                            joint_names=self.joints_to_control_ik)
                    if qpos_sol_final.success:
                        qpos_sol_final = qpos_sol_final.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
                        break
                    else:
                        max_delta_position_perturb += 0.1
                if max_delta_position_perturb >= 10:
                    print('Solution not found after superquadric perturbation. Resetting the environment.')
                    self.reset_model()
            current_state = self.get_state()
            current_state[self.joints_to_control_ik_ids] = qpos_sol_final
            self.set_state(current_state)
            self.update_init_qpos_act_from_current_state(current_state)
            self.env.physics.forward()
            self.already_touched_with_5_fingers = False
            self.previous_number_of_contacts = self.compute_num_fingers_touching_object()
        return self._get_obs()

    def update_init_qpos_act_from_current_state(self, current_state):
        for actuator_id, actuator_dict in enumerate(self.actuators_dict):
            self.init_icub_act_after_superquadrics[actuator_id] = 0.0
            for i in range(len(actuator_dict['jnt'])):
                self.init_icub_act_after_superquadrics[actuator_id] += \
                    current_state[self.joint_ids_icub_dict[actuator_dict['jnt'][i]]] * actuator_dict['coeff'][i]

    def diff_num_contacts(self):
        diff = self.number_of_contacts - self.previous_number_of_contacts
        self.previous_number_of_contacts = self.number_of_contacts
        return diff

    def moved_object(self):
        for list_id, joint_id in enumerate(self.joint_ids_objects):
            # Check only the z component
            if list_id % 7 != 2:
                continue
            else:
                if self.env.physics.data.qpos[joint_id] < 0.98:
                    return True
        return False

    def lifted_object(self):
        # Check only the z component
        if self.env.physics.data.qpos[self.joint_ids_objects[2]] > self.lift_object_height:
            return True
        return False

    def done_z_r_hand(self):
        if self.prev_obj_zpos > self.env.physics.named.data.xpos['r_hand'][2]:
            return True
        return False

    def collect_demonstrations(self):
        if self.lfd_stage == 'close_hand':
            # Close hand
            action_fingers = self.close_hand()
        else:
            action_fingers = self.close_hand_action_fingers
        if self.lfd_stage == 'lift_object':
            # Lift
            action_ik = self.lift_object()
        else:
            action_ik = np.zeros(len(self.cartesian_components))
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
        action_ik[self.cartesian_ids.index(2)] = self.max_delta_cartesian_pos/10
        return action_ik

