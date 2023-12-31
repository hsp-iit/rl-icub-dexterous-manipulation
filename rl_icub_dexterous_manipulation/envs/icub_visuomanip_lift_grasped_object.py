# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from rl_icub_dexterous_manipulation.envs.icub_visuomanip import ICubEnv
import numpy as np
from rl_icub_dexterous_manipulation.utils.pcd_utils import pcd_from_depth, points_in_world_coord
from rl_icub_dexterous_manipulation.utils.superquadrics_utils import SuperquadricEstimator
from pyquaternion import Quaternion
from dm_control.utils import inverse_kinematics as ik


class ICubEnvLiftGraspedObject(ICubEnv):

    def __init__(self, **kwargs):
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')

        self.object_lifted_state = None

        super().__init__(**kwargs)

        self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
        self.superquadric_estimator = SuperquadricEstimator()

        self.curriculum_max_steps = 1000000
        self.total_steps = 0
        self.curriculum_max_offset = self.max_delta_qpos / 10
        self.update_curriculum_max_offset()
        self.curriculum_max_offset_cartesian = self.max_delta_cartesian_pos / 10
        self.update_curriculum_max_offset_cartesian()

        self.prev_obj_zpos = None
        self.reward_obj_height = True
        self.already_touched_with_5_fingers = False
        self.superq_pose = None
        self.target_ik = None
        self.fingers_act_values = None

        self.env.physics.named.model.body_mass['004_sugar_box/'] = 0.5

    def step(self, action):
        qpos_ik = None
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        done_ik = False
        if 'cartesian' in self.icub_observation_space:
            cartesian_pose = np.concatenate((self.env.physics.named.data.xpos[self.eef_name],
                                             Quaternion(
                                                 matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                                   (3, 3)), atol=1e-05).q))[self.cartesian_ids]
            self.target_ik[self.cartesian_ids] = cartesian_pose + action[:len(self.cartesian_ids)]
            qpos_ik_result = ik.qpos_from_site_pose(physics=self.env.physics,
                                                    site_name='r_hand_dh_frame_site',
                                                    target_pos=self.target_ik[:3],
                                                    target_quat=self.target_ik[3:7],
                                                    joint_names=self.joints_to_control_ik)
            if qpos_ik_result.success:
                qpos_ik = qpos_ik_result.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
            else:
                done_ik = True
        target = self.init_icub_act_after_superquadrics.copy()
        if 'cartesian' in self.icub_observation_space:
            if not done_ik:
                target[self.actuators_to_control_ik_ids] = qpos_ik
        target[self.actuators_to_control_fingers_ids] = self.fingers_act_values
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
        if 'cartesian' in self.icub_observation_space:
            done = done or done_ik
            info['Done']['done IK'] = done_ik
        if done and self.print_done_info:
            print(info)
        return observation, reward, done, info

    def _get_reward(self, done_limits, done_goal, done_timesteps, done_moved_object, done_z_pos):
        if done_limits:
            return self.reward_out_of_joints
        if done_timesteps:
            return self.reward_end_timesteps
        if done_moved_object or done_z_pos:
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
        if self.object_lifted_state is not None:
            self.set_state(self.object_lifted_state)
            self.env.physics.forward()
        elif hasattr(self, 'superquadric_estimator'):
            img = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera)
            depth = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera, depth=True)
            pcd = pcd_from_depth(depth)
            pcd[:, 2] = -pcd[:, 2]
            cam_id = self.env.physics.model.name2id(self.superquadrics_camera, 'camera')
            pcd = points_in_world_coord(pcd,
                                        cam_xpos=self.env.physics.named.data.cam_xpos[cam_id, :],
                                        cam_xmat=np.reshape(self.env.physics.named.data.cam_xmat[cam_id, :], (3, 3)))
            pcd[:, 2] -= 0.95
            segm = self.env.physics.render(height=480,
                                           width=640,
                                           camera_id=self.superquadrics_camera,
                                           segmentation=True)
            ids = np.where(np.reshape(segm[:, :, 0], (segm[:, :, 0].size,)) ==
                           self.env.physics.model.name2id(self.objects[0]+"/mesh_"+self.objects[0]+"_00_visual",
                                                          'geom'))
            pcd_colors = np.concatenate((pcd, np.reshape(img, (int(img.size / 3), 3))), axis=1)[ids]
            self.superq_pose = self.superquadric_estimator.compute_grasp_pose_superquadrics(pcd_colors)
            self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['quaternion']),
                                            dtype=np.float64)
            qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                    site_name='r_hand_dh_frame_site',
                                                    target_pos=self.superq_pose['position'],
                                                    target_quat=self.superq_pose['quaternion'],
                                                    joint_names=self.joints_to_control_ik)
            if qpos_sol_final.success:
                qpos_sol_final = qpos_sol_final.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
            current_state = self.get_state()
            current_state[self.joints_to_control_ik_ids] = qpos_sol_final
            self.set_state(current_state)
            self.update_init_qpos_act_from_current_state(current_state)
            self.env.physics.forward()
            self.close_right_hand()
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

    def close_right_hand(self):
        target_before_close = self.init_icub_act_after_superquadrics.copy()
        i = 0
        target_i = None
        while i < 2000:
            i += 1
            named_qpos = self.env.physics.named.data.qpos
            target_i = np.empty([0, ], dtype=np.float32)
            for actuator in self.actuators_to_control_dict:
                if actuator['name'] in self.actuators_to_control_no_fingers or not actuator['name'].startswith('r_'):
                    continue
                else:
                    qpos_act = np.sum(named_qpos[actuator['jnt']] * actuator['coeff'])
                    act_delta = actuator['close_value'] - actuator['open_value']
                    target_i = np.append(target_i, min(qpos_act + act_delta / 500 * 20,
                                                       actuator['open_value'] + act_delta * i / 500))
            target_before_close[self.actuators_to_control_fingers_ids] = target_i
            self.env.step(target_before_close)
            self.render()
        self.fingers_act_values = target_i
        self.object_lifted_state = self.get_state()

    def update_curriculum_max_offset(self):
        if self.total_steps == 0:
            return
        if self.total_steps > self.curriculum_max_steps:
            self.curriculum_max_offset = self.max_delta_qpos
        else:
            self.curriculum_max_offset = max(self.max_delta_qpos / 10,
                                             self.max_delta_qpos * self.total_steps / self.curriculum_max_steps)

    def update_curriculum_max_offset_cartesian(self):
        if self.total_steps == 0:
            return
        if self.total_steps > self.curriculum_max_steps:
            self.curriculum_max_offset_cartesian = self.max_delta_cartesian_pos
        else:
            self.curriculum_max_offset_cartesian = max(self.max_delta_cartesian_pos / 10,
                                                       self.max_delta_cartesian_pos * self.total_steps
                                                       / self.curriculum_max_steps)
