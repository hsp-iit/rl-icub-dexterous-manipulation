# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from rl_icub_dexterous_manipulation.utils.gaze_controller import GazeController
from pyquaternion import Quaternion
from rl_icub_dexterous_manipulation.envs.icub_visuomanip_refine_grasp import ICubEnvRefineGrasp
from rl_icub_dexterous_manipulation.envs.icub_visuomanip_real import ICubEnvReal
import time


class ICubEnvRefineGraspReal(ICubEnvReal, ICubEnvRefineGrasp):

    def __init__(self, **kwargs):
        self.first_reset = True
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')

        super().__init__(**kwargs)

        if self.control_gaze:
            self.gaze_controller = GazeController()
            self.fixation_point = self.objects_positions[0]

        self.prev_obj_zpos = None
        self.reward_obj_height = True
        self.already_touched_with_2_fingers = False
        self.already_touched_with_5_fingers = False

        self.prev_dist_superq_center = None

        self.superq_pose = None
        self.superq_position = None
        self.target_ik = None

        self.close_hand_action_fingers = np.zeros(9)

        self.save_current_model = False
        self.init_time = time.time()

    def step(self, action, increase_steps=True, pre_approach_phase=False):
        # Residual learning
        if 'pretrained_output' in self.prev_obs.keys() \
                and not pre_approach_phase and not self.learning_from_demonstration:
            action += self.prev_obs['pretrained_output']
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        self.robot_module.target_ik = self.target_ik
        self.robot_module.execute_action(action)
        self.prev_obj_zpos = self.robot_module._r_arm_xpos_port.read(True)[2]
        if increase_steps:
            self.steps += 1
        done_goal = self.goal_reached()
        observation = self._get_obs()
        self.compute_num_fingers_touching_object()
        if self.number_of_contacts >= 2:
            self.already_touched_with_2_fingers = True
        if self.number_of_contacts == 5:
            self.already_touched_with_5_fingers = True
        done_timesteps = self.steps >= self._max_episode_steps
        done_moved_object = self.falling_or_moved_object()
        reward = self._get_reward(done_goal, done_timesteps, done_moved_object)
        done = done_goal or done_timesteps or done_moved_object
        info = {'Steps': self.steps,
                'Done': {'timesteps': done_timesteps,
                         'goal_reached': done_goal,
                         'object moved or falling from the table': done_moved_object,
                         },
                'is_success': done_goal}
        if done and self.print_done_info:
            print("Done time:", time.time() - self.init_time)
            print(info)
        return observation, reward, done, info

    def _get_reward(self, done_goal, done_timesteps, done_moved_object):
        if done_timesteps:
            return self.reward_end_timesteps
        if done_moved_object:
            return -1
        reward = 0
        reward += self.diff_num_contacts()
        # print('rew fingers', reward)
        if self.reward_obj_height:
            rew_height = (self.robot_module._r_arm_xpos_port.read(True)[2] - self.prev_obj_zpos) * 1000
            # Add positive reward only if all fingers are in contact, add negative reward in any case
            if (rew_height > 0 and self.number_of_contacts >= self.min_fingers_touching_object) or \
                    (rew_height < 0 and self.already_touched_with_2_fingers):
                if rew_height > 0 and self.scale_pos_lift_reward_wrt_touching_fingers:
                    rew_height *= self.number_of_contacts / 5
                reward += rew_height
        if self.reward_dist_superq_center and not self.already_touched_with_2_fingers:
            superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(self.robot_module.superq_pose['superq_center'])
            current_dist_superq_center = np.linalg.norm(superq_center_in_dh_frame[:2])
            delta_dist_superq_center = self.prev_dist_superq_center - current_dist_superq_center
            self.prev_dist_superq_center = current_dist_superq_center
            reward += delta_dist_superq_center * 100
        if done_goal:
            reward += self.reward_goal
        return reward

    def goal_reached(self):
        if self.goal_reached_only_with_lift_refine_grasp:
            return self.lifted_object()
        else:
            return self.lifted_object() and self.number_of_contacts == 5

    def reset_model(self):
        if self.robot_module.save_current_model:
            self.robot_module.save_current_model = False
            self.save_current_model = True
        grasp_found = False
        while self.robot_module.wait_reset:
            time.sleep(1)
        self.robot_module.state = 'go_back_with_torso'
        self.robot_module.updateModule()
        self.robot_module.state = 'go_to_initial_configuration'
        self.robot_module.updateModule()
        while not grasp_found:
            super().reset_model()
            if hasattr(self.robot_module, 'superquadric_estimator') and not self.first_reset:
                self.robot_module.state = 'compute_superq_grasp_pose'
                self.robot_module.updateModule()
                if self.robot_module.superq_pos[0] == 0.00:
                    print('Grasp pose not found. Resetting the environment.')
                    continue
                grasp_found = True
                self.robot_module.state = 'go_to_distanced_superq_pose_10cm'
                self.robot_module.updateModule()
                self.superq_or_ypr = np.array(Quaternion(self.robot_module.superq_or).yaw_pitch_roll)
                self.target_ik = np.concatenate((self.robot_module.superq_distanced_pos,
                                                 self.superq_or_ypr), dtype=np.float64)
                # Loop to give the robot the time to reach the target pose
                step_cnt = 0
                while step_cnt < 500:
                    current_pos_eef = self.robot_module._r_arm_xpos_port.read(True)
                    if np.linalg.norm(np.array(self.robot_module.superq_distanced_pos_10_cm)
                                      - np.array([current_pos_eef.get(k) for k in range(3)])) < 0.01:
                        break
                    step_cnt += 1
                self.robot_module.state = 'go_to_distanced_superq_pose'
                self.robot_module.updateModule()
                self.superq_or_ypr = np.array(Quaternion(self.robot_module.superq_or).yaw_pitch_roll)
                self.target_ik = np.concatenate((self.robot_module.superq_distanced_pos,
                                                 self.superq_or_ypr), dtype=np.float64)
                # Loop to give the robot the time to reach the target pose
                step_cnt = 0
                while step_cnt < 500:
                    current_pos_eef = self.robot_module._r_arm_xpos_port.read(True)
                    if np.linalg.norm(np.array(self.robot_module.superq_distanced_pos)
                                      - np.array([current_pos_eef.get(k) for k in range(3)])) < 0.01:
                        break
                    step_cnt += 1
                self.already_touched_with_2_fingers = False
                self.already_touched_with_5_fingers = False
                self.previous_number_of_contacts = self.compute_num_fingers_touching_object()
                if self.previous_number_of_contacts >= 2:
                    self.already_touched_with_2_fingers = True
                if self.previous_number_of_contacts == 5:
                    self.already_touched_with_5_fingers = True
                superq_center_in_dh_frame = self.point_in_r_hand_dh_frame(
                    self.robot_module.superq_pose['superq_center'])
                self.prev_dist_superq_center = np.linalg.norm(superq_center_in_dh_frame[:2])
            elif self.first_reset:
                self.first_reset = False
                # Initial reset, just need to return the observation
                break
        return self._get_obs()

    def falling_or_moved_object(self):
        # Need to find a way to understand if the object has fallen
        moved_object = self.robot_module.done_moved
        self.robot_module.done_moved = False
        return moved_object

    def lifted_object(self):
        # Check the z component
        if self.robot_module._r_arm_xpos_port.read(True)[2] >= self.lift_object_height and self.number_of_contacts >= 2:
            return True
        return False
