from icub_mujoco.envs.icub_visuomanip import ICubEnv
import numpy as np


class ICubEnvReaching(ICubEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        action += self.env.physics.data.qpos[self.joints_to_control_ids]
        action -= self.init_qpos[self.joints_to_control_ids]
        null_action = np.zeros(len(self.init_qpos))
        np.put(null_action, self.joints_to_control_ids, action)
        action = null_action
        target = np.clip(np.add(self.init_qpos, action), self.state_space.low, self.state_space.high)
        self.do_simulation(target, self.frame_skip)
        eef_pos_after_sim = self.env.physics.data.xpos[self.eef_id_xpos].copy()
        done_limits = len(self.joints_out_of_range()) > 0
        done_goal = self.goal_reached(eef_pos_after_sim)
        observation = self._get_obs()
        reward = self._get_reward(eef_pos_after_sim, done_limits, done_goal)
        self.eef_pos = eef_pos_after_sim.copy()
        done_timesteps = self.steps >= self._max_episode_steps
        done_object_falling = self.falling_object() and self.use_table
        done = done_limits or done_goal or done_timesteps or done_object_falling
        info = {'Steps': self.steps,
                'Done': {'timesteps': done_timesteps,
                         'goal_reached': done_goal,
                         'limits exceeded': self.joints_out_of_range(),
                         'object falling from the table': done_object_falling}}
        if done and self.print_done_info:
            print(info)

        return observation, reward, done, info

    def _get_reward(self, eef_pos_after_sim, done_limits, done_goal):
        if done_limits:
            return self.reward_out_of_joints
        reward = (np.linalg.norm(self.eef_pos - self.target_eef_pos)
                  - np.linalg.norm(eef_pos_after_sim - self.target_eef_pos)) * self.reward_single_step_multiplier
        if done_goal:
            reward += self.reward_goal
        return reward

    def goal_reached(self, eef_pos_after_sim):
        return np.linalg.norm(eef_pos_after_sim - self.target_eef_pos) < self.goal_xpos_tolerance

    def reset_model(self):
        super().reset_model()
        self.eef_pos = self.env.physics.data.xpos[self.eef_id_xpos].copy()
        return self._get_obs()
