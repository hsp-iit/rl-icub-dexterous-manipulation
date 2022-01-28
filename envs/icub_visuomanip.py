from dm_control import mujoco, composer, mjcf
import os
import gym
import numpy as np
import cv2
import yaml


class ICubEnv(gym.Env):

    def __init__(self,
                 model_path,
                 frame_skip=5,
                 obs_from_img=False,
                 random_initial_qpos=True,
                 render_cameras=(),
                 use_only_torso_and_arms=True,
                 initial_qpos_path='../config/initial_qpos.yaml',
                 print_done_infos=False,
                 reward_goal=1.0,
                 reward_out_of_joints=-1.0,
                 reward_single_step_multiplier=10.0):

        # Load xml model
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        # Initialize dm_control environment
        self.physics = mujoco.Physics.from_xml_path(fullpath)
        self.world = mjcf.from_path(fullpath)
        self.world_entity = composer.ModelWrapperEntity(self.world)
        self.task = composer.NullTask(self.world_entity)
        self.env = composer.Environment(self.task)

        # Set environment and task parameters
        self.obs_from_img = obs_from_img
        self.random_initial_qpos = random_initial_qpos
        self.frame_skip = frame_skip
        self.steps = 0
        self._max_episode_steps = 2000
        self.render_cameras = render_cameras
        self.print_done_infos = print_done_infos

        # Load initial qpos from yaml file and map joint ids to actuator ids
        with open(initial_qpos_path) as initial_qpos_file:
            self.init_qpos_dict = yaml.load(initial_qpos_file, Loader=yaml.FullLoader)
        self.actuator_names = [actuator.name for actuator in self.world_entity.mjcf_model.find_all('actuator')]
        self.joint_names = [joint.name for joint in self.world_entity.mjcf_model.find_all('joint')]
        self.map_joint_to_actuators = []
        for actuator in self.actuator_names:
            self.map_joint_to_actuators.append(self.joint_names.index(actuator))
        self.init_qpos = np.array([], dtype=np.float32)
        for joint in self.joint_names:
            self.init_qpos = np.append(self.init_qpos, self.init_qpos_dict[joint])
        self.init_qvel = np.zeros(self.init_qpos.shape, dtype=np.float32)
        self.joint_ids = np.arange(len(self.joint_names), dtype=np.int64)

        # Define if using the whole body or only torso and right arm
        self.use_only_torso_and_arms = use_only_torso_and_arms
        if self.use_only_torso_and_arms:
            self.joints_to_control = [j for j in self.joint_names if (j.startswith('r_wrist') or
                                                                      j.startswith('r_elbow') or
                                                                      j.startswith('r_shoulder') or
                                                                      j.startswith('torso_yaw'))]
            self.joint_ids = np.array([], dtype=np.int64)
            for joint_id, joint_name in enumerate(self.joint_names):
                if joint_name in self.joints_to_control:
                    self.joint_ids = np.append(self.joint_ids, joint_id)

        # Set spaces
        self.max_delta_qpos = 0.1
        self._set_action_space()
        self._set_observation_space()
        self._set_state_space()

        # Reset environment
        self.reset()
        self.env.reset()

        # Set reward values
        self.reward_goal = reward_goal
        self.reward_out_of_joints = reward_out_of_joints
        self.reward_single_step_multiplier = reward_single_step_multiplier

        # Set task parameters
        self.eef_name = 'r_hand'
        self.eef_id_xpos = self.env.physics.model.name2id('r_hand', 'body')
        self.eef_pos = self.env.physics.data.xpos[self.eef_id_xpos].copy()
        self.target_eef_pos = np.array([-0.3, 0.1, 1.01])
        self.goal_xpos_tolerance = 0.05

    def _set_action_space(self):
        if self.use_only_torso_and_arms:
            n_joints_to_control = len(self.joints_to_control)
        else:
            n_joints_to_control = len(self.actuator_names)
        self.action_space = gym.spaces.Box(low=-self.max_delta_qpos,
                                           high=self.max_delta_qpos,
                                           shape=(n_joints_to_control,),
                                           dtype=np.float32)

    def _set_observation_space(self):
        if self.obs_from_img:
            # TODO implement this for using observations from camera
            pass
        else:
            bounds = np.concatenate([np.expand_dims(joint.range, 0)
                                     for joint in self.world_entity.mjcf_model.find_all('joint')],
                                    axis=0,
                                    dtype=np.float32)
            low = bounds[:, 0][self.joint_ids]
            high = bounds[:, 1][self.joint_ids]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_state_space(self):
        bounds = np.concatenate([np.expand_dims(joint.range, 0)
                                 for joint in self.world_entity.mjcf_model.find_all('joint')], axis=0, dtype=np.float32)
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.state_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.steps = 0
        ob = self.reset_model()
        return ob

    # https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py
    def get_state(self):
        return np.concatenate(self._physics_state_items())

    # https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py
    def set_state(self, physics_state):
        state_items = self._physics_state_items()
        expected_shape = (sum(item.size for item in state_items),)
        if expected_shape != physics_state.shape:
            raise ValueError('Input physics state has shape {}. Expected {}.'.format(
                physics_state.shape, expected_shape))
        start = 0
        for state_item in state_items:
            size = state_item.size
            np.copyto(state_item, physics_state[start:start + size])
            start += size

    # https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py
    def _physics_state_items(self):
        return [self.env.physics.data.qpos, self.env.physics.data.qvel, self.env.physics.data.act]

    def do_simulation(self, ctrl, n_frames):
        for _ in range(n_frames):
            self.env.step(ctrl[self.map_joint_to_actuators])
        self.steps += 1

    def _get_obs(self):
        self.render()
        if self.obs_from_img:
            return self.env.physics.render(height=480, width=640, camera_id='head_cam')
        else:
            return self.get_state()[:len(self.joint_names)][self.joint_ids]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.use_only_torso_and_arms:
            # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
            # for the other joints
            action += self.env.physics.data.qpos[self.joint_ids]
            action -= self.init_qpos[self.joint_ids]
            null_action = np.zeros(len(self.joint_names))
            np.put(null_action, self.joint_ids, action)
            action = null_action
            target = np.clip(np.add(self.init_qpos, action), self.state_space.low, self.state_space.high)
        else:
            target = np.clip(np.add(self.env.physics.data.qpos, action), self.state_space.low, self.state_space.high)

        self.do_simulation(target, self.frame_skip)
        eef_pos_after_sim = self.env.physics.data.xpos[self.eef_id_xpos].copy()
        done_limits = len(self.joints_out_of_range()) > 0
        done_goal = self.goal_reached()
        observation = self._get_obs()
        reward = self._get_reward(eef_pos_after_sim, done_limits, done_goal)
        self.eef_pos = eef_pos_after_sim.copy()
        done_timesteps = self.steps >= self._max_episode_steps
        done = done_limits or done_goal or done_timesteps
        info = {'Steps': self.steps,
                'Done': {'timesteps': done_timesteps,
                         'goal_reached': done_goal,
                         'limits exceeded': self.joints_out_of_range()}}
        if done and self.print_done_infos:
            print(info)

        return observation, reward, done, info

    def reset_model(self):
        if self.random_initial_qpos:
            random_pos = self.state_space.sample()[self.joint_ids]
            self.init_qpos[self.joint_ids] = random_pos
        self.set_state(np.concatenate([self.init_qpos.copy(), self.init_qvel.copy(), self.env.physics.data.act]))
        self.env.physics.forward()
        return self._get_obs()

    def _get_reward(self, eef_pos_after_sim, done_limits, done_goal):
        if done_limits:
            return self.reward_out_of_joints
        reward = (np.linalg.norm(self.eef_pos - self.target_eef_pos)
                  - np.linalg.norm(eef_pos_after_sim - self.target_eef_pos)) * self.reward_single_step_multiplier
        if done_goal:
            reward += self.reward_goal
        return reward

    def joints_out_of_range(self):
        joints_out_of_range = []
        if not self.state_space.contains(self.env.physics.data.qpos):
            for i in range(len(self.env.physics.data.qpos)):
                if self.env.physics.data.qpos[i] < self.state_space.low[i] or \
                        self.env.physics.data.qpos[i] > self.state_space.high[i]:
                    joints_out_of_range.append(self.joint_names[i])
        return joints_out_of_range

    def goal_reached(self):
        return np.linalg.norm(self.eef_pos - self.target_eef_pos) < self.goal_xpos_tolerance

    def render(self, mode='human'):
        del mode  # Unused
        for cam in self.render_cameras:
            img = self.env.physics.render(height=480, width=640, camera_id=cam)
            cv2.imshow(cam, img[:, :, ::-1])
            cv2.waitKey(1)
