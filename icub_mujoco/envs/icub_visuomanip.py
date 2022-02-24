from dm_control import composer, mjcf
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
                 random_initial_pos=True,
                 objects=(),
                 use_table=True,
                 objects_positions=(),
                 objects_quaternions=(),
                 render_cameras=(),
                 obs_camera='head_cam',
                 render_objects_com=True,
                 training_components=('r_arm', 'torso_yaw'),
                 initial_qpos_path='../config/initial_qpos.yaml',
                 print_done_info=False,
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
        self.world = mjcf.from_path(fullpath)
        self.use_table = use_table
        if self.use_table:
            self.add_table()
        self.objects_positions = objects_positions
        self.objects_quaternions = objects_quaternions
        self.add_ycb_video_objects(objects)
        self.world_entity = composer.ModelWrapperEntity(self.world)
        self.task = composer.NullTask(self.world_entity)
        self.env = composer.Environment(self.task)

        # Set environment and task parameters
        self.obs_from_img = obs_from_img
        self.random_initial_pos = random_initial_pos
        self.frame_skip = frame_skip
        self.steps = 0
        self._max_episode_steps = 2000
        self.render_cameras = render_cameras
        self.obs_camera = obs_camera
        self.render_objects_com = render_objects_com
        self.print_done_info = print_done_info

        # Load initial qpos from yaml file and map joint ids to actuator ids
        with open(initial_qpos_path) as initial_qpos_file:
            self.init_icub_qpos_dict = yaml.load(initial_qpos_file, Loader=yaml.FullLoader)
        self.actuator_names = [actuator.name for actuator in self.world_entity.mjcf_model.find_all('actuator')]
        self.joint_names = [joint.full_identifier for joint in self.world_entity.mjcf_model.find_all('joint')]
        self.init_qpos = np.array([], dtype=np.float32)
        self.init_qvel = np.array([], dtype=np.float32)
        self.joint_ids_icub = np.array([], dtype=np.int64)
        self.joint_ids_objects = np.array([], dtype=np.int64)
        self.joint_names_icub = []
        self.joint_names_objects = []
        for joint_id, joint in enumerate(self.joint_names):
            # Compute the id or the starting id to add, depending on the type of the joint
            if len(self.joint_ids_icub) == 0 and len(self.joint_ids_objects) == 0:
                id_to_add = 0
            elif len(self.joint_ids_icub) == 0:
                id_to_add = np.max(self.joint_ids_objects) + 1
            elif len(self.joint_ids_objects) == 0:
                id_to_add = np.max(self.joint_ids_icub) + 1
            else:
                id_to_add = np.max(np.concatenate((self.joint_ids_icub, self.joint_ids_objects))) + 1

            if joint in self.init_icub_qpos_dict:
                self.init_qpos = np.append(self.init_qpos, self.init_icub_qpos_dict[joint])
                self.init_qvel = np.concatenate((self.init_qvel, np.zeros(1, dtype=np.float32)))
                self.joint_ids_icub = np.append(self.joint_ids_icub, id_to_add)
                self.joint_names_icub.append(joint)
            else:
                joint_id = self.env.physics.model.name2id(joint, 'joint')
                assert (self.world_entity.mjcf_model.find_all('joint')[joint_id].type == 'free')
                if self.world_entity.mjcf_model.find_all('joint')[joint_id].parent.pos is not None:
                    self.init_qpos = np.concatenate((self.init_qpos,
                                                     self.world_entity.mjcf_model.find_all('joint')
                                                     [joint_id].parent.pos))
                else:
                    self.init_qpos = np.concatenate((self.init_qpos, np.zeros(3, dtype=np.float32)))
                if self.world_entity.mjcf_model.find_all('joint')[joint_id].parent.quat is not None:
                    self.init_qpos = np.concatenate((self.init_qpos,
                                                     self.world_entity.mjcf_model.find_all('joint')
                                                     [joint_id].parent.quat))
                else:
                    self.init_qpos = np.concatenate((self.init_qpos, np.zeros(4, dtype=np.float32)))
                self.init_qvel = np.concatenate((self.init_qvel, np.zeros(6, dtype=np.float32)))
                self.joint_ids_objects = np.append(self.joint_ids_objects, np.arange(id_to_add, id_to_add + 7))
                self.joint_names_objects.extend([joint + 'xp',
                                                 joint + 'yp',
                                                 joint + 'zp',
                                                 joint + 'wq',
                                                 joint + 'xq',
                                                 joint + 'yq',
                                                 joint + 'zq'])
        self.map_joint_to_actuators = []
        for actuator in self.actuator_names:
            self.map_joint_to_actuators.append(self.joint_names_icub.index(actuator))

        # Define which icub joints to be used for training
        self.training_components = training_components
        self.joints_to_control = []
        if 'r_arm' in self.training_components:
            self.joints_to_control.extend([j for j in self.joint_names_icub if (j.startswith('r_wrist') or
                                                                                j.startswith('r_elbow') or
                                                                                j.startswith('r_shoulder'))])
        if 'l_arm' in self.training_components:
            self.joints_to_control.extend([j for j in self.joint_names_icub if (j.startswith('l_wrist') or
                                                                                j.startswith('l_elbow') or
                                                                                j.startswith('l_shoulder'))])
        if 'neck' in self.training_components:
            self.joints_to_control.extend([j for j in self.joint_names_icub if j.startswith('neck')])
        if 'torso' in self.training_components:
            self.joints_to_control.extend([j for j in self.joint_names_icub if j.startswith('torso')])
        if 'torso_yaw' in self.training_components and 'torso' not in self.training_components:
            self.joints_to_control.extend([j for j in self.joint_names_icub if j.startswith('torso_yaw')])
        if 'all' in self.training_components and len(self.training_components):
            self.joints_to_control.extend([j for j in self.joint_names_icub])
        self.joints_to_control_ids = np.array([], dtype=np.int64)
        for joint_id, joint_name in enumerate(self.joint_names_icub):
            if joint_name in self.joints_to_control:
                self.joints_to_control_ids = np.append(self.joints_to_control_ids, joint_id)

        # Set spaces
        self.max_delta_qpos = 0.1
        self._set_action_space()
        self._set_observation_space()
        self._set_state_space()

        # Set task parameters
        self.eef_name = 'r_hand'
        self.eef_id_xpos = self.env.physics.model.name2id('r_hand', 'body')
        self.target_eef_pos = np.array([-0.3, 0.1, 1.01])
        self.goal_xpos_tolerance = 0.05

        # Set reward values
        self.reward_goal = reward_goal
        self.reward_out_of_joints = reward_out_of_joints
        self.reward_single_step_multiplier = reward_single_step_multiplier

        # Reset environment
        self.reset()
        self.env.reset()
        self.eef_pos = self.env.physics.data.xpos[self.eef_id_xpos].copy()

    def _set_action_space(self):
        n_joints_to_control = len(self.joints_to_control)
        self.action_space = gym.spaces.Box(low=-self.max_delta_qpos,
                                           high=self.max_delta_qpos,
                                           shape=(n_joints_to_control,),
                                           dtype=np.float32)

    def _set_observation_space(self):
        if self.obs_from_img:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype='uint8')
        else:
            # Use as observation only iCub joints
            bounds = np.concatenate([np.expand_dims(joint.range, 0) if joint.name in self.init_icub_qpos_dict.keys()
                                     else np.empty([0, 2], dtype=np.float32)
                                     for joint in self.world_entity.mjcf_model.find_all('joint')],
                                    axis=0,
                                    dtype=np.float32)
            low = bounds[:, 0][self.joints_to_control_ids]
            high = bounds[:, 1][self.joints_to_control_ids]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_state_space(self):
        bounds = np.empty([0, 2], dtype=np.float32)
        for joint in self.world_entity.mjcf_model.find_all('joint'):
            assert (joint.type == "free" or joint.type == "hinge" or joint.type is None)
            if joint.range is not None:
                bounds = np.concatenate([bounds, np.expand_dims(joint.range, 0)], axis=0, dtype=np.float32)
            else:
                if joint.type == "free":
                    if self.use_table:
                        bounds = np.concatenate([bounds,
                                                 np.array([[-1.08, -0.28]], dtype=np.float32),
                                                 np.array([[-0.9, 0.9]], dtype=np.float32),
                                                 np.array([[0.95, 1.2]], dtype=np.float32),
                                                 np.array([[-1.0, 1.0]], dtype=np.float32),
                                                 np.array([[-1.0, 1.0]], dtype=np.float32),
                                                 np.array([[-1.0, 1.0]], dtype=np.float32),
                                                 np.array([[-1.0, 1.0]], dtype=np.float32)],
                                                axis=0,
                                                dtype=np.float32)
                    else:
                        bounds = np.concatenate([bounds, np.full((7, 2),
                                                                 np.array([-np.inf, np.inf]), dtype=np.float32)],
                                                axis=0,
                                                dtype=np.float32)
                else:
                    bounds = np.concatenate([bounds, np.array([[-np.inf, np.inf]], dtype=np.float32)],
                                            axis=0,
                                            dtype=np.float32)

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
            # Prevent simulation unstabillity
            try:
                self.env.step(ctrl[self.map_joint_to_actuators])
            except:
                print('Simulation unstable, environment reset.')
        self.steps += 1

    def _get_obs(self):
        self.render()
        if self.obs_from_img:
            return self.env.physics.render(height=480, width=640, camera_id=self.obs_camera)
        else:
            return self.get_state()[:len(self.init_qpos)][self.joints_to_control_ids]

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        if self.random_initial_pos:
            random_pos = self.state_space.sample()[self.joints_to_control_ids]
            self.init_qpos[self.joints_to_control_ids] = random_pos
            random_pos = self.state_space.sample()[self.joint_ids_objects]
            # Force z_objects > z_table and normalize quaternions
            for i in range(int(len(random_pos) / 7)):
                random_pos[i * 7 + 2] = np.maximum(random_pos[i * 7 + 2],
                                                   self.state_space.low[self.joint_ids_objects[i * 7 + 2]] + 0.1)
                random_pos[i * 7 + 3:i * 7 + 7] /= np.linalg.norm(random_pos[i * 7 + 3:i * 7 + 7])
            self.init_qpos[self.joint_ids_objects] = random_pos
        self.set_state(np.concatenate([self.init_qpos.copy(), self.init_qvel.copy(), self.env.physics.data.act]))
        self.env.physics.forward()
        self.eef_pos = self.env.physics.data.xpos[self.eef_id_xpos].copy()
        return self._get_obs()

    def joints_out_of_range(self):
        joints_out_of_range = []
        if not self.state_space.contains(self.env.physics.data.qpos):
            for i in self.joint_ids_icub:
                if self.env.physics.data.qpos[i] < self.state_space.low[i] or \
                        self.env.physics.data.qpos[i] > self.state_space.high[i]:
                    joints_out_of_range.append(self.joint_names_icub[i])
        return joints_out_of_range

    def falling_object(self):
        for list_id, joint_id in enumerate(self.joint_ids_objects):
            # Check only the z component
            if list_id % 7 != 2:
                continue
            else:
                if self.env.physics.data.qpos[joint_id] < self.state_space.low[joint_id]:
                    return True
        return False

    def render(self, mode='human'):
        del mode  # Unused
        for cam in self.render_cameras:
            img = np.array(self.env.physics.render(height=480, width=640, camera_id=cam), dtype=np.uint8)
            if cam == 'head_cam' and self.render_objects_com:
                objects_com_x_y_z = []
                for i in range(int(len(self.joint_ids_objects) / 7)):
                    objects_com_x_y_z.append(self.env.physics.data.qpos[self.joint_ids_objects[i*7:i*7+3]])
                com_uvs = self.points_in_pixel_coord(objects_com_x_y_z)
                for com_uv in com_uvs:
                    img = cv2.circle(img, com_uv, 5, (0, 255, 0), -1)
            cv2.imshow(cam, img[:, :, ::-1])
            cv2.waitKey(1)

    def add_ycb_video_objects(self, object_names):
        for obj_id, obj in enumerate(object_names):
            obj_path = "../meshes/YCB_Video/{}.xml".format(obj)
            obj_mjcf = mjcf.from_path(obj_path, escape_separators=True)
            self.world.attach(obj_mjcf.root_model)
            self.world.worldbody.body[len(self.world.worldbody.body) - 1].pos = \
                self.objects_positions[obj_id] if self.objects_positions else np.array([np.random.rand() - 1.18,
                                                                                        np.random.rand() * 2 - 1.0,
                                                                                        1.2])
            if self.objects_quaternions:
                object_quaternions = self.objects_quaternions[obj_id]
            else:
                object_quaternions = np.array([np.random.rand() * 2 - 1.0,
                                               np.random.rand() * 2 - 1.0,
                                               np.random.rand() * 2 - 1.0,
                                               np.random.rand() * 2 - 1.0])
                object_quaternions /= np.linalg.norm(object_quaternions)
            self.world.worldbody.body[len(self.world.worldbody.body) - 1].quat = object_quaternions
            self.world.worldbody.body[len(self.world.worldbody.body) - 1].add('joint',
                                                                              name=obj,
                                                                              type="free",
                                                                              pos="0 0 0",
                                                                              limited="false",
                                                                              damping="0.0",
                                                                              stiffness="0.01")

    def add_table(self):
        table_path = "../models/table.xml"
        table_mjcf = mjcf.from_path(table_path, escape_separators=False)
        self.world.attach(table_mjcf.root_model)

    def points_in_pixel_coord(self, points):
        com_uvs = []
        for point in points:
            # Point roto-translation matrix in world coordinates
            p_world = np.array([[1, 0, 0, point[0]],
                                [0, 1, 0, point[1]],
                                [0, 0, 1, point[2]],
                                [0, 0, 0, 1]],
                               dtype=np.float32)
            # Camera roto-translation matrix in world coordinates
            cam_id = self.env.physics.model.name2id('head_cam', 'camera')
            cam_world = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]],
                                 dtype=np.float32)
            cam_pos = self.env.physics.named.data.cam_xpos[cam_id, :]
            cam_world[:3, -1] = cam_pos
            cam_rot = np.reshape(self.env.physics.named.data.cam_xmat[cam_id, :], (3, 3))
            cam_world[:3, :3] = cam_rot
            # Point roto-translation matrix in camera coordinates
            p_cam = np.matmul(np.linalg.inv(cam_world), p_world)
            # Focal length set used to compute fovy in the xml file
            fy = 617.783447265625
            # Pixel coordinates computation
            x = p_cam[0, 3]/(-p_cam[2, 3])*fy
            y = p_cam[1, 3]/(-p_cam[2, 3])*fy
            u = int(x) + 320
            v = -int(y) + 240
            com_uvs.append(np.array([u, v]))
        return com_uvs
