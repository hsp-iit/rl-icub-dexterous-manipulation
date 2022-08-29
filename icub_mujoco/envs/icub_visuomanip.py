import math
from dm_control import composer, mjcf
import os
import gym
import numpy as np
import cv2
import yaml
from icub_mujoco.feature_extractors.images_feature_extractor import ImagesFeatureExtractor
from icub_mujoco.feature_extractors.images_feature_extractor_CLIP import ImagesFeatureExtractorCLIP
from pyquaternion import Quaternion


class ICubEnv(gym.Env):

    def __init__(self,
                 model_path,
                 frame_skip=5,
                 icub_observation_space=('joints',),
                 random_initial_pos=True,
                 ycb_video_graspable_objects_config_path='../config/ycb_video_objects_graspable_poses.yaml',
                 random_ycb_video_graspable_object=False,
                 objects=(),
                 use_table=True,
                 objects_positions=(),
                 objects_quaternions=(),
                 randomly_rotate_object_z_axis=False,
                 eef_name='r_hand',
                 render_cameras=(),
                 obs_camera='head_cam',
                 track_object=False,
                 superquadrics_camera='head_cam',
                 feature_extractor_model_name='alexnet',
                 render_objects_com=False,
                 training_components=('r_arm', 'torso_yaw'),
                 ik_components=(),
                 cartesian_components=('all_ypr',),
                 initial_qpos_path='../config/initial_qpos_actuated_hand.yaml',
                 print_done_info=False,
                 reward_goal=1.0,
                 reward_out_of_joints=-1.0,
                 reward_end_timesteps=-1.0,
                 reward_single_step_multiplier=10.0,
                 reward_dist_superq_center=False,
                 goal_reached_only_with_lift_refine_grasp=False,
                 joints_margin=0.0,
                 null_reward_out_image=False,
                 done_if_joints_out_of_limits=True,
                 do_not_consider_done_z_pos=False,
                 lift_object_height=1.02,
                 curriculum_learning=False,
                 learning_from_demonstration=False,
                 max_lfd_steps=10000,
                 max_delta_qpos=0.1,
                 lfd_keep_only_successful_episodes=False,
                 lfd_with_approach=False,
                 pregrasp_distance_from_grasp_pose=0.05,
                 max_delta_cartesian_pos=0.02,
                 max_delta_cartesian_rot=0.1,
                 distanced_superq_grasp_pose=False,
                 control_gaze=False,
                 ik_solver='idyntree',
                 use_only_right_hand_model=False
                 ):

        # Load xml model
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        # Check that the correct model is loaded when using ikin as IK solver
        if ik_solver == 'ikin' and 'ikin' not in fullpath:
            raise ValueError('The selected IK solver is ikin, but the selected iCub model uses the urdf joint limits.')

        # Set whether the whole iCub model or only the right hand must be used
        self.use_only_right_hand_model = use_only_right_hand_model

        # Initialize dm_control environment
        self.world = mjcf.from_path(fullpath)
        self.use_table = use_table
        if self.use_table:
            self.add_table()
        self.random_ycb_video_graspable_object = random_ycb_video_graspable_object
        with open(ycb_video_graspable_objects_config_path) as ycb_video_graspable_objects_config_file:
            self.ycb_video_graspable_objects_cfg = yaml.load(ycb_video_graspable_objects_config_file,
                                                             Loader=yaml.FullLoader)
        self.ycb_video_graspable_objects_list = []
        for obj in self.ycb_video_graspable_objects_cfg:
            for graspability in self.ycb_video_graspable_objects_cfg[obj]['graspable']:
                if self.ycb_video_graspable_objects_cfg[obj]['graspable'][graspability] == 'yes':
                    self.ycb_video_graspable_objects_list.append(
                        {'object': obj,
                         'position': self.ycb_video_graspable_objects_cfg[obj]['initial_positions'][graspability],
                         'orientation': self.ycb_video_graspable_objects_cfg[obj]['initial_orientations'][graspability],
                         'moved_object_height': self.ycb_video_graspable_objects_cfg[obj]['moved_object_height']})
        self.objects_positions = objects_positions
        self.objects_quaternions = objects_quaternions
        self.randomly_rotate_object_z_axis = randomly_rotate_object_z_axis
        self.objects = objects
        self.moved_object_height = 0.98
        self.add_ycb_video_objects(self.objects, remove_last_object=False)
        self.track_object = track_object
        if self.track_object:
            self.track_object_with_camera()
        self.world_entity = composer.ModelWrapperEntity(self.world)
        self.task = composer.NullTask(self.world_entity)
        self.env = composer.Environment(self.task)

        # Set environment and task parameters
        self.icub_observation_space = icub_observation_space
        if 'features' in icub_observation_space or 'flare' in icub_observation_space:
            if 'CLIP' in feature_extractor_model_name:
                self.feature_extractor = ImagesFeatureExtractorCLIP(model_name=feature_extractor_model_name)
            else:
                self.feature_extractor = ImagesFeatureExtractor(model_name=feature_extractor_model_name)
        self.random_initial_pos = random_initial_pos
        self.frame_skip = frame_skip
        self.steps = 0
        self._max_episode_steps = 2000
        self.render_cameras = render_cameras
        self.obs_camera = obs_camera
        self.superquadrics_camera = superquadrics_camera
        self.render_objects_com = render_objects_com
        self.print_done_info = print_done_info
        self.joints_margin = joints_margin
        self.null_reward_out_image = null_reward_out_image
        self.lift_object_height = lift_object_height
        if random_ycb_video_graspable_object:
            print('The value of lift_object_height will be overwritten whenever a new object will be added to the'
                  'environment.')
        self.curriculum_learning = curriculum_learning

        # Set if using the original superquadric grasp pose or the distanced pose
        self.distanced_superq_grasp_pose = distanced_superq_grasp_pose

        # Focal length set used to compute fovy in the xml file
        self.fy = 617.783447265625

        # Load initial qpos from yaml file and map joint ids to actuator ids
        with open(initial_qpos_path) as initial_qpos_file:
            self.init_icub_act_dict = yaml.load(initial_qpos_file, Loader=yaml.FullLoader)
        self.actuator_names = [actuator.name for actuator in self.world_entity.mjcf_model.find_all('actuator')]
        self.joint_names = [joint.full_identifier for joint in self.world_entity.mjcf_model.find_all('joint')]

        # Select actuators to control for the task
        if 'cartesian' in self.icub_observation_space:
            self.training_components = []
            for tr_component in training_components:
                if 'r_hand' in tr_component:
                    self.training_components.append('r_hand')
                elif 'l_hand' in tr_component:
                    self.training_components.append('l_hand')
                else:
                    print('Using cartesian observation space. {} cannot be used as training component.'.format(
                        tr_component))
            self.cartesian_components = cartesian_components
            self.cartesian_orientation = 'ypr'
            if 'all_ypr' in self.cartesian_components:
                self.cartesian_ids = list(range(6))
            elif 'all_quaternion' in self.cartesian_components:
                self.cartesian_orientation = 'quaternion'
                self.cartesian_ids = list(range(7))
            elif 'yaw' in self.cartesian_components \
                    or 'pitch' in self.cartesian_components or 'roll' in self.cartesian_components:
                self.cartesian_ids = []
                if 'x' in self.cartesian_components:
                    self.cartesian_ids.append(0)
                if 'y' in self.cartesian_components:
                    self.cartesian_ids.append(1)
                if 'z' in self.cartesian_components:
                    self.cartesian_ids.append(2)
                if 'yaw' in self.cartesian_components:
                    self.cartesian_ids.append(3)
                if 'pitch' in self.cartesian_components:
                    self.cartesian_ids.append(4)
                if 'roll' in self.cartesian_components:
                    self.cartesian_ids.append(5)
            else:
                self.cartesian_orientation = 'quaternion'
                self.cartesian_ids = []
                if 'x' in self.cartesian_components:
                    self.cartesian_ids.append(0)
                if 'y' in self.cartesian_components:
                    self.cartesian_ids.append(1)
                if 'z' in self.cartesian_components:
                    self.cartesian_ids.append(2)
                if 'qw' in self.cartesian_components:
                    self.cartesian_ids.append(3)
                if 'qx' in self.cartesian_components:
                    self.cartesian_ids.append(4)
                if 'qy' in self.cartesian_components:
                    self.cartesian_ids.append(5)
                if 'qz' in self.cartesian_components:
                    self.cartesian_ids.append(6)
        else:
            self.training_components = training_components
        self.actuators_to_control = []
        if 'r_arm' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('r_wrist') or
                                                                                 j.startswith('r_elbow') or
                                                                                 j.startswith('r_shoulder'))])
        if 'r_wrist' in self.training_components and 'r_arm' not in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('r_wrist')])
        if 'r_hand' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('r_hand') or
                                                                                 j.startswith('r_thumb') or
                                                                                 j.startswith('r_index') or
                                                                                 j.startswith('r_middle') or
                                                                                 j.startswith('r_pinky'))])
        if 'l_arm' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('l_wrist') or
                                                                                 j.startswith('l_elbow') or
                                                                                 j.startswith('l_shoulder'))])
        if 'l_wrist' in self.training_components and 'l_arm' not in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('l_wrist')])
        if 'l_hand' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('l_hand') or
                                                                                 j.startswith('l_thumb') or
                                                                                 j.startswith('l_index') or
                                                                                 j.startswith('l_middle') or
                                                                                 j.startswith('l_pinky'))])
        if 'neck' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('neck')])
        if 'torso' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('torso')])
        if 'torso_yaw' in self.training_components and 'torso' not in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('torso_yaw')])
        if 'torso_pitch' in self.training_components and 'torso' not in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if j.startswith('torso_pitch')])
        if 'all' in self.training_components and len(self.training_components) == 1:
            self.actuators_to_control.extend([j for j in self.actuator_names])

        self.actuators_to_control_no_fingers = []
        for actuator in self.actuators_to_control:
            if not ('hand' in actuator or 'thumb' in actuator or 'index' in actuator
                    or 'middle' in actuator or 'pinky' in actuator):
                self.actuators_to_control_no_fingers.append(actuator)

        # Select actuators to control for IK
        self.ik_components = ik_components
        self.joints_to_control_ik = []
        if 'r_arm' in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if (j.startswith('r_wrist') or
                                                                              j.startswith('r_elbow') or
                                                                              j.startswith('r_shoulder'))])
        if 'l_arm' in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if (j.startswith('l_wrist') or
                                                                              j.startswith('l_elbow') or
                                                                              j.startswith('l_shoulder'))])
        if 'neck' in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if j.startswith('neck')])
        if 'torso' in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if j.startswith('torso')])
        if 'torso_yaw' in self.ik_components and 'torso' not in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if j.startswith('torso_yaw')])
        if 'torso_pitch' in self.ik_components and 'torso' not in self.ik_components:
            self.joints_to_control_ik.extend([j for j in self.joint_names if j.startswith('torso_pitch')])
        if 'all' in self.ik_components and len(self.ik_components) == 1:
            self.joints_to_control_ik.extend([j for j in self.joint_names])

        # Create a list of actuators to control for inverse kinematics if using the cartesian controller
        if 'cartesian' in self.icub_observation_space:
            self.actuators_to_control_ik = []
            if 'r_arm' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if (j.startswith('r_wrist') or
                                                                                        j.startswith('r_elbow') or
                                                                                        j.startswith('r_shoulder'))])
            if 'r_wrist' in self.ik_components and 'r_arm' not in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('r_wrist')])
            if 'r_hand' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if (j.startswith('r_hand') or
                                                                                        j.startswith('r_thumb') or
                                                                                        j.startswith('r_index') or
                                                                                        j.startswith('r_middle') or
                                                                                        j.startswith('r_pinky'))])
            if 'l_arm' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if (j.startswith('l_wrist') or
                                                                                        j.startswith('l_elbow') or
                                                                                        j.startswith('l_shoulder'))])
            if 'l_wrist' in self.ik_components and 'l_arm' not in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('l_wrist')])
            if 'l_hand' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if (j.startswith('l_hand') or
                                                                                        j.startswith('l_thumb') or
                                                                                        j.startswith('l_index') or
                                                                                        j.startswith('l_middle') or
                                                                                        j.startswith('l_pinky'))])
            if 'neck' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('neck')])
            if 'torso' in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('torso')])
            if 'torso_yaw' in self.ik_components and 'torso' not in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('torso_yaw')])
            if 'torso_pitch' in self.ik_components and 'torso' not in self.ik_components:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names if j.startswith('torso_pitch')])
            if 'all' in self.ik_components and len(self.ik_components) == 1:
                self.actuators_to_control_ik.extend([j for j in self.actuator_names])

        # Set IK solver
        if ik_solver in ('idyntree', 'dm_robotics', 'dm_control', 'ikin'):
            self.ik_solver = ik_solver
        else:
            print('The required IK solver is not avalable. Using idyntree.')
            self.ik_solver = 'idyntree'

        # Extract joints-tendons information for each actuator
        self.init_icub_qpos_dict = {}
        self.actuators_to_control_dict = []
        self.actuators_dict = []
        self.actuators_to_control_ids = []
        self.actuators_to_control_no_fingers_ids = []
        self.actuators_to_control_fingers_ids = []
        self.actuators_to_control_ik_ids = []
        self.actuators_to_control_gaze_controller_ids = [-1, -1, -1]
        self.init_icub_act = np.array([], dtype=np.float32)
        self.joints_to_control_icub = []
        self.actuators_margin = np.array([], dtype=np.float32)
        for actuator_id, actuator in enumerate(self.world_entity.mjcf_model.find_all('actuator')):
            if actuator.joint is not None:
                actuator_dict = {'name': actuator.name,
                                 'jnt': [actuator.joint.full_identifier],
                                 'coeff': [1.0],
                                 'non_zero_coeff': 1}
                if 'thumb' in actuator.name or 'index' in actuator.name or 'middle' in actuator.name \
                        or 'pinky' in actuator.name or 'hand_finger' in actuator.name:
                    actuator_dict['close_value'] = self.env.physics.named.model.actuator_ctrlrange[actuator.name].max()
                    if actuator.name != 'r_thumb_oppose':
                        actuator_dict['open_value'] = \
                            self.env.physics.named.model.actuator_ctrlrange[actuator.name].min()
                    else:
                        actuator_dict['open_value'] = \
                            self.env.physics.named.model.actuator_ctrlrange[actuator.name].max(),
                self.init_icub_qpos_dict[actuator.joint.full_identifier] = self.init_icub_act_dict[actuator.name]
                if actuator.name in self.actuators_to_control:
                    self.joints_to_control_icub.append(actuator.joint.full_identifier)
                self.actuators_margin = np.append(self.actuators_margin, self.joints_margin)
            elif actuator.tendon is not None:
                jnt = []
                coeff = []
                non_zero_coeffs = len([joint.coef for joint in actuator.tendon.joint if joint.coef != 0.0])
                actuator_margin = 0
                for joint in actuator.tendon.joint:
                    jnt.append(joint.joint.full_identifier)
                    coeff.append(joint.coef)
                    if joint.coef != 0.0:
                        self.init_icub_qpos_dict[joint.joint.full_identifier] = \
                            self.init_icub_act_dict[actuator.name] / (joint.coef * non_zero_coeffs)
                        actuator_margin += self.joints_margin * joint.coef
                    else:
                        self.init_icub_qpos_dict[joint.joint.full_identifier] = 0.0
                    if actuator.name in self.actuators_to_control:
                        self.joints_to_control_icub.append(joint.joint.full_identifier)
                actuator_dict = {'name': actuator.name,
                                 'jnt': jnt,
                                 'coeff': coeff,
                                 'non_zero_coeff': non_zero_coeffs}
                if 'thumb' in actuator.name or 'index' in actuator.name or 'middle' in actuator.name \
                        or 'pinky' in actuator.name or 'hand_finger' in actuator.name:
                    actuator_dict['close_value'] = self.env.physics.named.model.actuator_ctrlrange[actuator.name].max()
                    actuator_dict['open_value'] = self.env.physics.named.model.actuator_ctrlrange[actuator.name].min()
                self.actuators_margin = np.append(self.actuators_margin, abs(actuator_margin))
            else:
                raise ValueError('Actuator {} has neither a joint nor a tendon.'.format(actuator.name))
            if actuator.name in self.actuators_to_control:
                self.actuators_to_control_dict.append(actuator_dict)
                self.actuators_to_control_ids.append(actuator_id)
                if actuator.name not in self.actuators_to_control_no_fingers:
                    self.actuators_to_control_fingers_ids.append(actuator_id)
            if actuator.name in self.actuators_to_control_no_fingers:
                self.actuators_to_control_no_fingers_ids.append(actuator_id)
            if 'neck' in actuator.name:
                if 'pitch' in actuator.name:
                    self.actuators_to_control_gaze_controller_ids[0] = actuator_id
                if 'roll' in actuator.name:
                    self.actuators_to_control_gaze_controller_ids[1] = actuator_id
                if 'yaw' in actuator.name:
                    self.actuators_to_control_gaze_controller_ids[2] = actuator_id
            if 'cartesian' in self.icub_observation_space:
                if actuator.name in self.actuators_to_control_ik:
                    self.actuators_to_control_ik_ids.append(actuator_id)
            self.actuators_dict.append(actuator_dict)
            self.init_icub_act = np.append(self.init_icub_act, self.init_icub_act_dict[actuator.name])

        # Add non-actuated iCub joints
        for joint in self.world_entity.mjcf_model.find_all('joint'):
            icub_joint = False
            jnt = joint
            while jnt.parent is not None and not icub_joint:
                if jnt.parent.full_identifier == 'icub' or jnt.parent.full_identifier == 'icub_r_hand':
                    icub_joint = True
                    if joint.full_identifier not in self.init_icub_qpos_dict:
                        self.init_icub_qpos_dict[joint.full_identifier] = 0.0
                    if joint.type == "free":
                        self.init_icub_qpos_dict[joint.full_identifier] = \
                            self.init_icub_act_dict[joint.full_identifier]
                jnt = jnt.parent

        # Set initial configuration for each joint
        self.init_qpos = np.array([], dtype=np.float32)
        self.init_qvel = np.array([], dtype=np.float32)
        self.joint_ids_icub = np.array([], dtype=np.int64)
        self.joint_ids_icub_free = np.array([], dtype=np.int64)
        self.joint_ids_icub_dict = {}
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
                joint_id = self.env.physics.model.name2id(joint, 'joint')
                if self.world_entity.mjcf_model.find_all('joint')[joint_id].type == "hinge" \
                        or self.world_entity.mjcf_model.find_all('joint')[joint_id].type is None:
                    self.init_qpos = np.append(self.init_qpos, self.init_icub_qpos_dict[joint])
                    self.init_qvel = np.concatenate((self.init_qvel, np.zeros(1, dtype=np.float32)))
                    self.joint_ids_icub = np.append(self.joint_ids_icub, id_to_add)
                    self.joint_ids_icub_dict[joint] = id_to_add
                    self.joint_names_icub.append(joint)
                if self.world_entity.mjcf_model.find_all('joint')[joint_id].type == "free":
                    self.init_qpos = np.append(self.init_qpos, self.init_icub_qpos_dict[joint])
                    self.init_qvel = np.concatenate((self.init_qvel, np.zeros(6, dtype=np.float32)))
                    self.joint_ids_icub = np.append(self.joint_ids_icub, np.arange(id_to_add, id_to_add + 7))
                    self.joint_ids_icub_free = np.append(self.joint_ids_icub_free, np.arange(id_to_add, id_to_add + 7))
                    self.joint_ids_icub_dict[joint] = np.arange(id_to_add, id_to_add + 7)
                    self.joint_names_icub.extend([joint + 'xp',
                                                  joint + 'yp',
                                                  joint + 'zp',
                                                  joint + 'wq',
                                                  joint + 'xq',
                                                  joint + 'yq',
                                                  joint + 'zq'])
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

        # Compute ids of iCub joints to control
        self.joints_to_control_ids = np.array([], dtype=np.int64)
        # Store joints ids to control excluding hand joints, just for random initialization purpose
        self.joints_to_control_no_hand_ids = np.array([], dtype=np.int64)
        self.joints_to_control_ik_ids = np.array([], dtype=np.int64)
        self.joints_to_control_ik_sorted = []

        for id, joint in enumerate(self.joint_names_icub):
            if joint in self.joints_to_control_icub:
                self.joints_to_control_ids = np.append(self.joints_to_control_ids, id)
                if not joint.startswith('r_hand') and not joint.startswith('l_hand'):
                    self.joints_to_control_no_hand_ids = np.append(self.joints_to_control_no_hand_ids, id)
            if joint in self.joints_to_control_ik:
                self.joints_to_control_ik_ids = np.append(self.joints_to_control_ik_ids, id)
                self.joints_to_control_ik_sorted.append(joint)

        # Set if controlling gaze
        self.control_gaze = control_gaze

        # Compute fingers-objects touch variables
        self.contact_geom_fingers_names = ['col_RightThumb3',
                                           'col_RightIndex3',
                                           'col_RightMiddle3',
                                           'col_RightRing3',
                                           'col_RightLittle3']
        self.contact_geom_ids_fingers_meshes = {'col_RightThumb3': -1,
                                                'col_RightIndex3': -1,
                                                'col_RightMiddle3': -1,
                                                'col_RightRing3': -1,
                                                'col_RightLittle3': -1}
        self.contact_geom_ids_objects_meshes = {}
        self.geoms = self.world_entity.mjcf_model.find_all('geom')
        for geom in self.geoms:
            if geom.mesh:
                if geom.mesh.name in self.contact_geom_ids_fingers_meshes.keys():
                    self.contact_geom_ids_fingers_meshes[geom.mesh.name] = \
                        self.env.physics.model.name2id(geom.full_identifier, 'geom')
            if geom.name == 'mesh_' + self.objects[0] + '_00_collision':
                self.contact_geom_ids_objects_meshes[geom.name] = \
                    self.env.physics.model.name2id(geom.full_identifier, 'geom')
        self.number_of_contacts = 0
        self.previous_number_of_contacts = 0
        self.fingers_touching_object = []

        self.flare_features = []

        # Set spaces
        self.max_delta_qpos = max_delta_qpos
        self.max_delta_cartesian_pos = max_delta_cartesian_pos
        self.max_delta_cartesian_rot = max_delta_cartesian_rot
        self._set_action_space()
        self._set_action_space_with_touch()
        self._set_observation_space()
        self._set_state_space()
        self._set_actuators_space()

        # Set curriculum learning parameters
        self.cartesian_actions_curriculum_learning = np.empty(0)
        if self.curriculum_learning:
            if 'cartesian' in self.icub_observation_space:
                self.cartesian_actions_curriculum_learning = np.ones(len(self.cartesian_ids))
            for act in self.actuators_to_control_ids:
                if act in self.actuators_to_control_no_fingers_ids:
                    self.cartesian_actions_curriculum_learning = np.append(self.cartesian_actions_curriculum_learning,
                                                                           1)
                else:
                    self.cartesian_actions_curriculum_learning = np.append(self.cartesian_actions_curriculum_learning,
                                                                           0)
            self.cartesian_actions_curriculum_learning = np.where(self.cartesian_actions_curriculum_learning > 0)

        # Set learning from demonstration parameters
        self.learning_from_demonstration = learning_from_demonstration
        self.learning_from_demonstration_max_steps = max_lfd_steps
        self.lfd_keep_only_successful_episodes = lfd_keep_only_successful_episodes and self.learning_from_demonstration
        self.lfd_with_approach = lfd_with_approach
        self.pregrasp_distance_from_grasp_pose = pregrasp_distance_from_grasp_pose

        # Set task parameters
        self.eef_name = eef_name
        self.eef_id_xpos = self.env.physics.model.name2id(eef_name, 'body')
        self.target_eef_pos = np.array([-0.3, 0.1, 1.01])
        self.goal_xpos_tolerance = 0.05
        self.done_if_joints_out_of_limits = done_if_joints_out_of_limits
        self.do_not_consider_done_z_pos = do_not_consider_done_z_pos

        # Set reward values
        self.reward_goal = reward_goal
        self.reward_out_of_joints = reward_out_of_joints
        self.reward_single_step_multiplier = reward_single_step_multiplier
        self.reward_end_timesteps = reward_end_timesteps
        self.reward_dist_superq_center = reward_dist_superq_center
        self.goal_reached_only_with_lift_refine_grasp = goal_reached_only_with_lift_refine_grasp

        # Reset environment
        self.reset()
        self.env.reset()
        self.eef_pos = self.env.physics.data.xpos[self.eef_id_xpos].copy()

    def _set_action_space(self):
        if 'cartesian' in self.icub_observation_space:
            low = np.array([])
            high = np.array([])
            for cartesian_id in self.cartesian_ids:
                if cartesian_id <= 2:
                    low = np.append(low, -self.max_delta_cartesian_pos)
                    high = np.append(high, self.max_delta_cartesian_pos)
                else:
                    low = np.append(low, -self.max_delta_cartesian_rot)
                    high = np.append(high, self.max_delta_cartesian_rot)
        else:
            low = np.array([])
            high = np.array([])
        for act in self.actuators_to_control:
            if 'pinky' in act:
                low = np.append(low, -2 * self.max_delta_qpos)
                high = np.append(high, 2 * self.max_delta_qpos)
            else:
                low = np.append(low, -self.max_delta_qpos)
                high = np.append(high, self.max_delta_qpos)
        self.action_space = gym.spaces.Box(low=low.astype(np.float32),
                                           high=high.astype(np.float32),
                                           dtype=np.float32)

    def _set_action_space_with_touch(self):
        if 'cartesian' in self.icub_observation_space:
            low = np.array([])
            high = np.array([])
            for cartesian_id in self.cartesian_ids:
                if cartesian_id <= 2:
                    low = np.append(low, -self.max_delta_cartesian_pos)
                    high = np.append(high, self.max_delta_cartesian_pos)
                else:
                    low = np.append(low, -self.max_delta_cartesian_rot)
                    high = np.append(high, self.max_delta_cartesian_rot)
        else:
            low = np.array([])
            high = np.array([])
        for act in self.actuators_to_control:
            low = np.append(low, -self.max_delta_qpos)
            if 'proximal' in act or 'distal' in act or 'pinky' in act:
                high = np.append(high, np.inf)
            else:
                high = np.append(high, self.max_delta_qpos)
        self.action_space_with_touch = gym.spaces.Box(low=low.astype(np.float32),
                                                      high=high.astype(np.float32),
                                                      dtype=np.float32)

    def _set_observation_space(self):
        if len(self.icub_observation_space) > 1:
            obs_space = {}
            for space in self.icub_observation_space:
                if space == 'camera':
                    obs_space['camera'] = gym.spaces.Box(low=0,
                                                         high=255,
                                                         shape=(480, 640, 3),
                                                         dtype='uint8')
                elif space == 'joints':
                    bounds = np.concatenate(
                        [np.expand_dims(actuator.ctrlrange, 0) if actuator.name in self.actuators_to_control
                         else np.empty([0, 2], dtype=np.float32)
                         for actuator in self.world_entity.mjcf_model.find_all('actuator')],
                        axis=0,
                        dtype=np.float32)
                    low = bounds[:, 0]
                    high = bounds[:, 1]
                    obs_space['joints'] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                elif space == 'cartesian':
                    obs_space['cartesian'] = gym.spaces.Box(low=-np.inf,
                                                            high=np.inf,
                                                            shape=[len(self.cartesian_ids)],
                                                            dtype=np.float32)
                elif space == 'features':
                    obs_space['features'] = gym.spaces.Box(low=-np.inf,
                                                           high=np.inf,
                                                           shape=self.feature_extractor.output_features_dimension,
                                                           dtype=np.float32)
                elif space == 'touch':
                    obs_space['touch'] = gym.spaces.Box(low=-np.inf,
                                                        high=np.inf,
                                                        shape=[len(self.contact_geom_ids_fingers_meshes)],
                                                        dtype=np.float32)
                # https://arxiv.org/pdf/2101.01857.pdf
                elif space == 'flare':
                    flare_shape = np.array(self.feature_extractor.output_features_dimension)
                    flare_shape[1] = flare_shape[1] * 5
                    obs_space['flare'] = gym.spaces.Box(low=-np.inf,
                                                        high=np.inf,
                                                        shape=flare_shape,
                                                        dtype=np.float32)
                elif space == 'superquadric_center':
                    obs_space['superquadric_center'] = gym.spaces.Box(low=-np.inf,
                                                                      high=np.inf,
                                                                      shape=[3],
                                                                      dtype=np.float32)
                elif space == 'object_pose':
                    if len(self.objects) != 1:
                        raise ValueError('There must be one and only one objects in the environment. Quitting.')
                    obs_space['object_pose'] = gym.spaces.Box(low=-np.inf,
                                                              high=np.inf,
                                                              shape=[7],
                                                              dtype=np.float32)
            self.observation_space = gym.spaces.Dict(obs_space)
        elif 'camera' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype='uint8')
        elif 'joints' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            # Use as observation only iCub joints
            bounds = np.concatenate(
                [np.expand_dims(actuator.ctrlrange, 0) if actuator.name in self.actuators_to_control
                 else np.empty([0, 2], dtype=np.float32)
                 for actuator in self.world_entity.mjcf_model.find_all('actuator')],
                axis=0,
                dtype=np.float32)
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif 'cartesian' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=[len(self.cartesian_ids)],
                                                    dtype=np.float32)
        elif 'features' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            # Use as observation features from images
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=self.feature_extractor.output_features_dimension,
                                                    dtype=np.float32)
        elif 'touch' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=[len(self.contact_geom_ids_fingers_meshes)],
                                                    dtype=np.float32)
        # https://arxiv.org/pdf/2101.01857.pdf
        elif 'flare' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            flare_shape = np.array(self.feature_extractor.output_features_dimension)
            flare_shape[1] = flare_shape[1] * 5
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=flare_shape,
                                                    dtype=np.float32)
        elif 'object_pose' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            if len(self.objects) != 1:
                raise ValueError('There must be one and only one objects in the environment. Quitting.')
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=[7],
                                                    dtype=np.float32)
        else:
            raise ValueError('The observation spaces must be of type joints, camera or features. Quitting.')

    def _set_state_space(self):
        bounds = np.empty([0, 2], dtype=np.float32)
        for joint in self.world_entity.mjcf_model.find_all('joint'):
            assert (joint.type == "free" or joint.type == "hinge" or joint.type is None)
            if joint.range is not None:
                bounds = np.concatenate([bounds, np.expand_dims(joint.range, 0)], axis=0, dtype=np.float32)
            else:
                if joint.type == "free":
                    if self.use_table and "r_hand_freejoint" not in joint.name:
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

    def _set_actuators_space(self):
        bounds = np.concatenate(
            [np.expand_dims(actuator.ctrlrange, 0) for actuator in self.world_entity.mjcf_model.find_all('actuator')],
            axis=0,
            dtype=np.float32)
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.actuators_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

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

    def do_simulation(self, ctrl, n_frames, increase_steps=True):
        for _ in range(n_frames):
            # Prevent simulation unstabillity
            try:
                self.env.step(ctrl)
            except:
                print('Simulation unstable, environment reset.')
        if increase_steps:
            self.steps += 1

    def _get_obs(self):
        self.render()
        if len(self.icub_observation_space) > 1:
            obs = {}
            for space in self.icub_observation_space:
                if space == 'camera':
                    obs['camera'] = self.env.physics.render(height=480, width=640, camera_id=self.obs_camera)
                elif space == 'joints':
                    obs['joints'] = np.empty([0, ], dtype=np.float32)
                    named_qpos = self.env.physics.named.data.qpos
                    for actuator in self.actuators_to_control_dict:
                        obs['joints'] = np.append(obs['joints'],
                                                  np.sum(named_qpos[actuator['jnt']] * actuator['coeff']))
                elif space == 'cartesian':
                    if self.cartesian_orientation == 'ypr':
                        obs['cartesian'] = np.concatenate(
                            (self.env.physics.named.data.xpos[self.eef_name],
                             Quaternion(matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                          (3, 3)), atol=1e-05).yaw_pitch_roll))[self.cartesian_ids]
                    else:
                        obs['cartesian'] = np.concatenate(
                            (self.env.physics.named.data.xpos[self.eef_name],
                             Quaternion(matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                          (3, 3)), atol=1e-05).q))[self.cartesian_ids]
                elif space == 'features':
                    obs['features'] = self.feature_extractor(self.env.physics.render(height=480,
                                                                                     width=640,
                                                                                     camera_id=self.obs_camera))
                elif space == 'touch':
                    self.compute_num_fingers_touching_object()
                    obs['touch'] = np.zeros(len(self.contact_geom_ids_fingers_meshes))
                    for finger in self.fingers_touching_object:
                        obs['touch'][self.contact_geom_fingers_names.index(finger)] = 1.0
                elif space == 'flare':
                    features = self.feature_extractor(self.env.physics.render(height=480,
                                                                              width=640,
                                                                              camera_id=self.obs_camera))
                    if self.steps == 0:
                        self.flare_features = [features,
                                               np.zeros(features.shape),
                                               features,
                                               np.zeros(features.shape),
                                               features]
                    else:
                        self.flare_features = [self.flare_features[2],
                                               self.flare_features[3],
                                               self.flare_features[4],
                                               self.flare_features[4] - features,
                                               features]
                    obs['flare'] = np.concatenate(self.flare_features, axis=1)
                elif space == 'superquadric_center':
                    if hasattr(self, 'superq_pose'):
                        if self.superq_pose is None:
                            obs['superquadric_center'] = np.zeros(3, dtype=np.float32)
                        else:
                            obs['superquadric_center'] = self.superq_pose['superq_center']
                    else:
                        obs['superquadric_center'] = np.zeros(3, dtype=np.float32)
                elif space == 'object_pose':
                    obs['object_pose'] = self.env.physics.data.qpos[self.joint_ids_objects]
            return obs
        elif 'camera' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            return self.env.physics.render(height=480, width=640, camera_id=self.obs_camera)
        elif 'joints' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            obs = np.empty([0, ], dtype=np.float32)
            named_qpos = self.env.physics.named.data.qpos
            for actuator in self.actuators_to_control_dict:
                obs = np.append(obs,
                                np.sum(named_qpos[actuator['jnt']] * actuator['coeff']))
            return obs
        elif 'cartesian' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            if self.cartesian_orientation == 'ypr':
                return np.concatenate((self.env.physics.named.data.xpos[self.eef_name],
                                       Quaternion(matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                                    (3, 3)),
                                                  atol=1e-05).yaw_pitch_roll))[self.cartesian_ids]
            else:
                return np.concatenate((self.env.physics.named.data.xpos[self.eef_name],
                                       Quaternion(matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                                    (3, 3)), atol=1e-05).q))[self.cartesian_ids]
        elif 'features' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            return self.feature_extractor(self.env.physics.render(height=480, width=640, camera_id=self.obs_camera))
        elif 'touch' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            self.compute_num_fingers_touching_object()
            obs = np.zeros(len(self.contact_geom_ids_fingers_meshes))
            for finger in self.fingers_touching_object:
                obs[self.contact_geom_fingers_names.index(finger)] = 1.0
            return obs
        elif 'flare' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            features = self.feature_extractor(self.env.physics.render(height=480,
                                                                      width=640,
                                                                      camera_id=self.obs_camera))
            if self.steps == 0:
                self.flare_features = [features,
                                       np.zeros(features.shape),
                                       features,
                                       np.zeros(features.shape),
                                       features]
            else:
                self.flare_features = [self.flare_features[2],
                                       self.flare_features[3],
                                       self.flare_features[4],
                                       self.flare_features[4] - features,
                                       features]
            return np.concatenate(self.flare_features, axis=1)
        elif 'superquadric_center' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            if hasattr(self, 'superq_pose'):
                if self.superq_pose is None:
                    return np.zeros(3, dtype=np.float32)
                else:
                    return self.superq_pose['superq_center']
            else:
                return np.zeros(3, dtype=np.float32)
        elif 'object_pose' in self.icub_observation_space and len(self.icub_observation_space) == 1:
            obs = self.env.physics.data.qpos[self.joint_ids_objects]
            return obs

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        # Set random qpos of the controlled joints, with the exception of the iCub hands
        if self.random_initial_pos:
            random_pos = self.state_space.sample()[self.joints_to_control_no_hand_ids]
            self.init_qpos[self.joints_to_control_no_hand_ids] = random_pos
            random_pos = self.state_space.sample()[self.joint_ids_objects]
            # Force z_objects > z_table and normalize quaternions
            for i in range(int(len(random_pos) / 7)):
                random_pos[i * 7 + 2] = np.maximum(random_pos[i * 7 + 2],
                                                   self.state_space.low[self.joint_ids_objects[i * 7 + 2]] + 0.1)
                random_pos[i * 7 + 3:i * 7 + 7] /= np.linalg.norm(random_pos[i * 7 + 3:i * 7 + 7])
            self.init_qpos[self.joint_ids_objects] = random_pos
        if self.random_ycb_video_graspable_object:
            random_object = self.ycb_video_graspable_objects_list \
                [np.random.randint(0, len(self.ycb_video_graspable_objects_list))]
            self.objects = [random_object['object']]
            self.objects_positions = [np.array(list(random_object['position'].values()))]
            self.objects_quaternions = [np.array(list(random_object['orientation'].values()))]
            self.moved_object_height = random_object['moved_object_height']
            self.lift_object_height = self.objects_positions[0][2] + 0.1
            self.add_ycb_video_objects(self.objects, remove_last_object=True)
            if self.track_object:
                self.track_object_with_camera()
            self.env.reset()
            self.init_qpos[self.joint_ids_objects[0:3]] = self.objects_positions[0]
            self.init_qpos[self.joint_ids_objects[3:7]] = self.objects_quaternions[0]
        if self.randomly_rotate_object_z_axis:
            for i in range(int(len(self.init_qpos[self.joint_ids_objects]) / 7)):
                object_quaternions_pyquaternion = Quaternion(self.init_qpos[
                                                                 self.joint_ids_objects[i * 7 + 3:i * 7 + 7]])
                z_rotation_quaternion = Quaternion(axis=[0, 0, 1], angle=np.random.rand() * 2 * math.pi)
                rotated_object_quaternions_pyquaternion = z_rotation_quaternion * object_quaternions_pyquaternion
                object_quaternions = rotated_object_quaternions_pyquaternion.q
                self.init_qpos[self.joint_ids_objects[i * 7 + 3:i * 7 + 7]] = object_quaternions
        if self.use_only_right_hand_model:
            self.env.physics.named.data.mocap_pos['icub_r_hand_welding'] = self.init_qpos[self.joint_ids_icub_free][:3]
            self.env.physics.named.data.mocap_quat['icub_r_hand_welding'] = \
                self.init_qpos[self.joint_ids_icub_free][3:7]
        self.set_state(np.concatenate([self.init_qpos.copy(), self.init_qvel.copy(), self.env.physics.data.act]))
        self.env.physics.forward()
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
        images = []
        for cam in self.render_cameras:
            img = np.array(self.env.physics.render(height=480, width=640, camera_id=cam), dtype=np.uint8)
            if cam == 'head_cam' and self.render_objects_com:
                objects_com_x_y_z = []
                for i in range(int(len(self.joint_ids_objects) / 7)):
                    objects_com_x_y_z.append(self.env.physics.data.qpos[self.joint_ids_objects[i * 7:i * 7 + 3]])
                com_uvs = self.points_in_pixel_coord(self.points_in_camera_coord(objects_com_x_y_z))
                for com_uv in com_uvs:
                    img = cv2.circle(img, com_uv, 5, (0, 255, 0), -1)
            images.append(img)
            cv2.imshow(cam, img[:, :, ::-1])
            cv2.waitKey(1)
        return images

    def add_ycb_video_objects(self, object_names, remove_last_object):
        if remove_last_object:
            obj_to_rm = self.world.worldbody.body[-1]
            obj_to_rm.remove(affect_attachments=True)
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

    def track_object_with_camera(self):
        if self.use_only_right_hand_model:
            self.world.worldbody.body['head_camera_track_hand'].camera._elements[0].target = \
                self.world.worldbody.body[len(self.world.worldbody.body) - 1]
        else:
            self.world.worldbody.body['icub'].body['torso_1'].body['torso_2'].body['chest'].body['neck_1'] \
                .body['neck_2'].body['head'].body['head_camera_track_hand'].camera._elements[0].target = \
                self.world.worldbody.body[len(self.world.worldbody.body) - 1]

    def add_table(self):
        table_path = "../models/table.xml"
        table_mjcf = mjcf.from_path(table_path, escape_separators=False)
        # Extract table information
        size_x = table_mjcf.worldbody.body['table'].body['wood'].geom['table_collision'].size[0]
        size_y = table_mjcf.worldbody.body['table'].body['wood'].geom['table_collision'].size[1]
        size_z = table_mjcf.worldbody.body['table'].body['wood'].geom['table_collision'].size[2]
        # Split the table in multiple boxes to avoid objects jittering
        x_split = 10
        y_split = 20
        for i in range(x_split):
            for j in range(y_split):
                pos_i = - size_x + size_x * 2 * i / x_split + size_x / x_split
                pos_j = - size_y + size_y * 2 * j / y_split + size_y / y_split
                table_mjcf.worldbody.body['table'].add('body',
                                                       name='wood_{}_{}'.format(i, j),
                                                       pos=[0, 0, 0])
                table_mjcf.worldbody.body['table'].body['wood_{}_{}'.format(i, j)]. \
                    add('geom',
                        name='wood_{}_{}'.format(i, j),
                        pos=[pos_i, pos_j, 0],
                        type='box',
                        rgba=[0, 0, 0, 0],
                        size=[size_x / x_split, size_y / y_split, size_z],
                        group="1")
        self.world.attach(table_mjcf.root_model)

    def points_in_camera_coord(self, points):
        com_xyzs = []
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
            com_xyzs.append(np.array([p_cam[0, 3], p_cam[1, 3], p_cam[2, 3]]))
        return com_xyzs

    def point_in_r_hand_dh_frame(self, point):
        # Point roto-translation matrix in world coordinates
        p_world = np.array([[1, 0, 0, point[0]],
                            [0, 1, 0, point[1]],
                            [0, 0, 1, point[2]],
                            [0, 0, 0, 1]],
                           dtype=np.float32)
        # r_hand_dh_frame roto-translation matrix in world coordinates
        dh_frame_world = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]],
                                  dtype=np.float32)
        dh_frame_pos = self.env.physics.named.data.site_xpos['r_hand_dh_frame_site']
        dh_frame_world[:3, -1] = dh_frame_pos
        dh_frame_rot = np.reshape(self.env.physics.named.data.site_xmat['r_hand_dh_frame_site'], (3, 3))
        dh_frame_world[:3, :3] = dh_frame_rot
        # Point roto-translation matrix in r_hand_dh_frame coordinates
        p_r_hand_dh_frame = np.matmul(np.linalg.inv(dh_frame_world), p_world)
        return np.array([p_r_hand_dh_frame[0, 3], p_r_hand_dh_frame[1, 3], p_r_hand_dh_frame[2, 3]])

    def points_in_pixel_coord(self, points):
        com_uvs = []
        for point in points:
            # Pixel coordinates computation
            x = point[0] / (-point[2]) * self.fy
            y = point[1] / (-point[2]) * self.fy
            u = int(x) + 320
            v = -int(y) + 240
            com_uvs.append(np.array([u, v]))
        return com_uvs

    def compute_num_fingers_touching_object(self):
        self.fingers_touching_object = []
        for contact in self.env.physics.data.contact:
            if (contact['geom1'] in self.contact_geom_ids_fingers_meshes.values()
                and contact['geom2'] in self.contact_geom_ids_objects_meshes.values()) or \
                    (contact['geom1'] in self.contact_geom_ids_objects_meshes.values()
                     and contact['geom2'] in self.contact_geom_ids_fingers_meshes.values()):
                if contact['geom1'] in self.contact_geom_ids_fingers_meshes.values():
                    self.fingers_touching_object.append((list(
                        self.contact_geom_ids_fingers_meshes.keys())[list(
                        self.contact_geom_ids_fingers_meshes.values()).index(contact['geom1'])]))
                else:
                    self.fingers_touching_object.append((list(
                        self.contact_geom_ids_fingers_meshes.keys())[list(
                        self.contact_geom_ids_fingers_meshes.values()).index(contact['geom2'])]))
                self.number_of_contacts += 1
        self.number_of_contacts = len(self.fingers_touching_object)
        return self.number_of_contacts

    @staticmethod
    def go_to(qpos_init, qpos_final, current_step, total_num_steps):
        # Minimum jerk trajectory
        if current_step > total_num_steps * 80 / 100:
            qpos_t = qpos_final
        else:
            T = total_num_steps * 80 / 100
            t = current_step
            t_T = t / T
            qpos_t = qpos_init + (qpos_final - qpos_init) * (10 * t_T ** 3 - 15 * t_T ** 4 + 6 * t_T ** 5)
        return qpos_t
