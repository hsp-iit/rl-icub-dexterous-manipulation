# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import gym
import numpy as np
from rl_icub_dexterous_manipulation.feature_extractors.images_feature_extractor_CLIP import ImagesFeatureExtractorCLIP
from rl_icub_dexterous_manipulation.feature_extractors.images_feature_extractor_MAE import ImagesFeatureExtractorMAE
from pyquaternion import Quaternion
from rl_icub_dexterous_manipulation.external.stable_baselines3_mod.sac import SAC
from rl_icub_dexterous_manipulation.envs.icub_visuomanip import ICubEnv
from rl_icub_dexterous_manipulation.yarp_modules.superquadrics_module import SuperquadricsModule

import sys
import yarp
# Initialize YARP
yarp.Network.init()


class ICubEnvReal(ICubEnv):

    def __init__(self,
                 model_path,
                 frame_skip=5,
                 icub_observation_space=('joints',),
                 random_initial_pos=True,
                 ycb_video_graspable_objects_config_path='../config/ycb_video_objects_graspable_poses.yaml',
                 mujoco_scanned_objects_config_path='../config/mujoco_scanned_objects_graspable.yaml',
                 random_ycb_video_graspable_object=False,
                 random_mujoco_scanned_object=False,
                 done_moved_object_mso_angle=90,
                 objects=(),
                 use_table=True,
                 objects_positions=(),
                 objects_quaternions=(),
                 randomly_rotate_object_z_axis=False,
                 randomly_move_objects=False,
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
                 rotated_dist_superq_center=False,
                 reward_line_pregrasp_superq_center=False,
                 reward_dist_original_superq_grasp_position=False,
                 high_negative_reward_approach_failures=False,
                 goal_reached_only_with_lift_refine_grasp=False,
                 exclude_vertical_touches=False,
                 min_fingers_touching_object=5,
                 scale_pos_lift_reward_wrt_touching_fingers=False,
                 joints_margin=0.0,
                 null_reward_out_image=False,
                 done_if_joints_out_of_limits=True,
                 do_not_consider_done_z_pos=False,
                 lift_object_height=1.02,
                 curriculum_learning=False,
                 curriculum_learning_approach_object=False,
                 curriculum_learning_approach_object_start_step=0,
                 curriculum_learning_approach_object_end_step=1000000,
                 learning_from_demonstration=False,
                 max_lfd_steps=10000,
                 max_delta_qpos=0.1,
                 lfd_keep_only_successful_episodes=False,
                 lfd_with_approach=False,
                 approach_in_reset_model=False,
                 pregrasp_distance_from_grasp_pose=0.05,
                 max_delta_cartesian_pos=0.02,
                 max_delta_cartesian_rot=0.1,
                 distanced_superq_grasp_pose=False,
                 control_gaze=False,
                 ik_solver='idyntree',
                 limit_torso_pitch_ikin=False,
                 use_only_right_hand_model=False,
                 grasp_planner='superquadrics',
                 pretrained_model_dir=None
                 ):

        self.rf = yarp.ResourceFinder()
        self.rf.setVerbose(True)
        self.rf.setDefaultContext("SuperquadricsModule")
        self.rf.setDefaultConfigFile('../yarp_modules/configs/config_refine_grasp.ini')

        self.rf.configure(sys.argv)
        # Run module
        self.robot_module = SuperquadricsModule()
        self.robot_module.configure(self.rf)
        self.robot_module.updateModule()

        # Connect  YARP ports
        yarp.NetworkBase_connect("/depthCamera/rgbImage:o", "/view_image", 'fast_tcp', False)
        yarp.NetworkBase_connect("/depthCamera/rgbImage:o", "/superq/image:i", 'fast_tcp', False)
        yarp.NetworkBase_connect("/depthCamera/depthImage:o", "/view_depth",
                                 'fast_tcp+recv.portmonitor+type.dll+file.depthimage_to_rgb', False)
        yarp.NetworkBase_connect("/depthCamera/depthImage:o", "/superq/depth:i", 'fast_tcp', False)
        yarp.NetworkBase_connect("/realsense-holder-publisher/pose:o", "/superq/realsense_pose", 'fast_tcp', False)
        yarp.NetworkBase_connect("/icub/right_arm/state:o", "/superq/r_arm_qpos", 'fast_tcp', False)
        yarp.NetworkBase_connect("/icub/cartesianController/right_arm/state:o", "/superq/r_arm_xpos", 'fast_tcp', False)
        yarp.NetworkBase_connect("/icub/skin/right_hand_comp", "/superq/r_hand_touch", 'fast_tcp', False)

        self.moved_object_height = 0.98

        # Set environment and task parameters
        self.icub_observation_space = icub_observation_space
        self.feature_extractor_model_name = feature_extractor_model_name
        if 'features' in icub_observation_space or 'flare' in icub_observation_space:
            if 'CLIP' in self.feature_extractor_model_name:
                self.feature_extractor = ImagesFeatureExtractorCLIP(model_name=self.feature_extractor_model_name)
            elif 'MAE' in self.feature_extractor_model_name:
                self.feature_extractor = ImagesFeatureExtractorMAE(model_name=self.feature_extractor_model_name)
        self.random_initial_pos = random_initial_pos
        self.frame_skip = frame_skip
        self.steps = 0
        self.total_steps = 0
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
            print('The value of lift_object_height will be overwritten whenever a new object will be added to the '
                  'environment.')
        self.curriculum_learning = curriculum_learning
        self.curriculum_learning_approach_object = curriculum_learning_approach_object
        self.curriculum_learning_approach_object_start_step = curriculum_learning_approach_object_start_step
        self.curriculum_learning_approach_object_end_step = curriculum_learning_approach_object_end_step

        # Set if using the original superquadric grasp pose or the distanced pose
        self.distanced_superq_grasp_pose = distanced_superq_grasp_pose
        self.actuator_names = ["r_shoulder_pitch",
                               "r_shoulder_roll",
                               "r_shoulder_yaw",
                               "r_elbow",
                               "r_wrist_prosup",
                               "r_wrist_pitch",
                               "r_wrist_yaw",
                               "r_hand_finger",
                               "r_thumb_oppose",
                               "r_thumb_proximal",
                               "r_thumb_distal",
                               "r_index_proximal",
                               "r_index_distal",
                               "r_middle_proximal",
                               "r_middle_distal",
                               "r_pinky"]
        self.joint_names = ["r_shoulder_pitch",
                            "r_shoulder_roll",
                            "r_shoulder_yaw",
                            "r_elbow",
                            "r_wrist_prosup",
                            "r_wrist_pitch",
                            "r_wrist_yaw",
                            "r_hand_finger",
                            "r_thumb_oppose",
                            "r_thumb_proximal",
                            "r_thumb_distal",
                            "r_index_proximal",
                            "r_index_distal",
                            "r_middle_proximal",
                            "r_middle_distal",
                            "r_pinky"]

        self.joint_bounds = {"r_shoulder_pitch": np.array([0, 0]),
                             "r_shoulder_roll": np.array([0, 0]),
                             "r_shoulder_yaw": np.array([0, 0]),
                             "r_elbow": np.array([0, 0]),
                             "r_wrist_prosup": np.array([0, 0]),
                             "r_wrist_pitch": np.array([0, 0]),
                             "r_wrist_yaw": np.array([0, 0]),
                             "r_hand_finger": np.array([0, 60]),
                             "r_thumb_oppose": np.array([90, 90]),
                             "r_thumb_proximal": np.array([0, 90]),
                             "r_thumb_distal": np.array([0, 170]),
                             "r_index_proximal": np.array([0, 90]),
                             "r_index_distal": np.array([0, 170]),
                             "r_middle_proximal": np.array([0, 90]),
                             "r_middle_distal": np.array([0, 170]),
                             "r_pinky": np.array([0, 250])}

        # Select actuators to control for the task
        if 'cartesian' in self.icub_observation_space:
            self.training_components = []
            for tr_component in training_components:
                if tr_component == 'r_hand':
                    self.training_components.append('r_hand')
                elif tr_component == 'l_hand':
                    self.training_components.append('l_hand')
                elif tr_component == 'r_hand_no_thumb_oppose':
                    self.training_components.append('r_hand_no_thumb_oppose')
                elif tr_component == 'l_hand_no_thumb_oppose':
                    self.training_components.append('l_hand_no_thumb_oppose')
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
        if 'r_hand_no_thumb_oppose' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('r_hand') or
                                                                                 j.startswith('r_thumb') or
                                                                                 j.startswith('r_index') or
                                                                                 j.startswith('r_middle') or
                                                                                 j.startswith('r_pinky')) and
                                                                                not j == 'r_thumb_oppose'])
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
        if 'l_hand_no_thumb_oppose' in self.training_components:
            self.actuators_to_control.extend([j for j in self.actuator_names if (j.startswith('l_hand') or
                                                                                 j.startswith('l_thumb') or
                                                                                 j.startswith('l_index') or
                                                                                 j.startswith('l_middle') or
                                                                                 j.startswith('l_pinky')) and
                                                                                not j == 'l_thumb_oppose'])
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

        # Set if controlling gaze
        self.control_gaze = control_gaze

        self.number_of_contacts = 0
        self.previous_number_of_contacts = 0
        self.fingers_touching_object = []

        self.flare_features = []

        # Set spaces
        self.max_delta_qpos = max_delta_qpos
        self.max_delta_cartesian_pos = max_delta_cartesian_pos
        self.max_delta_cartesian_rot = max_delta_cartesian_rot
        self._set_action_space()
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
        self.approach_in_reset_model = approach_in_reset_model
        self.pregrasp_distance_from_grasp_pose = pregrasp_distance_from_grasp_pose

        # Set task parameters
        self.eef_name = eef_name
        self.goal_xpos_tolerance = 0.05
        self.done_if_joints_out_of_limits = done_if_joints_out_of_limits
        self.do_not_consider_done_z_pos = do_not_consider_done_z_pos

        # Set reward values
        self.reward_goal = reward_goal
        self.reward_out_of_joints = reward_out_of_joints
        self.reward_single_step_multiplier = reward_single_step_multiplier
        self.reward_end_timesteps = reward_end_timesteps
        self.reward_dist_superq_center = reward_dist_superq_center
        self.rotated_dist_superq_center = rotated_dist_superq_center
        self.reward_line_pregrasp_superq_center = reward_line_pregrasp_superq_center
        self.reward_dist_original_superq_grasp_position = reward_dist_original_superq_grasp_position
        self.goal_reached_only_with_lift_refine_grasp = goal_reached_only_with_lift_refine_grasp
        self.exclude_vertical_touches = exclude_vertical_touches
        self.min_fingers_touching_object = min_fingers_touching_object
        self.scale_pos_lift_reward_wrt_touching_fingers = scale_pos_lift_reward_wrt_touching_fingers
        self.high_negative_reward_approach_failures = high_negative_reward_approach_failures
        self.done_moved_object_mso_angle = done_moved_object_mso_angle

        # Set grasp planner
        self.grasp_planner = grasp_planner

        # Upload pre-trained model
        if pretrained_model_dir is not None:
            # If the model is pretrained with an algorithm different from SAC, this part must be extended
            self.pretrained_model = SAC.load(pretrained_model_dir + '/best_model.zip', buffer_size=100)
        self.prev_obs = None

        # Reset environment
        self.reset()

    def _set_observation_space(self):
        obs_space = {}
        for space in self.icub_observation_space:
            if space == 'camera':
                obs_space['camera'] = gym.spaces.Box(low=0,
                                                     high=255,
                                                     shape=(480, 640, 3),
                                                     dtype='uint8')
            elif space == 'joints':
                bounds = np.concatenate(
                    [np.expand_dims(self.joint_bounds[actuator], 0) if actuator in self.actuators_to_control
                     else np.empty([0, 2], dtype=np.float32)
                     for actuator in self.actuator_names],
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
                                                    shape=[5],
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
            elif space == 'pretrained_output':
                obs_space['pretrained_output'] = gym.spaces.Box(low=self.action_space.low,
                                                                high=self.action_space.high,
                                                                shape=self.action_space.shape,
                                                                dtype=self.action_space.dtype)
            elif space == 'grasp_type':
                obs_space['grasp_type'] = gym.spaces.Box(low=-np.inf,
                                                         high=np.inf,
                                                         shape=[1],
                                                         dtype=np.int8)
        self.observation_space = gym.spaces.Dict(obs_space)

    def _set_state_space(self):
        bounds = np.empty([0, 2], dtype=np.float32)
        for joint in self.actuators_to_control:
            bounds = np.concatenate([bounds, np.expand_dims(self.joint_bounds[joint], 0)], axis=0, dtype=np.float32)
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.state_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_actuators_space(self):
        bounds = np.empty([0, 2], dtype=np.float32)
        for joint in self.actuators_to_control:
            bounds = np.concatenate([bounds, np.expand_dims(self.joint_bounds[joint], 0)], axis=0, dtype=np.float32)
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.actuators_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.total_steps += self.steps
        self.steps = 0
        ob = self.reset_model()
        return ob

    def _get_obs(self):
        obs = {}
        for key in self.icub_observation_space:
            if key == 'cartesian':
                r_arm_xpose = self.robot_module._r_arm_xpos_port.read(True)
                r_arm_xpos = np.array([r_arm_xpose.get(i) for i in range(3)])
                r_arm_xpos[2] += self.robot_module.height_simulated_icub_root
                r_arm_xor_axis_angle = np.array([r_arm_xpose.get(i) for i in range(3, 7)])
                r_arm_xor_ypr = Quaternion(axis=r_arm_xor_axis_angle[0:3],
                                           angle=r_arm_xor_axis_angle[3]).yaw_pitch_roll
                obs['cartesian'] = np.concatenate([r_arm_xpos, r_arm_xor_ypr])
            elif key == 'flare':
                received_image = self.robot_module._input_image_port.read(True)
                self.robot_module._in_buf_image.copy(received_image)
                assert self.robot_module._in_buf_array.__array_interface__['data'][0] == \
                       self.robot_module._in_buf_image.getRawImage().__int__()
                frame = self.robot_module._in_buf_array
                input_image = frame.copy()
                # Crop input image, the d405 fov is too large
                input_image = input_image[120:360, 160:480, :]
                features = self.feature_extractor(input_image)
                if self.robot_module.policy_steps == 0:
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
            elif key == 'joints':
                if self.robot_module.joints_to_control != 'r_hand':
                    raise ValueError('Only joints_control for the right hand fingers has been implemented.')
                encs = yarp.Vector(self.robot_module.ipos.getAxes())
                ret = self.robot_module.ienc.getEncoders(encs.data())
                obs['joints'] = np.array([encs.get(k + 7) / 180 * np.pi for k in range(9)])
            elif key == 'superquadric_center':
                if hasattr(self.robot_module, 'superq_pose'):
                    if self.robot_module.superq_pose is None:
                        obs['superquadric_center'] = np.zeros(3, dtype=np.float32)
                    else:
                        obs['superquadric_center'] = self.robot_module.superq_center.copy()
                        obs['superquadric_center'][2] += self.robot_module.height_simulated_icub_root
                else:
                    obs['superquadric_center'] = np.zeros(3, dtype=np.float32)
            elif key == 'touch':
                r_hand_touch = self.robot_module._r_hand_touch_port.read(True)
                # print('reading port touch')
                r_hand_touch_np = np.array([r_hand_touch.get(i) for i in range(60)])
                obs['touch'] = np.zeros((5,))
                # At least one taxel in contact
                min_value_touch = 100 / 12
                # TOuch information values are in the order for: index, medium, ring, little, thumb
                if np.mean(r_hand_touch_np[:12]) >= min_value_touch:
                    obs['touch'][1] = 1
                if np.mean(r_hand_touch_np[12:24]) >= min_value_touch:
                    obs['touch'][2] = 1
                if np.mean(r_hand_touch_np[24:36]) >= min_value_touch:
                    obs['touch'][3] = 1
                if np.mean(r_hand_touch_np[36:48]) >= min_value_touch:
                    obs['touch'][4] = 1
                if np.mean(r_hand_touch_np[48:]) >= min_value_touch:
                    obs['touch'][0] = 1
        if 'pretrained_output' in self.icub_observation_space:
            # If the observation space of the pretrained model and of the model to be trained are different, this
            # part must be extended
            action, _ = self.pretrained_model.predict(obs, deterministic=True)
            obs['pretrained_output'] = action
        self.prev_obs = obs
        # print(obs)
        return obs

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        return self._get_obs()

    def point_in_r_hand_dh_frame(self, point, site_name='r_hand_dh_frame_site'):
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
        r_arm_xpose = self.robot_module._r_arm_xpos_port.read(True)
        dh_frame_pos = np.array([r_arm_xpose.get(i) for i in range(3)])
        dh_frame_world[:3, -1] = dh_frame_pos
        r_arm_xor_axis_angle = np.array([r_arm_xpose.get(i) for i in range(3, 7)])
        dh_frame_world[:3, :3] = Quaternion(axis=r_arm_xor_axis_angle[0:3],
                                            angle=r_arm_xor_axis_angle[3]).rotation_matrix
        # Point roto-translation matrix in r_hand_dh_frame coordinates
        p_r_hand_dh_frame = np.matmul(np.linalg.inv(dh_frame_world), p_world)
        return np.array([p_r_hand_dh_frame[0, 3], p_r_hand_dh_frame[1, 3], p_r_hand_dh_frame[2, 3]])

    def compute_num_fingers_touching_object(self):
        r_hand_touch = self.robot_module._r_hand_touch_port.read(True)
        r_hand_touch_np = np.array([r_hand_touch.get(i) for i in range(60)])
        self.number_of_contacts = 0
        # At least one taxels in contact
        min_value_touch = 100 / 12
        # Touch information values are in the order for: index, medium, ring, little, thumb
        if np.mean(r_hand_touch_np[:12]) >= min_value_touch:
            self.number_of_contacts += 1
        if np.mean(r_hand_touch_np[12:24]) >= min_value_touch:
            self.number_of_contacts += 1
        if np.mean(r_hand_touch_np[24:36]) >= min_value_touch:
            self.number_of_contacts += 1
        if np.mean(r_hand_touch_np[36:48]) >= min_value_touch:
            self.number_of_contacts += 1
        if np.mean(r_hand_touch_np[48:]) >= min_value_touch:
            self.number_of_contacts += 1
        return self.number_of_contacts
