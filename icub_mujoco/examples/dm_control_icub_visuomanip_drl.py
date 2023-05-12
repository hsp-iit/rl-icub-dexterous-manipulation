import numpy as np
from icub_mujoco.envs.icub_visuomanip_reaching import ICubEnvReaching
from icub_mujoco.envs.icub_visuomanip_gaze_control import ICubEnvGazeControl
from icub_mujoco.envs.icub_visuomanip_refine_grasp import ICubEnvRefineGrasp
from icub_mujoco.envs.icub_visuomanip_keep_grasp import ICubEnvKeepGrasp
from icub_mujoco.envs.icub_visuomanip_lift_grasped_object import ICubEnvLiftGraspedObject
from icub_mujoco.external.stable_baselines3_mod.sac import SAC
import argparse
import cv2
import gym
from icub_mujoco.external.stable_baselines3_mod.common.evaluation_bc import evaluate_policy_bc
from stable_baselines3.common.save_util import load_from_pkl
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stable_baselines3.common import policies
import torch
from icub_mujoco.external.d3rlpy_mod.d3rlpy.algos.awac import AWAC
from d3rlpy.wrappers.sb3 import to_mdp_dataset
from d3rlpy.models.encoders import VectorEncoderFactory
from icub_mujoco.external.d3rlpy_mod.d3rlpy.online.buffers import ReplayBuffer


parser = argparse.ArgumentParser()
parser.add_argument('--test_model',
                    action='store_true',
                    help='Test the best_model.zip stored in --eval_dir.')
parser.add_argument('--fine_tune_model',
                    action='store_true',
                    help='Fine tune the best_model.zip stored in --pretrained_model_dir.')
parser.add_argument('--record_video',
                    action='store_true',
                    help='Test the best_model.zip stored in --eval_dir and record.')
parser.add_argument('--save_replay_buffer',
                    action='store_true',
                    help='Save the replay buffer.')
parser.add_argument('--save_demonstrations_replay_buffers_per_object',
                    action='store_true',
                    help='Save the replay buffers filled with demonstrations of different objects. This option is '
                         'currently only supported when the option --train_with_reptile is True.')
parser.add_argument('--save_replay_buffer_path',
                    type=str,
                    default='replay_buffer',
                    help='Path where the replay buffer will be stored.')
parser.add_argument('--load_replay_buffer',
                    action='store_true',
                    help='Load the replay buffer.')
parser.add_argument('--load_replay_buffer_path',
                    type=str,
                    default='replay_buffer',
                    help='Path where the replay buffer to load is located.')
parser.add_argument('--train_with_two_replay_buffers',
                    action='store_true',
                    help='Set options to train SAC with two replay buffers. The replay buffer containing the '
                         'demonstrations will be loaded from where specified in the '
                         'load_demonstrations_replay_buffer_path option.')
parser.add_argument('--train_with_OERLD',
                    action='store_true',
                    help='Set options to train SAC with the loss function in the paper "Overcoming Exploration in '
                         'Reinforcement Learning with Demonstrations". The replay buffer containing the '
                         'demonstrations will be loaded from where specified in the '
                         'load_demonstrations_replay_buffer_path option.')
parser.add_argument('--train_with_implicit_underparametrization_penalty',
                    action='store_true',
                    help='Set options to train SAC adding to the critic loss the penalty in the paper "Implicit '
                         'Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning".')
parser.add_argument('--train_with_residual_learning_pretrained_critic',
                    action='store_true',
                    help='Set options to train SAC with residual learning, setting the initial weights of the critic '
                         'from the model in --pretrained_model_dir.')
parser.add_argument('--train_with_reptile',
                    action='store_true',
                    help='Train a policy with reptile from the paper "On First-Order Meta-Learning Algorithms".')
parser.add_argument('--k_reptile',
                    action='store',
                    type=int,
                    default=1000,
                    help='Set the number of timesteps for each batch in reptile. Default is 1000.')
parser.add_argument('--train_with_behavior_cloning',
                    action='store_true',
                    help='Train a policy with behavior cloning, starting from data in a replay buffer stored in '
                         'load_replay_buffer_path.')
parser.add_argument('--train_with_AWAC',
                    action='store_true',
                    help='Train a policy with AWAC, using as offline data the replay buffer stored in '
                         'load_demonstrations_replay_buffer_path.')
parser.add_argument('--behavior_cloning_epochs',
                    action='store',
                    type=int,
                    default=1,
                    help='Set the number of training epochs with behavior cloning. Default is 1.')
parser.add_argument('--load_demonstrations_replay_buffer_path',
                    type=str,
                    help='Path where the replay buffer with the demonstrations to be loaded is located.')
parser.add_argument('--xml_model_path',
                    action='store',
                    type=str,
                    default='../models/icub_position_actuators_actuate_hands.xml',
                    help='Set the path of the xml model.')
parser.add_argument('--initial_qpos_path',
                    action='store',
                    type=str,
                    default='../config/initial_qpos_actuated_hand.yaml',
                    help='Set the path of the initial actuators values.')
parser.add_argument('--tensorboard_dir',
                    action='store',
                    type=str,
                    default='tensorboards',
                    help='Set the directory where tensorboard files are saved. Default directory is tensorboards.')
parser.add_argument('--buffer_size',
                    action='store',
                    type=int,
                    default=1000000,
                    help='Set the size of the replay buffer. Default is 1000000.')
parser.add_argument('--reward_goal',
                    action='store',
                    type=float,
                    default=1.0,
                    help='Set the reward for reaching the goal. Default is 1.0.')
parser.add_argument('--reward_out_of_joints',
                    action='store',
                    type=float,
                    default=-1.0,
                    help='Set the reward for violating joints limits. Default is -1.0.')
parser.add_argument('--reward_end_timesteps',
                    action='store',
                    type=float,
                    default=-1.0,
                    help='Set the reward for exceeding number of timesteps. Default is -1.0.')
parser.add_argument('--reward_single_step_multiplier',
                    action='store',
                    type=float,
                    default=10.0,
                    help='Set the multiplication factor of the default per-step reward in meters or pixels.')
parser.add_argument('--reward_dist_superq_center',
                    action='store_true',
                    help='Add a reward component in the grasp refinement task for the distance of the superquadric '
                         'center in the xy axes of the eef.')
parser.add_argument('--rotated_dist_superq_center',
                    action='store_true',
                    help='Compute the superquardric grasp pose reward w.r.t the r_hand_dh_frame rotated of 45° around '
                         'the y axis.')
parser.add_argument('--reward_line_pregrasp_superq_center',
                    action='store_true',
                    help='Add a reward component in the grasp refinement task for the distance to the line connecting '
                         'the superquadric center and the distanced superquadric grasp pose.')
parser.add_argument('--reward_dist_original_superq_grasp_position',
                    action='store_true',
                    help='Add a reward component in the grasp refinement task for the distance to the original '
                         'superquadric grasp position.')
parser.add_argument('--goal_reached_only_with_lift_refine_grasp',
                    action='store_true',
                    help='Successful episode only with object lifted in grasp refinement task.')
parser.add_argument('--high_negative_reward_approach_failures',
                    action='store_true',
                    help='Strongly penalize moved object in the approach phase in the grasp refinement task.')
parser.add_argument('--joints_margin',
                    action='store',
                    type=float,
                    default=0.0,
                    help='Set the margin from joints limits for joints control.')
parser.add_argument('--net_arch',
                    type=int,
                    nargs='+',
                    default=[64, 64],
                    help='Set the architecture of the MLP network. Default is [64, 64]')
parser.add_argument('--net_arch_critic',
                    type=int,
                    nargs='+',
                    default=[],
                    help='Set the architecture of the critic of the MLP network. This is considered only if '
                         '--train_with_residual_learning_pretrained_critic is True to set the critic as the one of the '
                         'pretrained model. If not specified, this will be set as specified in --net_arch.')
parser.add_argument('--initialize_actor_mu_weights_to_zero',
                    action='store_true',
                    help='Initialize weights and biases of the actor mu to zero. This is considered only if '
                         '--train_with_residual_learning_pretrained_critic is True.')
parser.add_argument('--train_freq',
                    action='store',
                    type=int,
                    default=10,
                    help='Set the update frequency for SAC. Default is 10.')
parser.add_argument('--gradient_steps',
                    action='store',
                    type=int,
                    default=1,
                    help='Set the number of gradient steps for SAC. Default is 1.')
parser.add_argument('--learning_starts',
                    action='store',
                    type=int,
                    default=100,
                    help='Set the number of timesteps to start the training. Default is 100')
parser.add_argument('--total_training_timesteps',
                    action='store',
                    type=int,
                    default=10000000,
                    help='Set the number of training episodes for SAC. Default is 10M')
parser.add_argument('--ent_coef',
                    action='store',
                    type=str,
                    default='auto',
                    help='Set the entropy coefficient for SAC, e.g. auto_0.01 to set 0.01 as initial value.')
parser.add_argument('--eval_dir',
                    action='store',
                    type=str,
                    default='logs_eval',
                    help='Set the directory where evaluation files are saved. Default directory is logs_eval.')
parser.add_argument('--pretrained_model_dir',
                    action='store',
                    type=str,
                    help='Set the directory where the requested pretrained model is saved.')
parser.add_argument('--eval_freq',
                    action='store',
                    type=int,
                    default=100000,
                    help='Set the evaluation frequency for SAC. Default is 100k')
parser.add_argument('--icub_observation_space',
                    type=str,
                    nargs='+',
                    default='joints',
                    help='Set the observation space: joints will use as observation space joints positions, '
                         'camera will use information from the camera specified with the argument obs_camera, '
                         'features the features extracted by the camera specified with the argument obs_camera, '
                         'flare a combination of the features with information at the previous timesteps, '
                         'pretrained_output the output of the pre-trained policy stored in pretrained_model_dir, '
                         'grasp_type an integer value that describes the grasp type based on the initial grasp pose '
                         'and touch the tactile information. If you pass multiple argument, you will use a '
                         'MultiInputPolicy.')
parser.add_argument('--exclude_vertical_touches',
                    action='store_true',
                    help='Do not consider vertical contacts to compute the number of fingers touching an object.')
parser.add_argument('--min_fingers_touching_object',
                    action='store',
                    type=int,
                    default=5,
                    help='Set the minimum number of fingers touching the object in the grasp refinement task to get a '
                         'positive reward when lifting it. Default is 5.')
parser.add_argument('--scale_pos_lift_reward_wrt_touching_fingers',
                    action='store_true',
                    help='Multiply the positive lift rewards by the fraction of fingers in contact with the object.')
parser.add_argument('--eef_name',
                    type=str,
                    default='r_hand',
                    help='Specify the name of the body to be considered as end-effector')
parser.add_argument('--render_cameras',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Set the cameras used for rendering. Available cameras are front_cam and head_cam.')
parser.add_argument('--obs_camera',
                    type=str,
                    default='head_cam',
                    help='Set the cameras used for observation. Available cameras are front_cam and head_cam.')
parser.add_argument('--track_object',
                    action='store_true',
                    help='Set the target of the tracking camera to the object, instead of the default r_hand')
parser.add_argument('--curriculum_learning',
                    action='store_true',
                    help='Use curriculum learning for joints/cartesian offsets.')
parser.add_argument('--curriculum_learning_approach_object',
                    action='store_true',
                    help='Use curriculum learning in the distance of the pre-grasp pose from the object.')
parser.add_argument('--curriculum_learning_approach_object_start_step',
                    action='store',
                    type=int,
                    default=0,
                    help='Set the initial step for the curriculum learning phase while approaching the object.')
parser.add_argument('--curriculum_learning_approach_object_end_step',
                    action='store',
                    type=int,
                    default=1000000,
                    help='Set the final step for the curriculum learning phase while approaching the object.')
parser.add_argument('--superquadrics_camera',
                    type=str,
                    default='head_cam',
                    help='Set the cameras used for observation. Available cameras are front_cam and head_cam.')
parser.add_argument('--print_done_info',
                    action='store_true',
                    help='Print information at the end of each episode')
parser.add_argument('--do_not_consider_done_z_pos',
                    action='store_true',
                    help='Do not consider the done_z_pos component in the grasp refinement task.')
parser.add_argument('--random_ycb_video_graspable_object',
                    action='store_true',
                    help='Use a random YCB-Video object.')
parser.add_argument('--ycb_video_graspable_objects_config_path',
                    action='store',
                    type=str,
                    default='../config/ycb_video_objects_graspable_poses.yaml',
                    help='Set the path of configuration file with the graspable configurations of the YCB-Video '
                         'objects.')
parser.add_argument('--random_mujoco_scanned_object',
                    action='store_true',
                    help='Use a random object from the mujoco scanned objects dataset.')
parser.add_argument('--done_moved_object_mso_angle',
                    action='store',
                    type=float,
                    default=90,
                    help='Set the rotation angle in degrees around the x/y axes to consider an object as moved when '
                         'using the mujoco scanned objects dataset.')
parser.add_argument('--mujoco_scanned_objects_config_path',
                    action='store',
                    type=str,
                    default='../config/mujoco_scanned_objects_graspable.yaml',
                    help='Set the path of configuration file with the graspable mujoco_scanned_objects.')
parser.add_argument('--objects',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify YCB-Video objects to be added to the scene. Available objects are: '
                         '002_master_chef_can, 003_cracker_box, 004_sugar_box, 005_tomato_soup_can, '
                         '006_mustard_bottle, 007_tuna_fish_can, 008_pudding_box, 009_gelatin_box, '
                         '010_potted_meat_can, 011_banana, 019_pitcher_base, 021_bleach_cleanser, 024_bowl, 025_mug, '
                         '035_power_drill, 036_wood_block, 037_scissors, 040_large_marker, 051_large_clamp, '
                         '052_extra_large_clamp, 061_foam_brick')
parser.add_argument('--use_table',
                    action='store_true',
                    help='Add table in the environment')
parser.add_argument('--fixed_initial_pos',
                    action='store_true',
                    help='Use a fixed initial position for the controlled joints and the objects in the environment.')
parser.add_argument('--objects_positions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order x_1 y_1 z_1 ... x_n y_n z_n '
                         'for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial position of all the objects is set randomly.')
parser.add_argument('--objects_quaternions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order w_1 x_1 y_1 z_1 ... w_n x_n y_n '
                         'z_n for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial orientation of all the objects is set randomly.')
parser.add_argument('--randomly_rotate_object_z_axis',
                    action='store_true',
                    help='Randomly rotate objects on the table around the z axis.')
parser.add_argument('--randomly_move_objects',
                    action='store_true',
                    help='Randomly move objects on the table.')
parser.add_argument('--task',
                    type=str,
                    default='reaching',
                    help='Set the task to perform.')
parser.add_argument('--training_components',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be trained. Choose values in r_arm, l_arm, r_wrist, l_wrist, '
                         'r_hand, l_hand, r_hand_no_thumb_oppose, l_hand_no_thumb_oppose, neck, torso, torso_yaw or '
                         'all to train all the joints.')
parser.add_argument('--ik_components',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be used for inverse kinematics computation. Choose values in '
                         'r_arm, l_arm, r_hand, l_hand, neck, torso, torso_yaw or all to use all the joints.')
parser.add_argument('--cartesian_components',
                    type=str,
                    nargs='+',
                    default=['all_ypr'],
                    help='Specify the eef components that must be used for cartesian control. Choose values in x, y, '
                         'z, for eef position. For eef orientation control, choose values in qw, qx, qy, qz for '
                         'quaternion orientation control or in yaw, pitch, roll for ypr orientation control. To '
                         'control all the position components and all the rotation components use all_ypr '
                         '(default option, with ypr orientation control) or all_quaternion (with quaternion '
                         'orientation control.')
parser.add_argument('--training_device',
                    type=str,
                    default='auto',
                    help='Set the training device. Available options are cuda, cpu or auto, which is also the default '
                         'value.')
parser.add_argument('--null_reward_out_image',
                    action='store_true',
                    help='Set reward equal to 0 for the gaze control task, if the center of mass of the object is '
                         'outside the image.')
parser.add_argument('--feature_extractor_model_name',
                    type=str,
                    default='alexnet',
                    help='Set feature extractor to process image input if features in icub_observation_space.')
parser.add_argument('--lift_object_height',
                    action='store',
                    type=float,
                    default=1.02,
                    help='Set the height of the object to complete the grasp refinement task. Default is 1.02. Note '
                         'that this parameter is not considered if the random_ycb_video_graspable_object is passed. '
                         'In that case the lift_object_height is set to 10cm above the initial position of the '
                         'object at hand.')
parser.add_argument('--learning_from_demonstration',
                    action='store_true',
                    help='Use demonstrations for replay buffer initialization.')
parser.add_argument('--max_lfd_steps',
                    action='store',
                    type=int,
                    default=10000,
                    help='Set max learning from demonstration steps for replay buffer initialization. '
                         'Default is 10000.')
parser.add_argument('--lfd_keep_only_successful_episodes',
                    action='store_true',
                    help='Store in the replay buffer only successful episodes in the learning from demonstration '
                         'phase.')
parser.add_argument('--lfd_with_approach',
                    action='store_true',
                    help='Set if the approach to the object is included in the learning from demonstration phase.')
parser.add_argument('--approach_in_reset_model',
                    action='store_true',
                    help='Approach the object when resetting the model.')
parser.add_argument('--pregrasp_distance_from_grasp_pose',
                    action='store',
                    type=float,
                    default=0.05,
                    help='Set the pre-grasp distance from the grasp pose.')
parser.add_argument('--max_delta_qpos',
                    action='store',
                    type=float,
                    default=0.1,
                    help='Set max delta qpos for joints control. Default is 0.1.')
parser.add_argument('--max_delta_cartesian_pos',
                    action='store',
                    type=float,
                    default=0.02,
                    help='Set max delta pos for cartesian control. Default is 0.02.')
parser.add_argument('--max_delta_cartesian_rot',
                    action='store',
                    type=float,
                    default=0.1,
                    help='Set max delta rot for cartesian control. Default is 0.1.')
parser.add_argument('--distanced_superq_grasp_pose',
                    action='store_true',
                    help='Start the refine grasping task from a pre-grasp pose distanced '
                         '--pregrasp_distance_from_grasp_pose from the desired grasp pose.')
parser.add_argument('--control_gaze',
                    action='store_true',
                    help='Set if using gaze control.')
parser.add_argument('--ik_solver',
                    type=str,
                    default='idyntree',
                    help='Set the IK solver between idyntree, dm_robotics, dm_control and ikin.')
parser.add_argument('--limit_torso_pitch_ikin',
                    action='store_true',
                    help='Set if using a limited range for torso_pitch joint in the iKin IK solver.')
parser.add_argument('--use_only_right_hand_model',
                    action='store_true',
                    help='Use only the right hand model instead of the whole iCub.')
parser.add_argument('--grasp_planner',
                    type=str,
                    default='superquadrics',
                    help='Set the grasp planner between superquadrics and vgn.')

args = parser.parse_args()

objects_positions = []
num_pos = 0
curr_obj_pos = np.empty(shape=0, dtype=np.float32)
for pos in args.objects_positions:
    curr_obj_pos = np.append(curr_obj_pos, pos)
    if num_pos < 2:
        num_pos += 1
    else:
        objects_positions.append(curr_obj_pos)
        num_pos = 0
        curr_obj_pos = np.empty(shape=0, dtype=np.float32)

objects_quaternions = []
num_quat = 0
curr_obj_quat = np.empty(shape=0, dtype=np.float32)
for quat in args.objects_quaternions:
    curr_obj_quat = np.append(curr_obj_quat, quat)
    if num_quat < 3:
        num_quat += 1
    else:
        objects_quaternions.append(curr_obj_quat)
        num_quat = 0
        curr_obj_quat = np.empty(shape=0, dtype=np.float32)

if args.task == 'reaching':
    iCub = ICubEnvReaching(model_path=args.xml_model_path,
                           icub_observation_space=args.icub_observation_space,
                           obs_camera=args.obs_camera,
                           track_object=args.track_object,
                           eef_name=args.eef_name,
                           render_cameras=tuple(args.render_cameras),
                           reward_goal=args.reward_goal,
                           reward_out_of_joints=args.reward_out_of_joints,
                           reward_end_timesteps=args.reward_end_timesteps,
                           reward_single_step_multiplier=args.reward_single_step_multiplier,
                           print_done_info=args.print_done_info,
                           objects=args.objects,
                           use_table=args.use_table,
                           objects_positions=objects_positions,
                           objects_quaternions=objects_quaternions,
                           random_initial_pos=not args.fixed_initial_pos,
                           training_components=args.training_components,
                           joints_margin=args.joints_margin,
                           feature_extractor_model_name=args.feature_extractor_model_name,
                           learning_from_demonstration=args.learning_from_demonstration,
                           max_lfd_steps=args.max_lfd_steps,
                           lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                           max_delta_qpos=args.max_delta_qpos,
                           max_delta_cartesian_pos=args.max_delta_cartesian_pos,
                           max_delta_cartesian_rot=args.max_delta_cartesian_rot)
elif args.task == 'gaze_control':
    iCub = ICubEnvGazeControl(model_path=args.xml_model_path,
                              icub_observation_space=args.icub_observation_space,
                              obs_camera=args.obs_camera,
                              track_object=args.track_object,
                              eef_name=args.eef_name,
                              render_cameras=tuple(args.render_cameras),
                              reward_goal=args.reward_goal,
                              reward_out_of_joints=args.reward_out_of_joints,
                              reward_end_timesteps=args.reward_end_timesteps,
                              reward_single_step_multiplier=args.reward_single_step_multiplier,
                              print_done_info=args.print_done_info,
                              objects=args.objects,
                              use_table=args.use_table,
                              objects_positions=objects_positions,
                              objects_quaternions=objects_quaternions,
                              random_initial_pos=not args.fixed_initial_pos,
                              training_components=args.training_components,
                              joints_margin=args.joints_margin,
                              null_reward_out_image=args.null_reward_out_image,
                              feature_extractor_model_name=args.feature_extractor_model_name,
                              learning_from_demonstration=args.learning_from_demonstration,
                              lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                              max_lfd_steps=args.max_lfd_steps,
                              max_delta_qpos=args.max_delta_qpos,
                              max_delta_cartesian_pos=args.max_delta_cartesian_pos,
                              max_delta_cartesian_rot=args.max_delta_cartesian_rot)
elif args.task == 'refine_grasp':
    iCub = ICubEnvRefineGrasp(model_path=args.xml_model_path,
                              initial_qpos_path=args.initial_qpos_path,
                              icub_observation_space=args.icub_observation_space,
                              obs_camera=args.obs_camera,
                              track_object=args.track_object,
                              eef_name=args.eef_name,
                              render_cameras=tuple(args.render_cameras),
                              reward_goal=args.reward_goal,
                              reward_out_of_joints=args.reward_out_of_joints,
                              reward_end_timesteps=args.reward_end_timesteps,
                              reward_single_step_multiplier=args.reward_single_step_multiplier,
                              reward_dist_superq_center=args.reward_dist_superq_center,
                              reward_line_pregrasp_superq_center=args.reward_line_pregrasp_superq_center,
                              reward_dist_original_superq_grasp_position=args.reward_dist_original_superq_grasp_position,
                              high_negative_reward_approach_failures=args.high_negative_reward_approach_failures,
                              rotated_dist_superq_center=args.rotated_dist_superq_center,
                              goal_reached_only_with_lift_refine_grasp=args.goal_reached_only_with_lift_refine_grasp,
                              exclude_vertical_touches=args.exclude_vertical_touches,
                              min_fingers_touching_object=args.min_fingers_touching_object,
                              scale_pos_lift_reward_wrt_touching_fingers=args.scale_pos_lift_reward_wrt_touching_fingers,
                              print_done_info=args.print_done_info,
                              random_ycb_video_graspable_object=args.random_ycb_video_graspable_object,
                              ycb_video_graspable_objects_config_path=args.ycb_video_graspable_objects_config_path,
                              random_mujoco_scanned_object=args.random_mujoco_scanned_object,
                              done_moved_object_mso_angle=args.done_moved_object_mso_angle,
                              mujoco_scanned_objects_config_path=args.mujoco_scanned_objects_config_path,
                              objects=args.objects,
                              use_table=args.use_table,
                              objects_positions=objects_positions,
                              objects_quaternions=objects_quaternions,
                              randomly_rotate_object_z_axis=args.randomly_rotate_object_z_axis,
                              randomly_move_objects=args.randomly_move_objects,
                              random_initial_pos=not args.fixed_initial_pos,
                              training_components=args.training_components,
                              ik_components=args.ik_components,
                              cartesian_components=args.cartesian_components,
                              joints_margin=args.joints_margin,
                              superquadrics_camera=args.superquadrics_camera,
                              feature_extractor_model_name=args.feature_extractor_model_name,
                              done_if_joints_out_of_limits=False,
                              do_not_consider_done_z_pos=args.do_not_consider_done_z_pos,
                              lift_object_height=args.lift_object_height,
                              curriculum_learning=args.curriculum_learning,
                              curriculum_learning_approach_object=args.curriculum_learning_approach_object,
                              curriculum_learning_approach_object_start_step=args.curriculum_learning_approach_object_start_step,
                              curriculum_learning_approach_object_end_step=args.curriculum_learning_approach_object_end_step,
                              learning_from_demonstration=args.learning_from_demonstration,
                              max_lfd_steps=args.max_lfd_steps,
                              lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                              lfd_with_approach=args.lfd_with_approach,
                              approach_in_reset_model=args.approach_in_reset_model,
                              pregrasp_distance_from_grasp_pose=args.pregrasp_distance_from_grasp_pose,
                              max_delta_qpos=args.max_delta_qpos,
                              max_delta_cartesian_pos=args.max_delta_cartesian_pos,
                              max_delta_cartesian_rot=args.max_delta_cartesian_rot,
                              distanced_superq_grasp_pose=args.distanced_superq_grasp_pose,
                              control_gaze=args.control_gaze,
                              ik_solver=args.ik_solver,
                              limit_torso_pitch_ikin=args.limit_torso_pitch_ikin,
                              use_only_right_hand_model=args.use_only_right_hand_model,
                              grasp_planner=args.grasp_planner,
                              pretrained_model_dir=args.pretrained_model_dir)
elif args.task == 'keep_grasp':
    iCub = ICubEnvKeepGrasp(model_path=args.xml_model_path,
                            icub_observation_space=args.icub_observation_space,
                            obs_camera=args.obs_camera,
                            track_object=args.track_object,
                            eef_name=args.eef_name,
                            render_cameras=tuple(args.render_cameras),
                            reward_goal=args.reward_goal,
                            reward_out_of_joints=args.reward_out_of_joints,
                            reward_end_timesteps=args.reward_end_timesteps,
                            reward_single_step_multiplier=args.reward_single_step_multiplier,
                            print_done_info=args.print_done_info,
                            objects=args.objects,
                            use_table=args.use_table,
                            objects_positions=objects_positions,
                            objects_quaternions=objects_quaternions,
                            random_initial_pos=not args.fixed_initial_pos,
                            training_components=args.training_components,
                            ik_components=args.ik_components,
                            cartesian_components=args.cartesian_components,
                            joints_margin=args.joints_margin,
                            superquadrics_camera=args.superquadrics_camera,
                            feature_extractor_model_name=args.feature_extractor_model_name,
                            done_if_joints_out_of_limits=False,
                            lift_object_height=args.lift_object_height,
                            curriculum_learning=args.curriculum_learning,
                            learning_from_demonstration=args.learning_from_demonstration,
                            max_lfd_steps=args.max_lfd_steps,
                            lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                            max_delta_qpos=args.max_delta_qpos,
                            max_delta_cartesian_pos=args.max_delta_cartesian_pos,
                            max_delta_cartesian_rot=args.max_delta_cartesian_rot)
elif args.task == 'lift_grasped_object':
    iCub = ICubEnvLiftGraspedObject(model_path=args.xml_model_path,
                                    icub_observation_space=args.icub_observation_space,
                                    obs_camera=args.obs_camera,
                                    track_object=args.track_object,
                                    eef_name=args.eef_name,
                                    render_cameras=tuple(args.render_cameras),
                                    reward_goal=args.reward_goal,
                                    reward_out_of_joints=args.reward_out_of_joints,
                                    reward_end_timesteps=args.reward_end_timesteps,
                                    reward_single_step_multiplier=args.reward_single_step_multiplier,
                                    print_done_info=args.print_done_info,
                                    objects=args.objects,
                                    use_table=args.use_table,
                                    objects_positions=objects_positions,
                                    objects_quaternions=objects_quaternions,
                                    random_initial_pos=not args.fixed_initial_pos,
                                    training_components=args.training_components,
                                    ik_components=args.ik_components,
                                    cartesian_components=args.cartesian_components,
                                    joints_margin=args.joints_margin,
                                    superquadrics_camera=args.superquadrics_camera,
                                    feature_extractor_model_name=args.feature_extractor_model_name,
                                    done_if_joints_out_of_limits=False,
                                    lift_object_height=args.lift_object_height,
                                    curriculum_learning=args.curriculum_learning,
                                    learning_from_demonstration=args.learning_from_demonstration,
                                    max_lfd_steps=args.max_lfd_steps,
                                    lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                                    max_delta_qpos=args.max_delta_qpos,
                                    max_delta_cartesian_pos=args.max_delta_cartesian_pos,
                                    max_delta_cartesian_rot=args.max_delta_cartesian_rot)
else:
    raise ValueError('The task specified as argument is not valid. Quitting.')

if args.test_model:
    model = SAC.load(args.eval_dir + '/best_model.zip', env=iCub)
    obs = iCub.reset()
    images = []
    # Evaluate the agent
    episode_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = iCub.step(action)
        imgs = iCub.render()
        if args.record_video:
            images.append(imgs)
        episode_reward += reward
        if done:
            break
    if args.record_video:
        print('Recording video.')
        for i in range(len(args.render_cameras)):
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(args.eval_dir + '/{}.mp4'.format(args.render_cameras[i]), fourcc, 30, (640, 480))
            for num_img, imgs in enumerate(images):
                writer.write(imgs[i][:, :, ::-1])
            writer.release()
    print("Reward:", episode_reward)
elif args.fine_tune_model:
    model = SAC.load(args.pretrained_model_dir + '/best_model.zip', env=iCub)
    model.learn(total_timesteps=args.total_training_timesteps,
                eval_freq=args.eval_freq,
                eval_env=iCub,
                eval_log_path=args.eval_dir,
                reset_num_timesteps=True)
elif args.train_with_residual_learning_pretrained_critic:
    net_arch_critic = args.net_arch_critic if len(args.net_arch_critic) > 0 else args.net_arch
    model = SAC("MultiInputPolicy",
                iCub,
                verbose=1,
                tensorboard_log=args.tensorboard_dir,
                policy_kwargs=dict(net_arch=dict(qf=net_arch_critic, pi=args.net_arch)),
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                ent_coef=args.ent_coef,
                learning_starts=args.learning_starts,
                create_eval_env=True,
                buffer_size=args.buffer_size,
                device=args.training_device,
                curriculum_learning=args.curriculum_learning,
                curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                learning_from_demonstration=args.learning_from_demonstration,
                max_lfd_steps=args.max_lfd_steps,
                lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                train_with_residual_learning_pretrained_critic=args.train_with_residual_learning_pretrained_critic,
                train_with_implicit_underparametrization_penalty=args.train_with_implicit_underparametrization_penalty,
                train_with_reptile=args.train_with_reptile,
                k_reptile=args.k_reptile,
                save_demonstrations_replay_buffers_per_object=args.save_demonstrations_replay_buffers_per_object)
    model.set_parameters(args.pretrained_model_dir + '/best_model.zip', custom_params=['critic'])
    if args.initialize_actor_mu_weights_to_zero:
        model.set_actor_mu_weights_to_zero()
    if args.load_replay_buffer:
        if args.train_with_reptile:
            model.load_replay_buffers_list(args.load_replay_buffer_path)
        else:
            model.load_replay_buffer(args.load_replay_buffer_path)

    if args.train_with_two_replay_buffers:
        model.load_demonstrations_replay_buffer(args.load_demonstrations_replay_buffer_path)
    elif args.train_with_OERLD:
        model.load_demonstrations_replay_buffer(args.load_demonstrations_replay_buffer_path)
        model.train_with_OERLD = True
    model.learn(total_timesteps=args.total_training_timesteps,
                eval_freq=args.eval_freq,
                eval_env=iCub,
                eval_log_path=args.eval_dir,
                reset_num_timesteps=True)

    if args.save_replay_buffer:
        if args.save_demonstrations_replay_buffers_per_object and args.train_with_reptile:
            model.save_replay_buffers_list(args.save_replay_buffer_path)
        else:
            model.save_replay_buffer(args.save_replay_buffer_path)
else:
    if not args.train_with_behavior_cloning and not args.train_with_AWAC:
        if ('joints' in args.icub_observation_space or 'cartesian' in args.icub_observation_space
            or 'features' in args.icub_observation_space or 'touch' in args.icub_observation_space
            or 'flare' in args.icub_observation_space or 'superquadric_center' in args.icub_observation_space
            or 'pretrained_output' in args.icub_observation_space) \
                and len(args.icub_observation_space) == 1:
            model = SAC("MlpPolicy",
                        iCub,
                        verbose=1,
                        tensorboard_log=args.tensorboard_dir,
                        policy_kwargs=dict(net_arch=args.net_arch),
                        train_freq=args.train_freq,
                        gradient_steps=args.gradient_steps,
                        ent_coef=args.ent_coef,
                        learning_starts=args.learning_starts,
                        create_eval_env=True,
                        buffer_size=args.buffer_size,
                        device=args.training_device,
                        curriculum_learning=args.curriculum_learning,
                        curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                        learning_from_demonstration=args.learning_from_demonstration,
                        max_lfd_steps=args.max_lfd_steps,
                        lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                        train_with_implicit_underparametrization_penalty=args.train_with_implicit_underparametrization_penalty,
                        train_with_reptile=args.train_with_reptile,
                        k_reptile=args.k_reptile,
                        save_demonstrations_replay_buffers_per_object=args.save_demonstrations_replay_buffers_per_object)
        elif 'camera' in args.icub_observation_space and len(args.icub_observation_space) == 1:
            model = SAC("CnnPolicy",
                        iCub,
                        verbose=1,
                        tensorboard_log=args.tensorboard_dir,
                        policy_kwargs=dict(net_arch=args.net_arch),
                        train_freq=args.train_freq,
                        gradient_steps=args.gradient_steps,
                        ent_coef=args.ent_coef,
                        learning_starts=args.learning_starts,
                        create_eval_env=True,
                        buffer_size=args.buffer_size,
                        device=args.training_device,
                        curriculum_learning=args.curriculum_learning,
                        curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                        learning_from_demonstration=args.learning_from_demonstration,
                        max_lfd_steps=args.max_lfd_steps,
                        lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                        train_with_implicit_underparametrization_penalty=args.train_with_implicit_underparametrization_penalty,
                        train_with_reptile=args.train_with_reptile,
                        k_reptile=args.k_reptile,
                        save_demonstrations_replay_buffers_per_object=args.save_demonstrations_replay_buffers_per_object)
        elif ('camera' in args.icub_observation_space
              or 'joints' in args.icub_observation_space
              or 'cartesian' in args.icub_observation_space
              or 'features' in args.icub_observation_space
              or 'touch' in args.icub_observation_space
              or 'flare' in args.icub_observation_space
              or 'pretrained_output' in args.icub_observation_space) and len(args.icub_observation_space) > 1:
            model = SAC("MultiInputPolicy",
                        iCub,
                        verbose=1,
                        tensorboard_log=args.tensorboard_dir,
                        policy_kwargs=dict(net_arch=args.net_arch),
                        train_freq=args.train_freq,
                        gradient_steps=args.gradient_steps,
                        ent_coef=args.ent_coef,
                        learning_starts=args.learning_starts,
                        create_eval_env=True,
                        buffer_size=args.buffer_size,
                        device=args.training_device,
                        curriculum_learning=args.curriculum_learning,
                        curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                        learning_from_demonstration=args.learning_from_demonstration,
                        max_lfd_steps=args.max_lfd_steps,
                        lfd_keep_only_successful_episodes=args.lfd_keep_only_successful_episodes,
                        train_with_implicit_underparametrization_penalty=args.train_with_implicit_underparametrization_penalty,
                        train_with_reptile=args.train_with_reptile,
                        k_reptile=args.k_reptile,
                        save_demonstrations_replay_buffers_per_object=args.save_demonstrations_replay_buffers_per_object)
        else:
            raise ValueError('The observation space specified as argument is not valid. Quitting.')

        if args.load_replay_buffer:
            if args.train_with_reptile:
                model.load_replay_buffers_list(args.load_replay_buffer_path)
            else:
                model.load_replay_buffer(args.load_replay_buffer_path)

        if args.train_with_two_replay_buffers:
            model.load_demonstrations_replay_buffer(args.load_demonstrations_replay_buffer_path)
        elif args.train_with_OERLD:
            model.load_demonstrations_replay_buffer(args.load_demonstrations_replay_buffer_path)
            model.train_with_OERLD = True

        model.learn(total_timesteps=args.total_training_timesteps,
                    eval_freq=args.eval_freq,
                    eval_env=iCub,
                    eval_log_path=args.eval_dir)

        if args.save_replay_buffer:
            if args.save_demonstrations_replay_buffers_per_object and args.train_with_reptile:
                model.save_replay_buffers_list(args.save_replay_buffer_path)
            else:
                model.save_replay_buffer(args.save_replay_buffer_path)
    elif args.train_with_AWAC:
        # Load replay buffer and convert it to a mdp dataset
        rb = load_from_pkl(args.load_demonstrations_replay_buffer_path)
        total_obs_size = 0
        for obs_key in rb.observations.keys():
            total_obs_size += rb.observations[obs_key][0].squeeze().size
        transitions_obs = np.empty((rb.buffer_size, total_obs_size))
        obs_start_id = 0
        for obs_key in rb.observations.keys():
            obs_end_id = obs_start_id + rb.observations[obs_key][0].squeeze().size
            transitions_obs[:, obs_start_id:obs_end_id] = rb.observations[obs_key].squeeze()
            obs_start_id = obs_end_id
        rb.observations = np.expand_dims(transitions_obs, 1)

        # Convert to d3rlpy MDPDataset
        dataset = to_mdp_dataset(rb)

        actor = VectorEncoderFactory(hidden_units=args.net_arch, activation='relu')
        critic = VectorEncoderFactory(hidden_units=args.net_arch, activation='relu')

        # Initialize
        awac = AWAC(actor_encoder_factory=actor,
                    critic_encoder_factory=critic,
                    batch_size=256,
                    lam=0.3,
                    use_gpu=True)

        # Train offline
        awac.fit(dataset,
                 n_steps=25000,
                 n_steps_per_epoch=25000,
                 tensorboard_dir=args.tensorboard_dir)

        buffer = ReplayBuffer(maxlen=args.buffer_size, episodes=dataset.episodes)

        # Train online
        awac.fit_online(iCub,
                        n_steps=args.total_training_timesteps,
                        n_steps_per_epoch=100000,
                        buffer=buffer,
                        tensorboard_dir=args.tensorboard_dir)
    else:
        # Load replay buffer and adapt it to the format required for behavior cloning
        rb = load_from_pkl(args.load_replay_buffer_path)
        total_obs_size = 0
        for obs_key in rb.observations.keys():
            total_obs_size += rb.observations[obs_key][0].squeeze().size
        transitions_obs = np.empty((rb.buffer_size, total_obs_size))
        transitions_next_obs = np.empty((rb.buffer_size, total_obs_size))
        obs_start_id = 0
        lows_obs_space = np.empty((0,))
        highs_obs_space = np.empty((0,))
        for obs_key in rb.observations.keys():
            obs_end_id = obs_start_id + rb.observations[obs_key][0].squeeze().size
            transitions_obs[:, obs_start_id:obs_end_id] = rb.observations[obs_key].squeeze()
            transitions_next_obs[:, obs_start_id:obs_end_id] = rb.next_observations[obs_key].squeeze()
            obs_start_id = obs_end_id
            lows_obs_space = np.concatenate((lows_obs_space, rb.observation_space[obs_key].low.squeeze()))
            highs_obs_space = np.concatenate((highs_obs_space, rb.observation_space[obs_key].high.squeeze()))

        transitions = Transitions(obs=transitions_obs,
                                  acts=rb.actions,
                                  infos=np.array([{}] * rb.buffer_size),
                                  next_obs=transitions_next_obs,
                                  dones=rb.dones.squeeze().astype(bool))
        obs_space = gym.spaces.Box(low=lows_obs_space, high=highs_obs_space)
        action_space = rb.action_space

        # Refer to https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/bc.py#L324 for
        # lr_schedule setting
        policy = policies.ActorCriticPolicy(observation_space=obs_space,
                                            action_space=action_space,
                                            net_arch=args.net_arch,
                                            # Set lr_schedule to max value to force error if policy.optimizer
                                            # is used by mistake (should use self.optimizer instead).
                                            lr_schedule=lambda _: torch.finfo(torch.float32).max,
                                            activation_fn=torch.nn.ReLU)

        # Initialize the trainer
        bc_trainer = bc.BC(
            observation_space=obs_space,
            action_space=action_space,
            demonstrations=transitions,
            policy=policy
        )

        bc_trainer.train(n_epochs=args.behavior_cloning_epochs)

        reward, _ = evaluate_policy_bc(bc_trainer.policy, env=iCub, n_eval_episodes=100, render=False)
        print(f"Reward after training: {reward}")
