from icub_mujoco.envs.icub_visuomanip_reaching import ICubEnvReaching
from icub_mujoco.envs.icub_visuomanip_gaze_control import ICubEnvGazeControl
from stable_baselines3 import SAC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_model_path',
                    action='store',
                    type=str,
                    default='../models/icub_position_actuators.xml',
                    help='Set the path of the xml model.')
parser.add_argument('--tensorboard_dir',
                    action='store',
                    type=str,
                    default='tensorboards',
                    help='Set the directory where tensorboard files are saved. Default directory is tensorboards.')
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
parser.add_argument('--reward_single_step_multiplier',
                    action='store',
                    type=float,
                    default=10.0,
                    help='Set the multiplication factor of the default per-step reward in meters or pixels.')
parser.add_argument('--net_arch',
                    type=int,
                    nargs='+',
                    default=[64, 64],
                    help='Set the architecture of the MLP network. Default is [64, 64]')
parser.add_argument('--train_freq',
                    action='store',
                    type=int,
                    default=10,
                    help='Set the update frequency for SAC. Default is 10')
parser.add_argument('--total_training_timesteps',
                    action='store',
                    type=int,
                    default=10000000,
                    help='Set the number of training episodes for SAC. Default is 10M')
parser.add_argument('--eval_dir',
                    action='store',
                    type=str,
                    default='logs_eval',
                    help='Set the directory where evaluation files are saved. Default directory is logs_eval.')
parser.add_argument('--eval_freq',
                    action='store',
                    type=int,
                    default=100000,
                    help='Set the evaluation frequency for SAC. Default is 100k')
parser.add_argument('--icub_observation_space',
                    type=str,
                    nargs='+',
                    default='joints',
                    help='Set the observation space: joints will use as observation space joints positions,'
                         'camera will use realsense headset information. If you pass both as argument, you will '
                         'use a MultiInputPolicy.')
parser.add_argument('--render_cameras',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Set the cameras used for rendering. Available cameras are front_cam and head_cam.')
parser.add_argument('--obs_camera',
                    type=str,
                    default='head_cam',
                    help='Set the cameras used for observation. Available cameras are front_cam and head_cam.')
parser.add_argument('--print_done_info',
                    action='store_true',
                    help='Print information at the end of each episode')
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
parser.add_argument('--task',
                    type=str,
                    default='reaching',
                    help='Set the task to perform.')
parser.add_argument('--training_components',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be trained. Choose values in r_arm, l_arm, neck, torso, '
                         'torso_yaw or all to train all the joints.')

args = parser.parse_args()

objects_positions = []
num_pos = 0
curr_obj_pos = ''
for pos in args.objects_positions:
    curr_obj_pos += pos
    if num_pos < 2:
        curr_obj_pos += ' '
        num_pos += 1
    else:
        objects_positions.append(curr_obj_pos)
        num_pos = 0
        curr_obj_pos = ''

objects_quaternions = []
num_quat = 0
curr_obj_quat = ''
for quat in args.objects_quaternions:
    curr_obj_quat += quat
    if num_quat < 3:
        curr_obj_quat += ' '
        num_quat += 1
    else:
        objects_quaternions.append(curr_obj_quat)
        num_quat = 0
        curr_obj_quat = ''

if args.task == 'reaching':
    iCub = ICubEnvReaching(model_path=args.xml_model_path,
                           icub_observation_space=args.icub_observation_space,
                           obs_camera=args.obs_camera,
                           render_cameras=tuple(args.render_cameras),
                           reward_goal=args.reward_goal,
                           reward_out_of_joints=args.reward_out_of_joints,
                           reward_single_step_multiplier=args.reward_single_step_multiplier,
                           print_done_info=args.print_done_info,
                           objects=args.objects,
                           use_table=args.use_table,
                           objects_positions=objects_positions,
                           objects_quaternions=objects_quaternions,
                           random_initial_pos=not args.fixed_initial_pos,
                           training_components=args.training_components)
elif args.task == 'gaze_control':
    iCub = ICubEnvGazeControl(model_path=args.xml_model_path,
                              icub_observation_space=args.icub_observation_space,
                              obs_camera=args.obs_camera,
                              render_cameras=tuple(args.render_cameras),
                              reward_goal=args.reward_goal,
                              reward_out_of_joints=args.reward_out_of_joints,
                              reward_single_step_multiplier=args.reward_single_step_multiplier,
                              print_done_info=args.print_done_info,
                              objects=args.objects,
                              use_table=args.use_table,
                              objects_positions=objects_positions,
                              objects_quaternions=objects_quaternions,
                              random_initial_pos=not args.fixed_initial_pos,
                              training_components=args.training_components)
else:
    raise ValueError('The task specified as argument is not valid. Quitting.')

if 'joints' in args.icub_observation_space and 'camera' not in args.icub_observation_space:
    model = SAC("MlpPolicy",
                iCub,
                verbose=1,
                tensorboard_log=args.tensorboard_dir,
                policy_kwargs=dict(net_arch=args.net_arch),
                train_freq=args.train_freq,
                create_eval_env=True)
elif 'camera' in args.icub_observation_space and 'joints' not in args.icub_observation_space:
    model = SAC("CnnPolicy",
                iCub,
                verbose=1,
                tensorboard_log=args.tensorboard_dir,
                policy_kwargs=dict(net_arch=args.net_arch),
                train_freq=args.train_freq,
                create_eval_env=True,
                buffer_size=1000)
elif 'camera' in args.icub_observation_space and 'joints' in args.icub_observation_space:
    model = SAC("MultiInputPolicy",
                iCub,
                verbose=1,
                tensorboard_log=args.tensorboard_dir,
                policy_kwargs=dict(net_arch=args.net_arch),
                train_freq=args.train_freq,
                create_eval_env=True,
                buffer_size=1000)
else:
    raise ValueError('The observation space specified as argument is not valid. Quitting.')

model.learn(total_timesteps=args.total_training_timesteps,
            eval_freq=args.eval_freq,
            eval_env=iCub,
            eval_log_path=args.eval_dir)
