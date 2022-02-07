from icub_mujoco.envs.icub_visuomanip import ICubEnv
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
                    help='Set the multiplication factor of the default per-step reward in meters.')
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
                    action='store',
                    type=str,
                    default='joints',
                    help='Set the observation space: joints will use as observation space joints positions,'
                         'camera will use realsense headset information.')
parser.add_argument('--render_cameras',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Set the cameras used for rendering. Available cameras are front_cam and head_cam.')
parser.add_argument('--print_done_info',
                    action='store_true',
                    help='Print information at the end of each episode')


args = parser.parse_args()

iCub = ICubEnv(model_path=args.xml_model_path,
               obs_from_img=args.icub_observation_space == 'camera',
               render_cameras=tuple(args.render_cameras),
               reward_goal=args.reward_goal,
               reward_out_of_joints=args.reward_out_of_joints,
               reward_single_step_multiplier=args.reward_single_step_multiplier,
               print_done_info=args.print_done_info)

model = SAC("MlpPolicy",
            iCub,
            verbose=1,
            tensorboard_log=args.tensorboard_dir,
            policy_kwargs=dict(net_arch=args.net_arch),
            train_freq=args.train_freq,
            create_eval_env=True)
model.learn(total_timesteps=args.total_training_timesteps,
            eval_freq=args.eval_freq,
            eval_env=iCub,
            eval_log_path=args.eval_dir)
