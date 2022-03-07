from dm_control import viewer, composer, mjcf
from dm_robotics.moma.utils.ik_solver import IkSolver
from dm_robotics.geometry import geometry
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--controllable_joints',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be controlled. Choose values in r_arm, l_arm, neck, torso, '
                         'torso_yaw or all to train all the joints.')
parser.add_argument('--eef_body_name',
                    type=str,
                    default='r_hand',
                    help='Specify the name of the body to be considered as end-effector')
parser.add_argument('--eef_positions',
                    type=float,
                    nargs='+',
                    default=[-0.3, 0.1, 1.2],
                    help='Specify the cartesian positions to be reached by the eef. They must be in the order x_1 y_1 '
                         'z_1 ... x_n y_n z_n for the n desired goal positions.')
parser.add_argument('--eef_quaternions',
                    type=float,
                    nargs='+',
                    default=[1, 0, 0, 0],
                    help='Specify the cartesian orientations in quaternions to be reached by the eef. They must be in '
                         'the order w_1 x_1 y_1 z_1 ... w_n x_n y_n z_n for the n desired goal orientations.')
parser.add_argument('--verbose',
                    action='store_true',
                    help='Print debug information about xpos, xrot, qpos.')

args = parser.parse_args()

path = '../models/icub_position_actuators.xml'

world = mjcf.from_path(path)
world_entity = composer.ModelWrapperEntity(world)
task = composer.NullTask(world_entity)
env = composer.Environment(task)

actuator_names = [actuator.name for actuator in world_entity.mjcf_model.find_all('actuator')]
joint_names = [joint.name for joint in world_entity.mjcf_model.find_all('joint')]
map_joint_to_actuators = []
for actuator in actuator_names:
    map_joint_to_actuators.append(joint_names.index(actuator))

joints_to_control = []
if 'r_arm' in args.controllable_joints:
    joints_to_control.extend([j for j in joint_names if (j.startswith('r_wrist') or
                                                         j.startswith('r_elbow') or
                                                         j.startswith('r_shoulder'))])
if 'l_arm' in args.controllable_joints:
    joints_to_control.extend([j for j in joint_names if (j.startswith('l_wrist') or
                                                         j.startswith('l_elbow') or
                                                         j.startswith('l_shoulder'))])
if 'neck' in args.controllable_joints:
    joints_to_control.extend([j for j in joint_names if j.startswith('neck')])
if 'torso' in args.controllable_joints:
    joints_to_control.extend([j for j in joint_names if j.startswith('torso')])
if 'torso_yaw' in args.controllable_joints and 'torso' not in args.controllable_joints:
    joints_to_control.extend([j for j in joint_names if j.startswith('torso_yaw')])
if 'all' in args.controllable_joints and len(args.controllable_joints) == 1:
    joints_to_control.extend([j for j in joint_names])

controllable_joints = [j for j in world_entity.mjcf_model.find_all('joint') if j.name in joints_to_control]

joints_ids = []
for joint in controllable_joints:
    joints_ids.append(env.physics.model.name2id(joint.name, 'joint'))

ref_qpos = np.array([0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.1, 0.0,
                     0.0, -0.159, 2.0, 0.183, 0.54, 0.0, 0., 0., 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854,
                     0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854,
                     -0.159, 2.0, 0.183, 0.54, 0.0, 0.0, 0.0, 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854,
                     0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854,
                     0.0, 0.0, 0.0])

id_eef_world_entity = [body.name == args.eef_body_name for body in world_entity.mjcf_model.find_all('body')].index(True)
solver = IkSolver(world_entity.mjcf_model,
                  controllable_joints=controllable_joints,
                  element=world_entity.mjcf_model.find_all('body')[id_eef_world_entity]
                  )

assert(4*len(args.eef_positions) == 3*len(args.eef_quaternions))

eef_positions = []
num_pos = 0
curr_eef_pos = []
for pos in args.eef_positions:
    curr_eef_pos.append(pos)
    if num_pos < 2:
        num_pos += 1
    else:
        eef_positions.append(curr_eef_pos)
        num_pos = 0
        curr_eef_pos = []

eef_quaternions = []
num_quat = 0
curr_eef_quat = []
for quat in args.eef_quaternions:
    curr_eef_quat.append(quat)
    if num_quat < 3:
        num_quat += 1
    else:
        eef_quaternions.append(curr_eef_quat)
        num_quat = 0
        curr_eef_quat = []

for eef_position, eef_quaternion in zip(eef_positions, eef_quaternions):
    ref_pos = geometry.Pose(position=eef_position, quaternion=eef_quaternion)
    qpos_sol = solver.solve(ref_pos)

    def policy(time_step):
        id_eef_from_physics = env.physics.model.name2id(args.eef_body_name, 'body')
        del time_step  # Unused.
        ctrl = ref_qpos.copy()
        ctrl[joints_ids] = qpos_sol
        ctrl = ctrl[map_joint_to_actuators]
        if args.verbose:
            state_dict = {'Current xpos': env.physics.data.xpos[id_eef_from_physics],
                          'Target xpos': eef_position,
                          'Current xrot': R.from_matrix(np.reshape(env.physics.data.xmat[id_eef_from_physics],
                                                                   (3, 3))).as_quat()[[3, 0, 1, 2]],
                          'Target xrot': eef_quaternion,
                          'Current qpos': env.physics.data.qpos[joints_ids],
                          'Target qpos': qpos_sol,
                          'Controlled joints': np.array(joint_names)[joints_ids]}
            print(state_dict)
        return ctrl
    viewer.launch(env, policy=policy)
