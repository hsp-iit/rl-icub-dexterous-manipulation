from dm_control import mujoco, viewer, composer, mjcf
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render_from_camera',
                    action='store_true',
                    help='Set the parameter to render head camera images and display them with OpenCV. As default use '
                         'dm_control viewer.')
args = parser.parse_args()

path = '../models/icub_position_actuators.xml'

physics = mujoco.Physics.from_xml_path(path)
world = mjcf.from_path(path)
world_entity = composer.ModelWrapperEntity(world)
task = composer.NullTask(world_entity)
env = composer.Environment(task)

actuator_names = [actuator.name for actuator in world_entity.mjcf_model.find_all('actuator')]
joint_names = [joint.name for joint in world_entity.mjcf_model.find_all('joint')]
map_joint_to_actuators = []
for actuator in actuator_names:
    map_joint_to_actuators.append(joint_names.index(actuator))

if args.render_from_camera:
    for _ in range(10000):
        env.step(np.array([0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.48, 0.023, -0.005, -1.05, -0.57, -0.024,
                           0.1, 0.0, 0.0, -0.159, 2.0, 0.183, 0.54, 0.0, 0., 0., 0.7854, 0.7854,
                           0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854,
                           0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, -0.159, 2.0, 0.183, 0.54, 0.0, 0.0,
                           0.0, 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854,
                           0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])[map_joint_to_actuators])
        img = env.physics.render(height=480, width=640, camera_id='head_cam')
        cv2.imshow('cam_view', img[:, :, ::-1])
        cv2.waitKey(1)
else:
    # Define a policy to reach the initial state.
    def policy(time_step):
        del time_step  # Unused.
        ctrl = np.array([0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.48, 0.023, -0.005, -1.05, -0.57, -0.024,
                         0.1, 0.0, 0.0, -0.159, 2.0, 0.183, 0.54, 0.0, 0., 0., 0.7854, 0.7854,
                         0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854,
                         0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, -0.159, 2.0, 0.183, 0.54, 0.0, 0.0,
                         0.0, 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854,
                         0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0])[map_joint_to_actuators]
        return ctrl


    viewer.launch(env, policy=policy)
