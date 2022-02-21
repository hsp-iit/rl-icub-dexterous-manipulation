from dm_control import viewer, composer, mjcf
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render_from_camera',
                    action='store_true',
                    help='Set the parameter to render head camera images and display them with OpenCV. As default use '
                         'dm_control viewer.')
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
parser.add_argument('--objects_positions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order x_1 y_1 z_1 ... x_n y_n z_n '
                         'for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial position of all the objects is set '
                         'to -1 0 2.')
parser.add_argument('--objects_quaternions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order w_1 x_1 y_1 z_1 ... w_n x_n y_n '
                         'z_n for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial orientation of all the objects is set '
                         'to 1 0 0 0.')
parser.add_argument('--table',
                    action='store_true',
                    help='Add table to the scene')
parser.add_argument('--print_contact_geoms',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Print if the geoms passed as argument collide.')

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

path = '../models/icub_position_actuators.xml'

world = mjcf.from_path(path)
for id, obj in enumerate(args.objects):
    obj_path = "../meshes/YCB_Video/{}.xml".format(obj)
    obj_mjcf = mjcf.from_path(obj_path, escape_separators=True)
    world.attach(obj_mjcf.root_model)
    world.worldbody.body[len(world.worldbody.body)-1].pos = objects_positions[id] if objects_positions else "-1 0 2"
    world.worldbody.body[len(world.worldbody.body)-1].quat = \
        objects_quaternions[id] if objects_quaternions else "1 0 0 0"
    world.worldbody.body[len(world.worldbody.body)-1].add('joint',
                                                          name=obj,
                                                          type="free",
                                                          pos="0 0 0",
                                                          limited="false",
                                                          damping="0.0",
                                                          stiffness="0.01")

if args.table:
    table_path = "../models/table.xml"
    table_mjcf = mjcf.from_path(table_path, escape_separators=False)
    world.attach(table_mjcf.root_model)

world_entity = composer.ModelWrapperEntity(world)
task = composer.NullTask(world_entity)
env = composer.Environment(task)

actuator_names = [actuator.name for actuator in world_entity.mjcf_model.find_all('actuator')]
joint_names = [joint.name for joint in world_entity.mjcf_model.find_all('joint')]
map_joint_to_actuators = []
for actuator in actuator_names:
    map_joint_to_actuators.append(joint_names.index(actuator))

# Map names of contact geoms to their id and extract couples
contact_geoms_ids = []
for geom_name in args.print_contact_geoms:
    geom_id = env.physics.model.name2id(geom_name, 'geom')
    contact_geoms_ids.append(geom_id)

if args.render_from_camera:
    for _ in range(10000):
        env.step(np.array([0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.48, 0.023, -0.005, -1.05, -0.57, -0.024,
                           0.1, 0.0, 0.0, -0.159, 2.0, 0.183, 0.54, 0.0, 0., 0., 0.7854, 0.7854,
                           0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854,
                           0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, -0.159, 2.0, 0.183, 0.54, 0.0, 0.0,
                           0.0, 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854,
                           0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.0, 0.0, 0.0
                           ])[map_joint_to_actuators])
        img = env.physics.render(height=480, width=640, camera_id='head_cam')
        cv2.imshow('cam_view', img[:, :, ::-1])
        cv2.waitKey(1)
        if contact_geoms_ids:
            for contact in env.physics.data.contact:
                if contact['geom1'] in contact_geoms_ids and contact['geom2'] in contact_geoms_ids:
                    print('Collision between geom {} and geom {}.'
                          .format(args.print_contact_geoms[contact_geoms_ids.index(contact['geom1'])],
                                  args.print_contact_geoms[contact_geoms_ids.index(contact['geom2'])]))

else:
    # Define a policy to reach the initial state.
    def policy(time_step):
        del time_step  # Unused.
        ctrl = np.array([0.48, 0.023, -0.005, -1.05, -0.57, -0.024, 0.48, 0.023, -0.005, -1.05, -0.57, -0.024,
                         0.1, 0.0, 0.0, -0.159, 2.0, 0.183, 0.54, 0.0, 0., 0., 0.7854, 0.7854,
                         0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854,
                         0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, -0.159, 2.0, 0.183, 0.54, 0.0, 0.0,
                         0.0, 0.7854, 0.7854, 0.7854, 0.7854, -0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854,
                         0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.15, 0.7854, 0.7854, 0.7854, 0.0, 0.0, 0.0
                         ])[map_joint_to_actuators]
        if contact_geoms_ids:
            for contact in env.physics.data.contact:
                if contact['geom1'] in contact_geoms_ids and contact['geom2'] in contact_geoms_ids:
                    print('Collision between geom {} and geom {}.'
                          .format(args.print_contact_geoms[contact_geoms_ids.index(contact['geom1'])],
                                  args.print_contact_geoms[contact_geoms_ids.index(contact['geom2'])]))
        return ctrl


    viewer.launch(env, policy=policy)
