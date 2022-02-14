from dm_robotics.manipulation.props import mesh_object
import glob
import os

# http://berkcalli.com/publications/2015-CalliSinghWalsmanDollar.pdf
masses = {"002_master_chef_can": 0.414,
          "003_cracker_box": 0.411,
          "004_sugar_box": 0.514,
          "005_tomato_soup_can": 0.349,
          "006_mustard_bottle": 0.603,
          "007_tuna_fish_can": 0.171,
          "008_pudding_box": 0.187,
          "009_gelatin_box": 0.097,
          "010_potted_meat_can": 0.370,
          "011_banana": 0.066,
          "019_pitcher_base": 0.178,
          "021_bleach_cleanser": 1.131,
          "024_bowl": 0.147,
          "025_mug": 0.118,
          "035_power_drill": 0.895,
          "036_wood_block": 0.729,
          "037_scissors": 0.082,
          "040_large_marker": 0.0158,
          "051_large_clamp": 0.125,
          "052_extra_large_clamp": 0.202,
          "061_foam_brick": 0.028
          }


def ycb_video_obj_to_msh():
    mesh_path = "../meshes/YCB_Video/models/*/textured.obj"
    mesh_files = []
    for mesh in sorted(glob.glob(mesh_path)):
        mesh_files.append(mesh)

    texture_path = "../meshes/YCB_Video/models/*/texture_map.png"
    texture_files = []
    for texture in sorted(glob.glob(texture_path)):
        texture_files.append(texture)

    for mesh, texture in zip(mesh_files, texture_files):
        obj_name = mesh.split(os.sep)[-2]
        print('Creating mesh for {}.'.format(obj_name))
        mesh_object.MeshProp(name=obj_name,
                             visual_meshes=[mesh],
                             texture_file=texture,
                             mjcf_model_export_dir="../meshes/YCB_Video/",
                             masses=[masses[obj_name]],
                             size=[1, 1, 1])


ycb_video_obj_to_msh()
