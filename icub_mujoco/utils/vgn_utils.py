# Code adapted from https://github.com/ethz-asl/vgn
import sys
import numpy as np
import math
from pyquaternion import Quaternion
from icub_mujoco.external.vgn.src.vgn.perception import *
from icub_mujoco.external.vgn.src.vgn.grasp import *
from icub_mujoco.external.vgn.src.vgn.utils.transform import *
from icub_mujoco.external.vgn.src.vgn.networks import get_network, load_network
import torch
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from icub_mujoco.utils.ikin_ik import IKinIK



class VGNEstimator:

    def __init__(self,
                 distance_from_grasp_pose_distanced_position,
                 iKin_joints_to_control_names):
        self.distance_from_grasp_pose_disanced_position = distance_from_grasp_pose_distanced_position
        self.cam_intrinsic = CameraIntrinsic(640, 480, 617.783447265625, 617.783447265625, 320.0, 240.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = get_network('conv').to(self.device)
        self.net.load_state_dict(torch.load("../external/vgn/data/models/vgn_conv.pth", map_location=self.device))
        self.medianx = 0
        self.mediany = 0
        self.medianz = 0
        if len(iKin_joints_to_control_names) > 0:
            self.iKin_IK_solver = IKinIK(iKin_joints_to_control_names)
        else:
            self.iKin_IK_solver = None
        self.pcd = None
        self.depth = None
        self.cam_pos = None
        self.cam_rot = None
        self.ids = None

    def compute_grasp_pose_vgn(self,
                               pcd,
                               depth,
                               camera_pos,
                               camera_rot,
                               ids,
                               iKin_current_qpos,
                               iKin_joints_to_control_ik_ids):
        self.pcd = pcd
        self.depth = depth
        self.cam_pos = camera_pos
        self.cam_rot = camera_rot
        self.ids = ids
        tsdf = self.compute_tsdf()
        grasps, scores = self.compute_grasps_from_tsdf(tsdf)
        self.pos_grasp_icub = None
        self.rot_grasp_icub = None
        min_err = 30 / 180 * np.pi
        for i in range(len(scores)):
            if grasps[i].pose.translation[2] <= 0.05:
                continue
            else:
                grasp = grasps[i]
                score = scores[i]
            pos_grasp_icub_0, rot_grasp_icub_0, pos_grasp_icub_1, rot_grasp_icub_1 = \
                self.compute_icub_grasp_from_panda_grasp(grasp)
            if self.iKin_IK_solver is None:
                rand_id = np.random.randint(0, 2)
                if rand_id == 0:
                    self.pos_grasp_icub, self.rot_grasp_icub = pos_grasp_icub_0, rot_grasp_icub_0
                else:
                    self.pos_grasp_icub, self.rot_grasp_icub = pos_grasp_icub_1, rot_grasp_icub_1
                break
            # Check configuration 0
            target_ik_pyquaternion = Quaternion(matrix=rot_grasp_icub_0)
            target_ik_axis_angle = np.append(target_ik_pyquaternion.axis, target_ik_pyquaternion.angle)
            ik_sol, solved = self.iKin_IK_solver.solve_ik(eef_pos=pos_grasp_icub_0,
                                                          eef_axis_angle=target_ik_axis_angle,
                                                          current_qpos=iKin_current_qpos,
                                                          joints_to_control_ik_ids=iKin_joints_to_control_ik_ids,
                                                          on_step=False)
            if solved:
                err_0 = self.compute_angular_error(rot_grasp_icub_0)
            else:
                # Set a value > min_err s.t. the solution is not considered
                err_0 = min_err + 1
            # Check configuration 1
            target_ik_pyquaternion = Quaternion(matrix=rot_grasp_icub_1)
            target_ik_axis_angle = np.append(target_ik_pyquaternion.axis, target_ik_pyquaternion.angle)
            ik_sol, solved = self.iKin_IK_solver.solve_ik(eef_pos=pos_grasp_icub_1,
                                                          eef_axis_angle=target_ik_axis_angle,
                                                          current_qpos=iKin_current_qpos,
                                                          joints_to_control_ik_ids=iKin_joints_to_control_ik_ids,
                                                          on_step=False)
            if solved:
                err_1 = self.compute_angular_error(rot_grasp_icub_1)
            else:
                # Set a value > min_err s.t. the solution is not considered
                err_1 = min_err + 1
            if err_0 < err_1 and err_0 < min_err:
                self.pos_grasp_icub, self.rot_grasp_icub = pos_grasp_icub_0, rot_grasp_icub_0
                break
            elif err_1 < err_0 and err_1 < min_err:
                self.pos_grasp_icub, self.rot_grasp_icub = pos_grasp_icub_1, rot_grasp_icub_1
                break
        return self.set_grasp_pose_to_superquadric_format()

    def compute_tsdf(self):
        tsdf = TSDFVolume(0.3, 40)
        depth_img_orig = self.depth
        depth_img = np.full(depth_img_orig.shape, 1000.0, dtype=depth_img_orig.dtype)
        for id in zip(self.ids[0], self.ids[1]):
            depth_img[id] = depth_img_orig[id]
        # Compute median values of the point cloud to keep the object in the center of the vgn workspace
        self.medianx = np.median(self.pcd[:, 0])
        self.mediany = np.median(self.pcd[:, 1])
        self.medianz = np.median(self.pcd[:, 2]) + 0.95
        # Compute camera pose in the vgn reference frame
        cam_pos_orig = self.cam_pos.copy()
        self.cam_pos[0] = cam_pos_orig[1] - self.mediany + 0.15
        self.cam_pos[1] = -cam_pos_orig[0] + self.medianx + 0.15
        self.cam_pos[2] -= 0.95
        # Rotate cam_rot of 90 degrees to align icub reference system to the default of vgn
        self.cam_rot = np.matmul(np.array([[0, 1, 0],
                                           [-1, 0, 0],
                                           [0, 0, 1]]), self.cam_rot)
        cam_rototransl = np.identity(4)
        cam_rototransl[:3, :3] = self.cam_rot
        cam_rototransl[:3, 3] = self.cam_pos
        # 180° rotation around the x axis required to align o3d and mujoco camera coordinate systems
        cam_rototransl = np.linalg.inv(np.matmul(cam_rototransl, np.array([[1, 0, 0, 0],
                                                                           [0, -1, 0, 0],
                                                                           [0, 0, -1, 0],
                                                                           [0, 0, 0, 1]])))

        cam_rot_ext = cam_rototransl[:3, :3]
        cam_pos_ext = cam_rototransl[:3, 3]
        extrinsic = Transform(R.from_matrix(cam_rot_ext), cam_pos_ext)
        tsdf.integrate(depth_img, self.cam_intrinsic, extrinsic)

        return tsdf

    def compute_grasps_from_tsdf(self, tsdf):
        tsdf_vol = tsdf.get_grid()
        voxel_size = tsdf.voxel_size
        qual_vol, rot_vol, width_vol = self.predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = self.process(tsdf_vol, qual_vol, rot_vol, width_vol)
        grasps, scores = self.select(qual_vol.copy(), rot_vol, width_vol, threshold=0.7, max_filter_size=2)
        grasps, scores = np.asarray(grasps), np.asarray(scores)
        # Return grasps sorted by score
        if len(grasps) > 0:
            p = np.argsort(scores)[::-1]
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        return grasps, scores

    @staticmethod
    def axisangle_to_quat(axisangle):
        s = math.sin(axisangle[3] / 2)
        x = axisangle[0] * s
        y = axisangle[1] * s
        z = axisangle[2] * s
        w = math.cos(axisangle[3] / 2)
        return np.array([w, x, y, z])

    @staticmethod
    def line_coefficients_between_two_point(point1, point2):
        if len(point1) != 3 or len(point2) != 3:
            raise ValueError('Points must have length 3.')
        point1 = np.array(point1)
        point2 = np.array(point2)
        dist = np.linalg.norm(point2 - point1)
        mx = point2[0] - point1[0]
        my = point2[1] - point1[1]
        mz = point2[2] - point1[2]
        return mx/dist, my/dist, mz/dist

    @staticmethod
    def predict(tsdf_vol, net, device):
        assert tsdf_vol.shape == (1, 40, 40, 40)

        # move input to the GPU
        tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

        # forward pass
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = net(tsdf_vol)

        # move output back to the CPU
        qual_vol = qual_vol.cpu().squeeze().numpy()
        rot_vol = rot_vol.cpu().squeeze().numpy()
        width_vol = width_vol.cpu().squeeze().numpy()
        return qual_vol, rot_vol, width_vol

    @staticmethod
    def process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=1.0,
        min_width=1.33,
        max_width=9.33,
    ):
        tsdf_vol = tsdf_vol.squeeze()

        # smooth quality volume with a Gaussian
        qual_vol = ndimage.gaussian_filter(
            qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
        )

        # mask out voxels too far away from the surface
        outside_voxels = tsdf_vol > 0.5
        inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
        valid_voxels = ndimage.binary_dilation(
            outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
        )
        qual_vol[valid_voxels==False] = 0.0

        # reject voxels with predicted widths that are too small or too large
        qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

        return qual_vol, rot_vol, width_vol

    def select(self, qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

        # non maximum suppression
        max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
        qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
        mask = np.where(qual_vol, 1.0, 0.0)

        # construct grasps
        grasps, scores = [], []
        for index in np.argwhere(mask):
            grasp, score = self.select_index(qual_vol, rot_vol, width_vol, index)
            grasps.append(grasp)
            scores.append(score)

        return grasps, scores

    @staticmethod
    def select_index(qual_vol, rot_vol, width_vol, index):
        i, j, k = index
        score = qual_vol[i, j, k]
        ori = Rotation.from_quat(rot_vol[:, i, j, k])
        pos = np.array([i, j, k], dtype=np.float64)
        width = width_vol[i, j, k]
        return Grasp(Transform(ori, pos), width), score

    def compute_icub_grasp_from_panda_grasp(self, grasp_panda):
        # Rotate the grasp pose for the panda of 45 deg
        grasp_pose_0 = grasp_panda.pose * Transform(
            Rotation.from_rotvec(np.r_[0.0, 0.0, np.pi]), [0.0, 0.0, 0.0]
        )
        grasp_pose_0 = grasp_pose_0 * Transform(
            Rotation.from_rotvec(np.r_[0.0, -np.pi / 4, 0.0]), [0.0, 0.0, 0.0]
        )
        pos_grasp_icub_0 = np.array([-grasp_pose_0.translation[1] + self.medianx + 0.15,
                                     grasp_pose_0.translation[0] + self.mediany - 0.15,
                                     grasp_pose_0.translation[2] + 0.95])
        grasp_pose_1 = grasp_panda.pose * Transform(
            Rotation.from_rotvec(-np.pi / 4 * np.r_[0.0, 1.0, 0.0]), [0.0, 0.0, 0.0]
        )
        pos_grasp_icub_1 = np.array([-grasp_pose_1.translation[1] + self.medianx + 0.15,
                                     grasp_pose_1.translation[0] + self.mediany - 0.15,
                                     grasp_pose_1.translation[2] + 0.95])
        return pos_grasp_icub_0, grasp_pose_0.rotation.as_matrix(), pos_grasp_icub_1, grasp_pose_1.rotation.as_matrix()

    def set_grasp_pose_to_superquadric_format(self):
        if self.pos_grasp_icub is None or self.rot_grasp_icub is None:
            return {'position': [0.0, 0.0, 0.0], 'superq_center': [0.0, 0.0, 0.0]}
        sq_center = np.array([self.medianx, self.mediany, self.medianz])
        # Rototranslation world dh frame
        rt_dh = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
                         dtype=np.float32)
        rt_dh[:3, :3] = self.rot_grasp_icub
        rt_dh[:3, -1] = self.pos_grasp_icub
        best_grasp_pose = rt_dh
        best_grasp_pose_quat = Quaternion(matrix=best_grasp_pose[:3, :3], atol=1e-05)
        # Find distanced position on the sq_center-best_grasp_pos line
        mx, my, mz = self.line_coefficients_between_two_point(sq_center,
                                                              [best_grasp_pose[0, 3],
                                                               best_grasp_pose[1, 3],
                                                               best_grasp_pose[2, 3]], )
        distanced_position = np.array([best_grasp_pose[0, 3] +
                                       mx * self.distance_from_grasp_pose_disanced_position,
                                       best_grasp_pose[1, 3] +
                                       my * self.distance_from_grasp_pose_disanced_position,
                                       best_grasp_pose[2, 3] +
                                       mz * self.distance_from_grasp_pose_disanced_position])
        distanced_position_10_cm = np.array([best_grasp_pose[0, 3] + mx * 0.1,
                                             best_grasp_pose[1, 3] + my * 0.1,
                                             best_grasp_pose[2, 3] + mz * 0.1])
        best_grasp_pose_to_ret = {'position': [best_grasp_pose[0, 3],
                                               best_grasp_pose[1, 3],
                                               best_grasp_pose[2, 3]],
                                  'quaternion': best_grasp_pose_quat.q,
                                  'superq_center': sq_center,
                                  'distanced_grasp_position': distanced_position,
                                  'distanced_grasp_position_10_cm': distanced_position_10_cm}
        return best_grasp_pose_to_ret

    def compute_angular_error(self, vgn_rot_sol):
        # https://github.com/robotology/icub-gazebo-grasping-sandbox/blob/a26aef96fcc7bb4cc70742ac9fbb2c33568a6a2b/
        # src/cardinal_points_grasp.h#L62
        # Sec. 3.6 in https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
        iKin_sol_eef_pose = self.iKin_IK_solver.chain.EndEffPose()
        predicted_eef_axis_angle = np.array([iKin_sol_eef_pose.get(i) for i in range(3, 7)])
        predicted_eef_rot_matrix = Quaternion(axis=predicted_eef_axis_angle[:3],
                                              angle=predicted_eef_axis_angle[3]).rotation_matrix
        rot = Quaternion(matrix=np.matmul(vgn_rot_sol, np.transpose(predicted_eef_rot_matrix))).angle
        return abs(rot)

