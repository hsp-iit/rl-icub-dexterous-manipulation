# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import sys
import numpy as np
import math
from pyquaternion import Quaternion


class SuperquadricEstimator:

    def __init__(self, distance_from_grasp_pose_distanced_position):
        sys.path.append('/usr/local/lib/superquadriclib/bindings')
        import superquadric_bindings as sb
        self.sb = sb
        self.sq_estimator = self.sb.SuperqEstimatorApp()
        self.grasp_estimator = self.sb.GraspEstimatorApp()
        self.vector_superquadric = self.sb.vector_superquadric
        self.distance_from_grasp_pose_distanced_position = distance_from_grasp_pose_distanced_position

    def compute_grasp_pose_superquadrics(self,
                                         pcd,
                                         object_class="default",
                                         plane_height=0.05,
                                         height_icub_root=1.0,
                                         custom_displacements=None):
        pointcloud = self.sb.PointCloud()
        points = self.sb.deque_Vector3d()
        colors = self.sb.vector_vector_uchar()
        for p in pcd:
            points.push_back(np.array([p[0], p[1], p[2]], dtype='float'))
            colors.push_back([int(p[3]), int(p[4]), int(p[5])])
        pointcloud.setPoints(points)
        pointcloud.setColors(colors)
        self.sq_estimator.SetIntegerValue("optimizer_points", 100)
        self.sq_estimator.SetBoolValue("random_sampling", False)
        # https://github.com/robotology/superquadric-lib/blob/f38e76324a863c9c21b059b586a9e88618db11c9/src
        # /SuperquadricLib/SuperquadricModel/src/superquadricEstimator.cpp#L293
        self.sq_estimator.SetStringValue("object_class", object_class)
        # Compute superquadric
        sq_vec = self.vector_superquadric(self.sq_estimator.computeSuperq(pointcloud))
        sq_center = sq_vec.front().center[0]
        sq_center[2] += height_icub_root
        # Compute grasp pose
        if custom_displacements is not None:
            displacements = custom_displacements
        else:
            displacements = [np.array([0.02, 0.0, 0.0]), np.array([0.0, 0.02, 0.0]), np.array([0.0, 0.0, 0.02])]
        best_grasp_pose_to_ret = None
        for displacement in displacements:
            self.grasp_estimator.setVector('plane', np.array([0, 0, 1.0, plane_height]))
            self.grasp_estimator.setVector('displacement', displacement)
            grasp_res_hand = self.grasp_estimator.computeGraspPoses(sq_vec)
            best_grasp_position = grasp_res_hand.grasp_poses.front().position[0]
            best_grasp_position[2] += height_icub_root
            update_pose = False
            if best_grasp_pose_to_ret is None:
                update_pose = True
            if best_grasp_pose_to_ret is not None:
                if best_grasp_pose_to_ret['position'][2] < best_grasp_position[2]:
                    update_pose = True
            if update_pose:
                rotation_axis_angle = grasp_res_hand.grasp_poses.front().axisangle[0]
                # Rototranslation world dh frame
                rt_dh = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]],
                                 dtype=np.float32)
                rt_dh[:3, :3] = Quaternion(self.axisangle_to_quat(rotation_axis_angle)).rotation_matrix
                tr = best_grasp_position
                rt_dh[:3, -1] = tr
                best_grasp_pose = rt_dh
                best_grasp_pose_quat = Quaternion(matrix=best_grasp_pose[:3, :3], atol=1e-05)
                # Find distanced position on the sq_center-best_grasp_pos line
                mx, my, mz = self.line_coefficients_between_two_point(sq_center,
                                                                      [best_grasp_pose[0, 3],
                                                                       best_grasp_pose[1, 3],
                                                                       best_grasp_pose[2, 3]], )
                distanced_position = np.array([best_grasp_pose[0, 3] +
                                               mx * self.distance_from_grasp_pose_distanced_position,
                                               best_grasp_pose[1, 3] +
                                               my * self.distance_from_grasp_pose_distanced_position,
                                               best_grasp_pose[2, 3] +
                                               mz * self.distance_from_grasp_pose_distanced_position])
                distanced_position_10_cm = np.array([best_grasp_pose[0, 3] + mx * 0.1,
                                                     best_grasp_pose[1, 3] + my * 0.1,
                                                     best_grasp_pose[2, 3] + mz * 0.1])
                # Define grasp type as lateral (0) or top-down (1)
                x_z_unit_vector_world_coord = np.matmul(rt_dh,
                                                        np.array([[1, 0, 0, 1/np.sqrt(2)],
                                                                  [0, 1, 0, 0],
                                                                  [0, 0, 1, 1/np.sqrt(2)],
                                                                  [0, 0, 0, 1]],
                                                                 dtype=np.float32))
                x_z_unit_vector_world_coord = x_z_unit_vector_world_coord[:3, 3] - tr
                grasp_type = np.argmax(np.array([np.linalg.norm([x_z_unit_vector_world_coord[0],
                                                                 x_z_unit_vector_world_coord[1]]),
                                                 abs(x_z_unit_vector_world_coord[2])]))
                best_grasp_pose_to_ret = {'position': [best_grasp_pose[0, 3],
                                                       best_grasp_pose[1, 3],
                                                       best_grasp_pose[2, 3]],
                                          'quaternion': best_grasp_pose_quat.q,
                                          'superq_center': sq_center,
                                          'distanced_grasp_position': distanced_position,
                                          'distanced_grasp_position_10_cm': distanced_position_10_cm,
                                          'grasp_type': grasp_type}
        return best_grasp_pose_to_ret

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
