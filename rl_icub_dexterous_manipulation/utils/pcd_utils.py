# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def pcd_from_depth(depth,
                   fx=617.783447265625,
                   fy=617.783447265625,
                   cx=320,
                   cy=240):
    pcd = np.empty(shape=(depth.size, 3))
    pcd[:, 2] = np.reshape(depth, (depth.size,))
    grid = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    u, v = np.reshape(grid[0], (grid[0].size, )), np.reshape(grid[1], (grid[1].size, ))
    pcd[:, 0] = (u - cy) * pcd[:, 2] / fy
    pcd[:, 1] = (v - cx) * pcd[:, 2] / fx
    return pcd


def points_in_world_coord(points, cam_xpos, cam_xmat):
    p_cam = np.stack([np.eye(4, dtype=np.float32)]*len(points), axis=-1)
    p_cam[:3, 3, :] = np.array([np.array(points[:, 1]), np.array(-points[:, 0]), np.array(points[:, 2])])
    cam_world = np.eye(4, dtype=np.float32)
    cam_world[:3, -1] = cam_xpos
    cam_world[:3, :3] = cam_xmat
    # Points roto-translation matrix in camera coordinates
    p_world = np.matmul(p_cam.transpose(), cam_world.transpose()).transpose()
    return p_world[:3, 3, :].transpose()
