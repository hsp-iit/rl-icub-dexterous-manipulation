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
    xyzs = []
    for point in points:
        # Point roto-translation matrix in world coordinates
        p_cam = np.array([[1, 0, 0, point[1]],
                          [0, 1, 0, -point[0]],
                          [0, 0, 1, point[2]],
                          [0, 0, 0, 1]],
                         dtype=np.float32)
        cam_world = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]],
                             dtype=np.float32)
        cam_world[:3, -1] = cam_xpos
        cam_world[:3, :3] = cam_xmat
        # Point roto-translation matrix in camera coordinates
        p_world = np.matmul(cam_world, p_cam)
        xyzs.append(np.array([p_world[0, 3], p_world[1, 3], p_world[2, 3]]))
    return np.array(xyzs)
