import numpy as np
from scipy import optimize
import math


class GazeController:

    def gaze_control(self,
                     fixation_point,
                     current_neck_pitch,
                     current_neck_roll,
                     current_neck_yaw,
                     chest_xpos,
                     chest_xmat):

        constraints = ({'type': 'ineq',
                        'fun': self.fun_constraint_gaze_control,
                        'args': (self, chest_xpos, chest_xmat, fixation_point)},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_lower_pitch},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_lower_roll},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_lower_yaw},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_upper_pitch},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_upper_roll},
                       {'type': 'ineq',
                        'fun': self.fun_constraint_upper_yaw}
                       )
        neck_joints = optimize.minimize(self.fun_gaze_control,
                                        x0=np.array([current_neck_pitch,
                                                     current_neck_roll,
                                                     current_neck_yaw]),
                                        args=(-0.5, 0, 0),
                                        constraints=constraints,
                                        method='COBYLA')
        return neck_joints

    @staticmethod
    def points_in_camera_coord_mod_neck_joints(points, chest_xpos, chest_xmat, neck_pitch, neck_roll, neck_yaw):
        com_xyzs = []
        for point in points:
            # Point roto-translation matrix in world coordinates
            p_world = np.array([[1, 0, 0, point[0]],
                                [0, 1, 0, point[1]],
                                [0, 0, 1, point[2]],
                                [0, 0, 0, 1]],
                               dtype=np.float32)
            # Camera roto-translation matrix in world coordinates
            chest_world = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]],
                                   dtype=np.float32)
            chest_world[:3, -1] = chest_xpos
            chest_world[:3, :3] = np.reshape(chest_xmat, (3, 3))

            # Transformation matrix chest-neck_1
            chest_neck_1 = np.array([[1, 0, 0, -0.0143],
                                     [0, math.cos(neck_pitch), -math.sin(-neck_pitch), 0.172797],
                                     [0, math.sin(-neck_pitch), math.cos(neck_pitch), 0.0053118],
                                     [0, 0, 0, 1]],
                                    dtype=np.float32)

            # neck_1 in world coordinates
            neck_1 = np.matmul(chest_world, chest_neck_1)

            # Transformation matrix neck_1-neck_2
            neck_1_neck_2 = np.array([[math.cos(neck_roll), -math.sin(neck_roll), 0, 0.0143],
                                      [math.sin(neck_roll), math.cos(neck_roll), 0, 0.0095],
                                      [0, 0, 1, 0.0208],
                                      [0, 0, 0, 1]],
                                     dtype=np.float32)

            # neck_2 in world coordinates
            neck_2 = np.matmul(neck_1, neck_1_neck_2)

            # Transformation matrix neck_2-head
            neck_2_head = np.array([[math.cos(neck_yaw), 0, math.sin(neck_yaw), 0],
                                    [0, 1, 0, -0.022344],
                                    [-math.sin(neck_yaw), 0, math.cos(neck_yaw), -0.0208],
                                    [0, 0, 0, 1]],
                                   dtype=np.float32)

            # head in world coordinates
            head = np.matmul(neck_2, neck_2_head)

            # Transformation matrix head-head_cam
            head_cam = np.array([[-0.9987914, 0.02398326, -0.04290156, 0.043454],
                                 [0.00680677, 0.93195106, 0.3625202, 0.217822],
                                 [0.04867657,  0.36179004, -0.93098795, 0.0835957],
                                 [0, 0, 0, 1]],
                                dtype=np.float32)

            # head_cam in world coordinates
            cam_world = np.matmul(head, head_cam)

            # Point roto-translation matrix in camera coordinates
            p_cam = np.matmul(np.linalg.inv(cam_world), p_world)
            com_xyzs.append(np.array([p_cam[0, 3], p_cam[1, 3], p_cam[2, 3]]))
        return com_xyzs

    @staticmethod
    def fun_gaze_control(x, qn_0, qn_1, qn_2):
        vec_to_optimize = np.array([x[0] - qn_0, x[1] - qn_1, x[2] - qn_2])
        return np.linalg.norm(vec_to_optimize) ** 2

    @staticmethod
    def fun_constraint_gaze_control(x, self, chest_xpos, chest_xmat, fixation_point, epsilon=1e-2):
        fixation_point_camera_coord = self.points_in_camera_coord_mod_neck_joints([fixation_point],
                                                                                  chest_xpos,
                                                                                  chest_xmat,
                                                                                  x[0],
                                                                                  x[1],
                                                                                  x[2])[0]
        cos_theta = np.dot(np.array([0, 0, -1]),
                           - fixation_point_camera_coord) / np.linalg.norm(fixation_point_camera_coord)
        return -1 - cos_theta + epsilon

    @staticmethod
    def fun_constraint_lower_pitch(x):
        return x[0] + 0.698132

    @staticmethod
    def fun_constraint_lower_roll(x):
        return x[1] + 0.349066

    @staticmethod
    def fun_constraint_lower_yaw(x):
        return x[2] + 0.872665

    @staticmethod
    def fun_constraint_upper_pitch(x):
        return - x[0] + 0.383972

    @staticmethod
    def fun_constraint_upper_roll(x):
        return - x[1] + 0.349066

    @staticmethod
    def fun_constraint_upper_yaw(x):
        return - x[2] + 0.872665
