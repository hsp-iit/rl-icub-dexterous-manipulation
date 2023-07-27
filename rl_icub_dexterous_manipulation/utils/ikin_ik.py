# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import yarp
import icub


class IKinIK:

    def __init__(self,
                 joints_to_control_names,
                 real_robot_limits=False,
                 limit_torso_pitch=False):

        self.arm_chain = icub.iCubArm('right_v2.5')
        icub.iCubAdditionalArmConstraints(self.arm_chain)

        self.joints_to_control_names = joints_to_control_names
        # Enable torso joints, if necessary
        if 'torso_pitch' in self.joints_to_control_names:
            self.arm_chain.releaseLink(0)
        if 'torso_roll' in self.joints_to_control_names:
            self.arm_chain.releaseLink(1)
        if 'torso_yaw' in self.joints_to_control_names:
            self.arm_chain.releaseLink(2)

        # Initialize arm and solver
        self.chain = icub.iKinChain(self.arm_chain)
        if real_robot_limits:
            self.set_real_robot_arm_chain_link_limits()
        if 'torso_pitch' in self.joints_to_control_names and limit_torso_pitch:
            self.chain.access(0).setMax(0.5)
        self.solver = icub.iKinIpOptMin(self.chain, 0, 1e-2, 1e-2, 5000, verbose=0)

        # Initialize variable to store solutions at the previous step
        self.prev_sol = None

    def set_real_robot_arm_chain_link_limits(self):
        # Torso limits from https://github.com/robotology/robots-configuration/blob/
        # master/iCubGenova01/hardware/motorControl/icub_torso.xml
        # Arm limits from https://github.com/robotology/robots-configuration/blob/
        # master/iCubGenova01/hardware/motorControl/icub_right_arm.xml
        deg2rad = np.pi / 180
        tot_torso_joints = 0
        if 'torso_pitch' in self.joints_to_control_names:
            self.chain.access(0).setMin(-22 * deg2rad)
            self.chain.access(0).setMax(70 * deg2rad)
            tot_torso_joints += 1
        if 'torso_roll' in self.joints_to_control_names:
            self.chain.access(tot_torso_joints).setMin(-30 * deg2rad)
            self.chain.access(tot_torso_joints).setMax(30 * deg2rad)
            tot_torso_joints += 1
        if 'torso_yaw' in self.joints_to_control_names:
            self.chain.access(tot_torso_joints).setMin(-50 * deg2rad)
            self.chain.access(tot_torso_joints).setMax(50 * deg2rad)
            tot_torso_joints += 1
        # r_shoulder_pitch
        self.chain.access(tot_torso_joints).setMin(-95.5 * deg2rad)
        self.chain.access(tot_torso_joints).setMax(10 * deg2rad)
        # r_shoulder_roll
        self.chain.access(tot_torso_joints + 1).setMin(0 * deg2rad)
        self.chain.access(tot_torso_joints + 1).setMax(161 * deg2rad)
        # r_shoulder_yaw
        self.chain.access(tot_torso_joints + 2).setMin(-30 * deg2rad)
        self.chain.access(tot_torso_joints + 2).setMax(75 * deg2rad)
        # r_elbow
        self.chain.access(tot_torso_joints + 3).setMin(15 * deg2rad)
        self.chain.access(tot_torso_joints + 3).setMax(106 * deg2rad)
        # r_wrist_prosup
        self.chain.access(tot_torso_joints + 4).setMin(-60 * deg2rad)
        self.chain.access(tot_torso_joints + 4).setMax(60 * deg2rad)
        # r_wrist_pitch
        self.chain.access(tot_torso_joints + 5).setMin(-80 * deg2rad)
        self.chain.access(tot_torso_joints + 5).setMax(25 * deg2rad)
        # r_wrist_yaw
        self.chain.access(tot_torso_joints + 6).setMin(-20 * deg2rad)
        self.chain.access(tot_torso_joints + 6).setMax(25 * deg2rad)

    def solve_ik(self,
                 eef_pos=None,
                 eef_axis_angle=None,
                 current_qpos=None,
                 joints_to_control_ik_ids=(),
                 on_step=True):

        # In the reset model phase, do not consider the previous solution
        if not on_step:
            self.prev_sol = None

        # Set current values in the arm chain for the torso joints which are not controlled
        if 'torso_pitch' not in self.joints_to_control_names:
            self.chain.setBlockingValue(0, current_qpos['torso_pitch'][0])
        if 'torso_roll' not in self.joints_to_control_names:
            self.chain.setBlockingValue(1, current_qpos['torso_roll'][0])
        if 'torso_yaw' not in self.joints_to_control_names:
            self.chain.setBlockingValue(2, current_qpos['torso_yaw'][0])

        current_qpos = yarp.Vector(current_qpos[joints_to_control_ik_ids])
        target_eef_pos = eef_pos.copy()
        target_eef_pos[2] -= 1
        target = yarp.Vector(np.concatenate((target_eef_pos, eef_axis_angle)))

        # Solve only IK task
        if not on_step:
            self.prev_sol = None
            # Keep the torso as close as possible to the vertical pose (3rd task)
            # Do not consider the 2nd task (i.e. constraint on a second link)
            solution_yarp = self.solver.solve(current_qpos,
                                              target,
                                              0.0,
                                              yarp.Vector([0.0]),
                                              yarp.Vector([0.0]),
                                              1.0,
                                              yarp.Vector(np.zeros(len(joints_to_control_ik_ids))),
                                              yarp.Vector(np.concatenate((np.ones(len(joints_to_control_ik_ids) - 7),
                                                                          np.zeros(7)))),
                                              )
            self.prev_sol = solution_yarp
        else:
            # Find a solution which is as close as possible to the previous solution (3rd task)
            # Do not consider the 2nd task (i.e. constraint on a second link)
            solution_yarp = self.solver.solve(self.prev_sol,
                                              target,
                                              0.0,
                                              yarp.Vector([0.0]),
                                              yarp.Vector([0.0]),
                                              1.0,
                                              self.prev_sol,
                                              yarp.Vector(np.ones(len(joints_to_control_ik_ids))),
                                              )
            self.prev_sol = solution_yarp

        # Compute and check solution
        solution = np.array([solution_yarp.get(i) for i in range(solution_yarp.getListSize())])
        predicted_pose_yarp = self.chain.EndEffPose()
        predicted_pos = np.array([predicted_pose_yarp.get(i) for i in range(3)])

        if np.linalg.norm(predicted_pos - target_eef_pos) < 0.03:
            solved = True
        else:
            solved = False

        return solution, solved
