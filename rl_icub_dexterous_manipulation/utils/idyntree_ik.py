# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import idyntree.bindings as idt
from pyquaternion import Quaternion
import numpy as np


class IDynTreeIK:

    def __init__(self,
                 joints_to_control,
                 joints_icub,
                 eef_frame,
                 urdf_model='../models/iCubGazeboV2_5_visuomanip_model.urdf',
                 reduced_model=True
                 ):

        self.reduced_model = reduced_model
        self.joints_to_control = joints_to_control
        if self.reduced_model:
            self.joints_icub = joints_to_control
        else:
            self.joints_icub = joints_icub
        self.eef_frame = eef_frame
        self.ik = idt.InverseKinematics()

        if self.reduced_model:
            # Load reduced model
            self.mdlLoader = idt.ModelLoader()
            self.mdlLoader.loadReducedModelFromFile(urdf_model, self.joints_to_control)
            self.ik.setModel(self.mdlLoader.model())
        else:
            # https://robotology.github.io/idyntree/classiDynTree_1_1InverseKinematics.html#ac3e893981cbfa2afc88a231d1cace9a4
            if not self.ik.loadModelFromFile(urdf_model,
                                             joints_to_control):
                raise RuntimeError("Urdf model not loaded.")

        self.ik.setFloatingBaseOnFrameNamed('root_link')
        base_transform = idt.Transform.Identity()
        self.ik.addFrameConstraint('root_link', base_transform)

        self.ik.setVerbosity(0)
        self.ik.setMaxIterations(3000)
        self.ik.setCostTolerance(1e-8)
        self.ik.setConstraintsTolerance(1e-5)

        self.ik.setDefaultTargetResolutionMode(idt.InverseKinematicsTreatTargetAsConstraintNone)
        self.ik.setRotationParametrization(idt.InverseKinematicsRotationParametrizationRollPitchYaw)

        # TODO swap weights after idyntree release 6.0.1
        self.ik.addTarget(self.eef_frame,
                          idt.Transform.Identity(),
                          1.0,
                          100.0)

    def solve_ik(self,
                 eef_pos=None,
                 eef_quat=None,
                 current_qpos=None,
                 desired_configuration=None):

        joints_order = []
        joints_np_array = np.zeros(self.ik.fullModel().getNrOfJoints(), dtype=np.float32)
        joints_desired_np_array = np.zeros(self.ik.fullModel().getNrOfJoints(), dtype=np.float32)

        for jid, joint in enumerate(self.joints_icub):
            joints_order.append(self.ik.fullModel().getJointIndex(joint))
            joints_np_array[self.ik.fullModel().getJointIndex(joint)] = float(current_qpos[joint])
            if desired_configuration is not None:
                joints_desired_np_array[self.ik.fullModel().getJointIndex(joint)] = desired_configuration[jid]
            else:
                joints_desired_np_array[self.ik.fullModel().getJointIndex(joint)] = current_qpos[joint]

        base_transform = idt.Transform.Identity()
        self.ik.setFullJointsInitialCondition(base_transform,
                                              idt.VectorDynSize_FromPython(joints_np_array))

        self.ik.setDesiredFullJointsConfiguration(idt.VectorDynSize_FromPython(joints_desired_np_array),
                                                  1e-6)

        eef_target = idt.Transform.Identity()
        eef_target.setPosition(idt.Position(eef_pos[0],
                                            eef_pos[1],
                                            eef_pos[2] - 1))
        eef_rot_matrix = Quaternion(eef_quat).rotation_matrix
        eef_target.setRotation(idt.Rotation(eef_rot_matrix[0][0],
                                            eef_rot_matrix[0][1],
                                            eef_rot_matrix[0][2],
                                            eef_rot_matrix[1][0],
                                            eef_rot_matrix[1][1],
                                            eef_rot_matrix[1][2],
                                            eef_rot_matrix[2][0],
                                            eef_rot_matrix[2][1],
                                            eef_rot_matrix[2][2]))

        # TODO swap weights after idyntree release 6.0.1
        if not self.ik.updateTarget(self.eef_frame,
                                    eef_target,
                                    1.0,
                                    100.0):
            print('Target not updated')
            return None, False

        if not self.ik.solve():
            print('IK not solved')
            return None, False

        base_transform = idt.Transform.Identity()
        joint_positions = idt.VectorDynSize(self.ik.fullModel().getNrOfJoints())
        self.ik.getFullJointsSolution(base_transform,
                                      joint_positions)

        joint_positions_ordered = joint_positions.toNumPy()[joints_order]

        return joint_positions_ordered, True
