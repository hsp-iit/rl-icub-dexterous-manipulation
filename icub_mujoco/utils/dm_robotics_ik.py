import numpy as np
from dm_robotics.moma.utils.ik_solver import IkSolver
from dm_robotics.geometry import geometry


class DMRoboticsIK:

    def __init__(self,
                 mjcf_model,
                 joints_to_control,
                 ):

        self.joints_to_control = joints_to_control
        self.controllable_joints = [j for j in mjcf_model.find_all('joint') if j.name in self.joints_to_control]
        self.site_id = [site.name == 'r_hand_dh_frame_site' for site in mjcf_model.find_all('site')].index(True)
        self.solver = IkSolver(mjcf_model,
                               controllable_joints=self.controllable_joints,
                               element=mjcf_model.find_all('site')[self.site_id]
                               )

    def solve_ik(self,
                 eef_pos=None,
                 eef_quat=None,
                 current_qpos=None,
                 ):

        ref_pos = geometry.Pose(position=eef_pos, quaternion=eef_quat)
        initial_qpos = None
        if current_qpos is not None:
            initial_qpos = np.zeros(len(self.controllable_joints), dtype=np.float32)
            for jid, joint in enumerate(self.joints_to_control):
                initial_qpos[jid] = current_qpos[joint]

        # Not working with initial_qpos different from None due to a bug in "dm_robotics/moma/utils/ik_solver.py"
        qpos_sol = self.solver.solve(ref_pos,
                                     linear_tol=1e-5,
                                     angular_tol=1e-5,
                                     early_stop=True,
                                     stop_on_first_successful_attempt=True,
                                     inital_joint_configuration=initial_qpos)
        if qpos_sol is None:
            print('DM_Robotics IK not solved')
            return None, False
        else:
            return qpos_sol, True
