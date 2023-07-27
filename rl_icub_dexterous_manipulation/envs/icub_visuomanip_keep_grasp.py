from rl_icub_dexterous_manipulation.envs.icub_visuomanip import ICubEnv
import numpy as np
from rl_icub_dexterous_manipulation.utils.pcd_utils import pcd_from_depth, points_in_world_coord
from rl_icub_dexterous_manipulation.utils.superquadrics_utils import SuperquadricEstimator
from pyquaternion import Quaternion
from dm_control.utils import inverse_kinematics as ik


class ICubEnvKeepGrasp(ICubEnv):

    def __init__(self, **kwargs):
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')
        self.object_lifted_state = None
        super().__init__(**kwargs)
        self.init_icub_act_after_superquadrics = self.init_icub_act.copy()
        self.superquadric_estimator = SuperquadricEstimator()
        self.prev_obj_zpos = None
        self.target_ctrl_lifted_object = None
        self.prev_qpos_ik = None
        self.superq_pose = None
        self.target_ik = None
        self.fingers_act_values = None
        self.env.physics.named.model.body_mass['004_sugar_box/'] = 0.5
        self.height_reached = False
        self.actuators_to_control_lift = ['r_shoulder_pitch',
                                          'r_shoulder_roll',
                                          'r_shoulder_yaw',
                                          'r_elbow',
                                          'r_wrist_prosup',
                                          'r_wrist_pitch',
                                          'r_wrist_yaw',
                                          'torso_pitch',
                                          'torso_roll',
                                          'torso_yaw']
        self.actuators_to_control_lift_ids = []
        for actuator_id, actuator in enumerate(self.world_entity.mjcf_model.find_all('actuator')):
            if actuator.name in self.actuators_to_control_lift:
                self.actuators_to_control_lift_ids.append(actuator_id)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        named_qpos = self.env.physics.named.data.qpos
        qpos_jnt_tendons = np.empty([0, ], dtype=np.float32)
        for actuator in self.actuators_to_control_dict:
            qpos_jnt_tendons = np.append(qpos_jnt_tendons,
                                         np.sum(named_qpos[actuator['jnt']] * actuator['coeff']))
        action += qpos_jnt_tendons
        action -= self.target_ctrl_lifted_object[self.actuators_to_control_ids]
        null_action = np.zeros(len(self.target_ctrl_lifted_object))
        np.put(null_action, self.actuators_to_control_ids, action)
        action = null_action
        target = np.clip(np.add(self.target_ctrl_lifted_object, action),
                         self.actuators_space.low + self.actuators_margin,
                         self.actuators_space.high - self.actuators_margin)
        self.prev_obj_zpos = self.env.physics.data.qpos[self.joint_ids_objects[2]]
        self.do_simulation(target, self.frame_skip)
        if self.done_if_joints_out_of_limits:
            done_limits = len(self.joints_out_of_range()) > 0
        else:
            done_limits = False
        done_goal = self.steps >= self._max_episode_steps
        observation = self._get_obs()
        done_fallen_object = self.fallen_object()
        reward = self._get_reward(done_limits, done_goal, done_fallen_object)
        done = done_limits or done_goal or done_fallen_object
        info = {'Steps': self.steps,
                'Done': {'goal_reached': done_goal,
                         'limits exceeded': self.joints_out_of_range(),
                         'fallen object': done_fallen_object}}
        if done and self.print_done_info:
            print(info)

        return observation, reward, done, info

    def _get_reward(self, done_limits, done_goal, done_fallen_object):
        if done_limits:
            return self.reward_out_of_joints
        reward = 0
        if done_goal:
            reward += self.reward_goal
        if done_fallen_object:
            reward -= 1
        return reward

    def reset_model(self):
        super().reset_model()
        if self.object_lifted_state is not None:
            self.set_state(self.object_lifted_state)
            self.env.physics.forward()
        elif hasattr(self, 'superquadric_estimator'):
            img = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera)
            depth = self.env.physics.render(height=480, width=640, camera_id=self.superquadrics_camera, depth=True)
            pcd = pcd_from_depth(depth)
            pcd[:, 2] = -pcd[:, 2]
            cam_id = self.env.physics.model.name2id(self.superquadrics_camera, 'camera')
            pcd = points_in_world_coord(pcd,
                                        cam_xpos=self.env.physics.named.data.cam_xpos[cam_id, :],
                                        cam_xmat=np.reshape(self.env.physics.named.data.cam_xmat[cam_id, :], (3, 3)))
            pcd[:, 2] -= 0.95
            segm = self.env.physics.render(height=480,
                                           width=640,
                                           camera_id=self.superquadrics_camera,
                                           segmentation=True)
            ids = np.where(np.reshape(segm[:, :, 0], (segm[:, :, 0].size,)) ==
                           self.env.physics.model.name2id(self.objects[0] + "/mesh_" + self.objects[0] + "_00_visual",
                                                          'geom'))
            pcd_colors = np.concatenate((pcd, np.reshape(img, (int(img.size / 3), 3))), axis=1)[ids]
            self.superq_pose = self.superquadric_estimator.compute_grasp_pose_superquadrics(pcd_colors)
            self.target_ik = np.concatenate((self.superq_pose['position'], self.superq_pose['quaternion']),
                                            dtype=np.float64)
            qpos_sol_final = ik.qpos_from_site_pose(physics=self.env.physics,
                                                    site_name='r_hand_dh_frame_site',
                                                    target_pos=self.superq_pose['position'],
                                                    target_quat=self.superq_pose['quaternion'],
                                                    joint_names=self.joints_to_control_ik)
            if qpos_sol_final.success:
                qpos_sol_final = qpos_sol_final.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
            current_state = self.get_state()
            current_state[self.joints_to_control_ik_ids] = qpos_sol_final
            self.set_state(current_state)
            self.update_init_qpos_act_from_current_state(current_state)
            self.env.physics.forward()

            self.close_right_hand()
            self.lift()
        return self._get_obs()

    def update_init_qpos_act_from_current_state(self, current_state):
        for actuator_id, actuator_dict in enumerate(self.actuators_dict):
            self.init_icub_act_after_superquadrics[actuator_id] = 0.0
            for i in range(len(actuator_dict['jnt'])):
                self.init_icub_act_after_superquadrics[actuator_id] += \
                    current_state[self.joint_ids_icub_dict[actuator_dict['jnt'][i]]] * actuator_dict['coeff'][i]

    def moved_object(self):
        for list_id, joint_id in enumerate(self.joint_ids_objects):
            # Check only the z component
            if list_id % 7 != 2:
                continue
            else:
                if self.env.physics.data.qpos[joint_id] < 0.98:
                    return True
        return False

    def fallen_object(self):
        # Check only the z component
        if self.env.physics.data.qpos[self.joint_ids_objects[2]] < 1.05:
            return True
        return False

    def close_right_hand(self):
        target_before_close = self.init_icub_act_after_superquadrics.copy()
        i = 0
        target_i = None
        while i < 2000:
            i += 1
            named_qpos = self.env.physics.named.data.qpos
            target_i = np.empty([0, ], dtype=np.float32)
            for actuator in self.actuators_to_control_dict:
                if actuator['name'] in self.actuators_to_control_no_fingers or not actuator['name'].startswith('r_'):
                    continue
                else:
                    qpos_act = np.sum(named_qpos[actuator['jnt']] * actuator['coeff'])
                    act_delta = actuator['close_value'] - actuator['open_value']
                    target_i = np.append(target_i, min(qpos_act + act_delta / 500 * 20,
                                                       actuator['open_value'] + act_delta * i / 500))
            target_before_close[self.actuators_to_control_fingers_ids] = target_i
            self.env.step(target_before_close)
            self.render()
        self.fingers_act_values = target_i

    def lift(self):
        lift_steps = 0
        target = None
        qpos_ik = None
        while lift_steps < 1000:
            lift_steps += 1
            cartesian_pose = np.concatenate((self.env.physics.named.data.xpos[self.eef_name],
                                             Quaternion(
                                                 matrix=np.reshape(self.env.physics.named.data.xmat[self.eef_name],
                                                                   (3, 3)), atol=1e-05).q))[2]
            if cartesian_pose > 1.17:
                self.height_reached = True
            # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
            # for the other joints
            self.target_ik[2] = min(cartesian_pose + np.array([0.05]), 1.17)
            target = self.init_icub_act_after_superquadrics.copy()
            if not self.height_reached:
                done_ik = False
                qpos_ik_result = ik.qpos_from_site_pose(physics=self.env.physics,
                                                        site_name='r_hand_dh_frame_site',
                                                        target_pos=self.target_ik[:3],
                                                        target_quat=self.target_ik[3:7],
                                                        joint_names=self.joints_to_control_ik)
                if qpos_ik_result.success:
                    qpos_ik = qpos_ik_result.qpos[np.array(self.joints_to_control_ik_ids, dtype=np.int64)]
                    self.prev_qpos_ik = qpos_ik.copy()
                else:
                    done_ik = True
                # Use as action only the offsets for the joints to control (e.g. hands)
                if not done_ik:
                    target[np.array(self.actuators_to_control_lift_ids)] = qpos_ik
            else:
                target[np.array(self.actuators_to_control_lift_ids)] = self.prev_qpos_ik
            target[self.actuators_to_control_fingers_ids] = self.fingers_act_values
            self.do_simulation(target, 5)
            self.render()
        self.target_ctrl_lifted_object = target.copy()
        self.object_lifted_state = self.get_state()
