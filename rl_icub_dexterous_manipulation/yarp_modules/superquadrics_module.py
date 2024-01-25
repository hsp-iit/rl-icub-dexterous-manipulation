# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import yarp
import sys
from rl_icub_dexterous_manipulation.utils.pcd_utils import *
from pyquaternion import Quaternion
from rl_icub_dexterous_manipulation.utils.superquadrics_utils import SuperquadricEstimator
import yaml

# Initialize YARP
yarp.Network.init()


class SuperquadricsModule(yarp.RFModule):
    def configure(self, rf):

        self.module_name = rf.find("module_name").asString()
        self.state = 'do_nothing'

        self.image_w = 640
        self.image_h = 480

        self.fx = rf.find("fx").asFloat64()
        self.fy = rf.find("fy").asFloat64()
        self.cx = rf.find("cx").asFloat64()
        self.cy = rf.find("cy").asFloat64()

        self.out_ports = {}
        self.in_ports = {}

        self.cmd_port = yarp.Port()
        self.cmd_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.cmd_port)

        self._input_image_port = yarp.BufferedPortImageRgb()
        self._input_image_port.open('/' + self.module_name + '/image:i')
        print('{:s} opened'.format('/' + self.module_name + '/image:i'))

        print('Preparing input image...')
        self._in_buf_array = np.ones((self.image_h, self.image_w, 3), dtype=np.uint8)
        self._in_buf_image = yarp.ImageRgb()
        self._in_buf_image.resize(self.image_w, self.image_h)
        self._in_buf_image.setExternal(self._in_buf_array.data, self._in_buf_array.shape[1],
                                       self._in_buf_array.shape[0])

        self._input_depth_port = yarp.BufferedPortImageFloat()
        self._input_depth_port.open('/' + self.module_name + '/depth:i')
        print('{:s} opened'.format('/' + self.module_name + '/depth:i'))

        print('Preparing input depth...')
        self._in_buf_depth_array = np.zeros((self.image_h, self.image_w), dtype=np.float32)
        self._in_buf_depth = yarp.ImageFloat()
        self._in_buf_depth.resize(self.image_w, self.image_h)
        self._in_buf_depth.setExternal(self._in_buf_depth_array.data,
                                       self._in_buf_depth_array.shape[1],
                                       self._in_buf_depth_array.shape[0])

        self._rs_pose_port = yarp.BufferedPortBottle()
        self._rs_pose_port.open('/' + self.module_name + '/realsense_pose')
        print('{:s} opened'.format('/' + self.module_name + '/realsense_pose'))

        self._r_arm_qpos_port = yarp.BufferedPortBottle()
        self._r_arm_qpos_port.open('/' + self.module_name + '/r_arm_qpos')
        print('{:s} opened'.format('/' + self.module_name + '/r_arm_qpos'))

        self._r_arm_xpos_port = yarp.BufferedPortVector()
        self._r_arm_xpos_port.open('/' + self.module_name + '/r_arm_xpos')
        print('{:s} opened'.format('/' + self.module_name + '/r_arm_xpos'))

        self._r_hand_touch_port = yarp.BufferedPortVector()
        self._r_hand_touch_port.open('/' + self.module_name + '/r_hand_touch')
        print('{:s} opened'.format('/' + self.module_name + '/r_hand_touch'))

        self.robot_name = 'icub'

        # https://github.com/robotology/yarp/blob/ea6ed180ddb102679d8b8492fc40329719d7b8e4/bindings/python/examples/example_enc.py
        # Arm controllers
        self.props = yarp.Property()
        self.driver = yarp.PolyDriver()

        self.props.put("device", "remote_controlboard")
        self.props.put("local", "/enc_r_arm/client")
        self.props.put("remote", "/" + self.robot_name + "/right_arm")

        # Opening the drivers
        print('Opening the motor driver...')
        self.driver.open(self.props)
        if not self.driver.isValid():
            print('Cannot open the driver!')
            sys.exit()

        # Opening the drivers
        print('Viewing motor position/encoders...')
        self.ipos = self.driver.viewIPositionControl()
        for j in range(9):
            self.ipos.setRefSpeed(j + 7, 10)
        self.ienc = self.driver.viewIEncoders()
        self.imode = self.driver.viewIControlMode()
        if self.ienc is None or self.ipos is None:
            print('Cannot view motor positions/encoders!')
            sys.exit()

        # Cartesian controller
        self.props_cart = yarp.Property()
        self.props_cart.put('device', 'cartesiancontrollerclient')
        self.props_cart.put('local', '/example/right_arm')
        self.props_cart.put('remote', '/' + self.robot_name + '/cartesianController/right_arm')
        self.cart_driver = yarp.PolyDriver(self.props_cart)
        self.cart = self.cart_driver.viewICartesianControl()
        self.cart.setTrajTime(2.0)
        self.cur_dof = yarp.Vector()
        self.cart.getDOF(self.cur_dof)

        # Set torso active joints for cartesian controller
        self.new_dof = yarp.Vector(3)
        self.new_dof[0] = 1
        # Torso roll
        self.new_dof[1] = 0
        self.new_dof[2] = 1
        self.cart.setDOF(self.new_dof, self.cur_dof)

        # Left Arm
        # https://github.com/robotology/yarp/blob/ea6ed180ddb102679d8b8492fc40329719d7b8e4/bindings/python/examples/example_enc.py
        # Arm controllers
        self.props_left = yarp.Property()
        self.driver_left = yarp.PolyDriver()

        self.props_left.put("device", "remote_controlboard")
        self.props_left.put("local", "/enc_l_arm/client")
        self.props_left.put("remote", "/" + self.robot_name + "/left_arm")

        # Opening the drivers
        print('Opening the motor driver...')
        self.driver_left.open(self.props_left)
        if not self.driver_left.isValid():
            print('Cannot open the driver!')
            sys.exit()

        # Opening the drivers
        print('Viewing motor position/encoders...')
        self.ipos_left = self.driver_left.viewIPositionControl()
        self.ienc_left = self.driver_left.viewIEncoders()
        self.imode_left = self.driver_left.viewIControlMode()
        if self.ienc_left is None or self.ipos_left is None:
            print('Cannot view motor positions/encoders!')
            sys.exit()

        # Torso controllers
        self.props_torso = yarp.Property()
        self.driver_torso = yarp.PolyDriver()

        self.props_torso.put("device", "remote_controlboard")
        self.props_torso.put("local", "/enc_torso/client")
        self.props_torso.put("remote", "/" + self.robot_name + "/torso")

        # Opening the drivers
        print('Opening the motor driver...')
        self.driver_torso.open(self.props_torso)
        if not self.driver_torso.isValid():
            print('Cannot open the driver!')
            sys.exit()

        # Opening the drivers
        print('Viewing motor position/encoders...')
        self.ipos_torso = self.driver_torso.viewIPositionControl()
        self.ienc_torso = self.driver_torso.viewIEncoders()
        self.imode_torso = self.driver_torso.viewIControlMode()
        if self.ienc_torso is None or self.ipos_torso is None:
            print('Cannot view motor positions/encoders!')
            sys.exit()

        # Head controllers
        self.props_head = yarp.Property()
        self.driver_head = yarp.PolyDriver()

        self.props_head.put("device", "remote_controlboard")
        self.props_head.put("local", "/enc_head/client")
        self.props_head.put("remote", "/" + self.robot_name + "/head")

        # Opening the drivers
        print('Opening the motor driver...')
        self.driver_head.open(self.props_head)
        if not self.driver_head.isValid():
            print('Cannot open the driver!')
            sys.exit()

        # Opening the drivers
        print('Viewing motor position/encoders...')
        self.ipos_head = self.driver_head.viewIPositionControl()
        self.ienc_head = self.driver_head.viewIEncoders()
        self.imode_head = self.driver_head.viewIControlMode()
        if self.ienc_head is None or self.ipos_head is None:
            print('Cannot view motor positions/encoders!')
            sys.exit()

        # Gaze controller
        self.props_gaze = yarp.Property()
        self.driver_gaze = yarp.PolyDriver()

        self.props_gaze.put('device', 'gazecontrollerclient')
        self.props_gaze.put('local', '/gaze_controller/client')
        self.props_gaze.put('remote', '/iKinGazeCtrl')

        # Opening the drivers
        print('Opening the motor driver...')
        self.driver_gaze.open(self.props_gaze)
        if not self.driver_gaze.isValid():
            print('Cannot open the driver!')
            sys.exit()

        self.gaze_ctrl = self.driver_gaze.viewIGazeControl()

        if self.gaze_ctrl is None:
            print('Cannot view motor positions/encoders!')
            sys.exit()
        self.gaze_ctrl.setEyesTrajTime(1.0)
        self.gaze_ctrl.setNeckTrajTime(1.0)
        self.gaze_ctrl.blockEyes(0)

        # Load initial qpos from yaml file and map joint ids to actuator ids
        with open('../config/icub_visuomanip_initial_qpos_actuated_hand.yaml') as initial_qpos_file:
            self.init_icub_act_dict = yaml.load(initial_qpos_file, Loader=yaml.FullLoader)

        self.init_icub_act_dict["r_elbow"] = 0.7
        self.init_arm_enc_values = yarp.Vector([self.init_icub_act_dict[act] * 180 / np.pi for act in
                                                ["r_shoulder_pitch",
                                                 "r_shoulder_roll",
                                                 "r_shoulder_yaw",
                                                 "r_elbow",
                                                 "r_wrist_prosup",
                                                 "r_wrist_pitch",
                                                 "r_wrist_yaw",
                                                 "r_hand_finger",
                                                 "r_thumb_oppose",
                                                 "r_thumb_proximal",
                                                 "r_thumb_distal",
                                                 "r_index_proximal",
                                                 "r_index_distal",
                                                 "r_middle_proximal",
                                                 "r_middle_distal",
                                                 "r_pinky"]])
        self.r_hand_open_values = np.array([0, 75, 0, 0, 0, 0, 0, 0, 0])
        self.r_hand_close_values = np.array([60, 75, 90, 170, 90, 170, 90, 170, 250])

        print('Preparing output image...')
        self._output_image_port = yarp.Port()
        self._output_image_port.open('/' + self.module_name + '/image:o')
        print('{:s} opened'.format('/' + self.module_name + '/image:o'))
        self.out_ports['output_image_port'] = self._output_image_port

        self.pre_grasp_distance_from_grasp_pose = rf.find("pre_grasp_distance_from_grasp_pose").asFloat64()
        self.superquadric_estimator = \
            SuperquadricEstimator(distance_from_grasp_pose_distanced_position=self.pre_grasp_distance_from_grasp_pose)

        self.height_simulated_icub_root = rf.find("height_simulated_icub_root").asFloat64()
        self.policy_steps = 0

        self.done_moved = False
        self.save_current_model = False
        self.wait_reset = False

        self.joints_to_control = 'r_hand'

        if self.joints_to_control != 'r_hand':
            raise ValueError('Only joints_control for the right hand fingers has been implemented.')

        self.offsets_distanced_grasp_pose_x = rf.find("offsets_distanced_grasp_pose_x").asFloat64()
        self.offsets_distanced_grasp_pose_y = rf.find("offsets_distanced_grasp_pose_y").asFloat64()
        self.offsets_distanced_grasp_pose_z = rf.find("offsets_distanced_grasp_pose_z").asFloat64()
        self.offsets_distanced_grasp_pose_10cm_x = rf.find("offsets_distanced_grasp_pose_10cm_x").asFloat64()
        self.offsets_distanced_grasp_pose_10cm_y = rf.find("offsets_distanced_grasp_pose_10cm_y").asFloat64()
        self.offsets_distanced_grasp_pose_10cm_z = rf.find("offsets_distanced_grasp_pose_10cm_z").asFloat64()

        self.pcd_min_x = rf.find("pcd_min_x").asFloat64()
        self.pcd_max_x = rf.find("pcd_max_x").asFloat64()
        self.pcd_min_y = rf.find("pcd_min_y").asFloat64()
        self.pcd_max_y = rf.find("pcd_max_y").asFloat64()
        self.pcd_min_z = rf.find("pcd_min_z").asFloat64()
        self.pcd_max_z = rf.find("pcd_max_z").asFloat64()

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            print('Quitting module')
            self.state = 'quit'
        elif command.get(0).asString() == 'go_to_initial_configuration':
            self.state = 'go_to_initial_configuration'
            reply.addString('Going to initial configuration.')
        elif command.get(0).asString() == 'fingers_in_idle':
            self.state = 'fingers_in_idle'
            reply.addString('Setting fingers in idle control mode.')
        elif command.get(0).asString() == 'compute_superq_grasp_pose':
            self.state = 'compute_superq_grasp_pose'
            reply.addString('Computing superquadrics grasp pose.')
        elif command.get(0).asString() == 'fingers_in_position_control':
            self.state = 'fingers_in_position_control'
            reply.addString('Setting fingers in idle control mode.')
        elif command.get(0).asString() == 'open_right_hand':
            self.state = 'open_right_hand'
            reply.addString('Opening right hand.')
        elif command.get(0).asString() == 'go_to_distanced_superq_pose':
            self.state = 'go_to_distanced_superq_pose'
            reply.addString('Going to the distanced superquadrics cartesian pose.')
        elif command.get(0).asString() == 'go_to_distanced_superq_pose_10cm':
            self.state = 'go_to_distanced_superq_pose_10cm'
            reply.addString('Going to the distanced superquadrics cartesian pose of 10cm.')
        elif command.get(0).asString() == 'stop_motion':
            self.state = 'stop_motion'
            reply.addString('Stopping motion.')
        elif command.get(0).asString() == 'look_at_fixation_point':
            if command.size() < 4:
                self.state = 'do_nothing'
                reply.addString('Need to pass also fixation point 3D position.')
            else:
                self.fixation_position = [command.get(1).asFloat64(),
                                          command.get(2).asFloat64(),
                                          command.get(3).asFloat64()]
                self.state = 'look_at_fixation_point'
                reply.addString('Looking at the desired fixation point.')
        elif command.get(0).asString() == 'done':
            self.done = 'done'
            reply.addString('Done policy.')
        elif command.get(0).asString() == 'done_moved':
            self.done_moved = True
            reply.addString('Done policy moved object.')
        elif command.get(0).asString() == 'save_model':
            self.save_current_model = True
            reply.addString('Saving model at next reset')
        elif command.get(0).asString() == 'wait':
            self.wait_reset = True
            reply.addString('Waiting to reset model')
        elif command.get(0).asString() == 'stop_wait':
            self.wait_reset = False
            reply.addString('Reset model')
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))
        return True

    def cleanup(self):
        print('Cleanup function')
        for part in self.out_ports:
            self.out_ports[part].close()
        for part in self.in_ports:
            self.in_ports[part].close()

        self.cmd_port.close()

    def interruptModule(self):
        print('Interrupt function')
        for part in self.out_ports:
            self.out_ports[part].interrupt()
        for part in self.in_ports:
            self.in_ports[part].interrupt()
        self.cmd_port.interrupt()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):
        if self.state == 'quit':
            self.cleanup()
            self.interruptModule()
            quit()
        elif self.state == 'compute_superq_grasp_pose':
            received_image = self._input_image_port.read(True)

            self._in_buf_image.copy(received_image)
            assert self._in_buf_array.__array_interface__['data'][0] == self._in_buf_image.getRawImage().__int__()

            frame = self._in_buf_array
            input_image = frame.copy()
            received_depth = self._input_depth_port.read(True)
            self._in_buf_depth.copy(received_depth)
            assert self._in_buf_depth_array.__array_interface__['data'][0] == self._in_buf_depth.getRawImage().__int__()
            depth = self._in_buf_depth_array
            input_depth = depth.copy()
            rs_pose_bottle = self._rs_pose_port.read(True)
            rs_pose = np.zeros(rs_pose_bottle.size(), dtype=np.float32)
            for i in range(rs_pose_bottle.size()):
                rs_pose [i] = rs_pose_bottle.get(i).asFloat64()

            pcd = pcd_from_depth(depth,
                                 fx=self.fx,
                                 fy=self.fy,
                                 cx=self.cx,
                                 cy=self.cy)
            pcd[:, 0] = -pcd[:, 0]
            rs_xmat = Quaternion(axis=rs_pose[3:6], angle=rs_pose[6]).rotation_matrix
            pcd_world_coord = points_in_world_coord(pcd, rs_pose[:3], rs_xmat)
            pcd_colors = np.concatenate((pcd_world_coord, np.reshape(frame, (int(frame.size / 3), 3))), axis=1)
            pcd_colors_filt =np.empty((0, 6), dtype=pcd_colors.dtype)
            for point in pcd_colors:
                if (self.pcd_min_x < point[0] < self.pcd_max_x
                        and self.pcd_min_y < point[1] < self.pcd_max_y and self.pcd_min_z < point[2] < self.pcd_max_z):
                    pcd_colors_filt = np.concatenate((pcd_colors_filt, np.array([point])))

            self.superq_pose = self.superquadric_estimator.compute_grasp_pose_superquadrics(pcd_colors_filt,
                                                                                            plane_height=0.00,
                                                                                            height_icub_root=0.0,
                                                                                            custom_displacements=[np.array([0.02, 0.0, 0.0])])

            self.superq_pos = np.array(self.superq_pose['position'], dtype=np.float64)

            self.superq_distanced_pos = np.array(self.superq_pos)
            self.superq_distanced_pos[0] += self.offsets_distanced_grasp_pose_x
            self.superq_distanced_pos[1] += self.offsets_distanced_grasp_pose_y
            self.superq_distanced_pos[2] += self.offsets_distanced_grasp_pose_z
            self.superq_distanced_pos_10_cm = np.array(self.superq_pos)
            self.superq_distanced_pos_10_cm[0] += self.offsets_distanced_grasp_pose_10cm_x
            self.superq_distanced_pos_10_cm[1] += self.offsets_distanced_grasp_pose_10cm_y
            self.superq_distanced_pos_10_cm[2] += self.offsets_distanced_grasp_pose_10cm_z

            self.superq_or = np.array(self.superq_pose['quaternion'])
            self.superq_center = np.array(self.superq_pose['superq_center'])
            self.state = 'do_nothing'

        elif self.state == 'go_to_initial_configuration':
            # Reach right arm
            self.cart.stopControl()
            self.ipos.stop()
            for i in range(16):
                self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION)
            self.ipos.positionMove(self.init_arm_enc_values.data())
            while not self.ipos.checkMotionDone():
                if not self.check_shoulder_constraints():
                    self.ipos.stop()
                yarp.delay(0.01)
            # Reach left arm
            self.ipos_left.stop()
            for i in range(16):
                self.imode_left.setControlMode(i, yarp.VOCAB_CM_POSITION)
            self.ipos_left.positionMove(self.init_arm_enc_values.data())
            while not self.ipos_left.checkMotionDone():
                if not self.check_shoulder_constraints():
                    self.ipos_left.stop()
                yarp.delay(0.01)
            # Reach torso
            self.ipos_torso.stop()
            for i in range(3):
                self.imode_torso.setControlMode(i, yarp.VOCAB_CM_POSITION)
            torso_vector = yarp.Vector([0, 0, 10])
            self.ipos_torso.positionMove(torso_vector.data())
            while not self.ipos_torso.checkMotionDone():
                yarp.delay(0.01)
            # Reach head
            self.ipos_head.stop()
            for i in range(6):
                self.imode_head.setControlMode(i, yarp.VOCAB_CM_POSITION)
            head_vector = yarp.Vector([-35, 0, 0, 0, 0, 0])
            self.ipos_head.positionMove(head_vector.data())
            while not self.ipos_head.checkMotionDone():
                yarp.delay(0.01)
            self.state = 'do_nothing'

        elif self.state == 'get_observation':
            received_image = self._input_image_port.read(True)
            self._in_buf_image.copy(received_image)
            assert self._in_buf_array.__array_interface__['data'][0] == self._in_buf_image.getRawImage().__int__()
            frame = self._in_buf_array
            input_image = frame.copy()
            encs = yarp.Vector(self.ipos.getAxes())
            # read encoders
            ret = self.ienc.getEncoders(encs.data())

        elif self.state == 'open_right_hand':
            self.cart.stopControl()
            self.ipos.stop()
            for i in range(16):
                self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION)
            for j in range(9):
                self.ipos.positionMove(j + 7, float(self.r_hand_open_values[j]))

            while not self.ipos.checkMotionDone():
                if not self.check_shoulder_constraints():
                    self.ipos.stop()
                yarp.delay(0.1)
            self.state = 'do_nothing'

        elif self.state == 'fingers_in_idle':
            for i in range(9):
                self.imode.setControlMode(i+7, yarp.VOCAB_CM_IDLE)
            self.state = 'do_nothing'

        elif self.state == 'fingers_in_position_control':
            for i in range(9):
                self.imode.setControlMode(i+7, yarp.VOCAB_CM_POSITION)
            self.state = 'do_nothing'

        elif self.state == 'go_to_cartesian_pose':
            self.cart.stopControl()
            self.ipos.stop()
            joints_ok = False
            # Check not working
            while not joints_ok:
                for i in range(7):
                    self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
                joints_ok = True
                for i in range(7):
                    if self.imode.getControlMode(i) != yarp.VOCAB_CM_POSITION_DIRECT:
                        joints_ok = False
            self.go_to_cartesian_pose()
            self.state = 'do_nothing'

        elif self.state == 'look_at_fixation_point':
            self.look_at_fixation_point()
            self.state = 'do_nothing'

        elif self.state == 'go_to_distanced_superq_pose':
            self.cart.stopControl()
            self.ipos.stop()
            joints_ok = False
            # Check not working
            while not joints_ok:
                for i in range(7):
                    self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
                joints_ok = True
                for i in range(7):
                    if self.imode.getControlMode(i) != yarp.VOCAB_CM_POSITION_DIRECT:
                        joints_ok = False
            self.go_to_distanced_superq_pose()
            self.state = 'do_nothing'

        elif self.state == 'go_to_distanced_superq_pose_10cm':
            self.cart.stopControl()
            self.ipos.stop()
            joints_ok = False
            # Check not working
            while not joints_ok:
                for i in range(7):
                    self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
                joints_ok = True
                for i in range(7):
                    if self.imode.getControlMode(i) != yarp.VOCAB_CM_POSITION_DIRECT:
                        joints_ok = False
            self.go_to_distanced_superq_pose_10cm()
            self.state = 'do_nothing'

        elif self.state == 'go_to_superq_pose':
            self.cart.stopControl()
            self.ipos.stop()
            joints_ok = False
            # Check not working
            while not joints_ok:
                for i in range(7):
                    self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
                joints_ok = True
                for i in range(7):
                    if self.imode.getControlMode(i) != yarp.VOCAB_CM_POSITION_DIRECT:
                        joints_ok = False
            self.go_to_superq_pose()
            self.state = 'do_nothing'

        elif self.state == 'run_policy':
            self.run_policy()
            self.state = 'do_nothing'

        elif self.state == 'stop_motion':
            self.cart.stopControl()
            self.ipos.stop()

        elif self.state == 'do_nothing':
            pass

        else:
            pass

        return True

    def check_shoulder_constraints(self):
        encs = yarp.Vector(self.ipos.getAxes())
        ret = self.ienc.getEncoders(encs.data())
        r_shoulder_pitch_qpos = encs.get(0)
        r_shoulder_roll_qpos = encs.get(1)
        r_shoulder_yaw_qpos = encs.get(2)
        c = 1.71
        A = np.array([[c, -c, 0],
                      [c, -c, -c],
                      [0, 1, 1],
                      [-c, c, c],
                      [0, -1, -1]])
        b = np.array([347, 366.57, 66.6, 112.42, 213.3])
        x = np.array([r_shoulder_pitch_qpos, r_shoulder_roll_qpos, r_shoulder_yaw_qpos])
        res = np.matmul(A, x) + b > 0
        return bool(res.all())

    def go_to_distanced_superq_pose(self):
        position_yarp = yarp.Vector(3)
        for i in range(3):
            position_yarp[i] = self.superq_distanced_pos[i]

        tmp_quaternion = Quaternion(self.superq_or)
        tmp_axis = tmp_quaternion.axis
        tmp_angle = tmp_quaternion.angle

        axis_angle_yarp = yarp.Vector(4)
        for i in range(3):
            axis_angle_yarp[i] = tmp_axis[i]
        axis_angle_yarp[3] = tmp_angle

        self.cart.goToPoseSync(position_yarp, axis_angle_yarp)

        self.fixation_position = self.superq_center.copy()
        self.look_at_fixation_point()
        return True

    def go_to_distanced_superq_pose_10cm(self):
        position_yarp = yarp.Vector(3)
        for i in range(3):
            position_yarp[i] = self.superq_distanced_pos_10_cm[i]

        tmp_quaternion = Quaternion(self.superq_or)
        tmp_axis = tmp_quaternion.axis
        tmp_angle = tmp_quaternion.angle

        axis_angle_yarp = yarp.Vector(4)
        for i in range(3):
            axis_angle_yarp[i] = tmp_axis[i]
        axis_angle_yarp[3] = tmp_angle

        self.cart.goToPoseSync(position_yarp, axis_angle_yarp)

        self.fixation_position = self.superq_center.copy()
        self.look_at_fixation_point()
        return True

    def look_at_fixation_point(self):
        yarp_fixation_point = yarp.Vector(3)
        for i in range(3):
            yarp_fixation_point[i] = self.fixation_position[i]
        self.gaze_ctrl.lookAtFixationPoint(yarp_fixation_point)

    def execute_action(self, action):
        self.target_ik += action[:6]
        # Compute targets joints, at the moment also obs['joints'] could be used
        encs = yarp.Vector(self.ipos.getAxes())
        ret = self.ienc.getEncoders(encs.data())
        target_joints = np.array([encs.get(k + 7) for k in range(9)]) + action[6:] * 180 / np.pi
        # Clip values
        target_joints = np.clip(target_joints, self.r_hand_open_values, self.r_hand_close_values)
        joints_ok = False
        # Check not working
        while not joints_ok:
            for i in range(7):
                self.imode.setControlMode(i, yarp.VOCAB_CM_POSITION_DIRECT)
            joints_ok = True
            for i in range(7):
                if self.imode.getControlMode(i) != yarp.VOCAB_CM_POSITION_DIRECT:
                    joints_ok = False
        position_yarp = yarp.Vector(3)
        for i in range(3):
            position_yarp[i] = self.target_ik[i]

        qy = Quaternion(axis=[0, 0, 1], angle=self.target_ik[3])
        qp = Quaternion(axis=[0, 1, 0], angle=self.target_ik[4])
        qr = Quaternion(axis=[1, 0, 0], angle=self.target_ik[5])
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L1018
        target_ik_pyquaternion = qr * qp * qy
        target_ik_quaternion = target_ik_pyquaternion.q

        tmp_quaternion = Quaternion(target_ik_quaternion)
        tmp_axis = tmp_quaternion.axis
        tmp_angle = tmp_quaternion.angle

        axis_angle_yarp = yarp.Vector(4)
        for i in range(3):
            axis_angle_yarp[i] = tmp_axis[i]
        axis_angle_yarp[3] = tmp_angle

        self.cart.goToPoseSync(position_yarp, axis_angle_yarp)

        # Go to target fingers qpos
        for j in range(9):
            self.ipos.positionMove(j + 7, target_joints[j])

        # Control gaze
        self.fixation_position = self.superq_center.copy()
        self.look_at_fixation_point()

        self.policy_steps += 1
        return True
