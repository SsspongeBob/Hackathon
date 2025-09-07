import copy
import socket
import struct
import time
import numpy as np
import math
import hardware.util as util
import rtde_control
import rtde_receive
import rtde_io
from hardware.robot.robotiq_gripper import RobotiqGripper


# UR5连接参数
class RobotControl:

    def __init__(self, host="192.168.1.10", port=30003, is_use_gripper=False, limit_space=None):
        # 初始化socket来获得数据
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2
        self.tool_offset = [0, 0, 0.150]
        self.host = host
        self.port = port
        self.is_use_gripper = is_use_gripper
        self.limit_space = limit_space

        # UR官方的RTDE接口,可用于控制和读取数据
        # reference https://gitlab.com/sdurobotics/ur_rtde
        # rtde_c复制UR5的控制
        self.rtde_c = rtde_control.RTDEControlInterface(self.host)
        # rtde_r负责UR5的数据读取
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
        # rtde_io负责控制机器臂的数字输出等
        self.rtde_io = rtde_io.RTDEIOInterface(self.host)
        # robotiq85 gripper configuration
        self.reset()
        if self.is_use_gripper:
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.host, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            # time.sleep(1.5)

    def __del__(self):
        self.rtde_r.disconnect()
        self.rtde_c.disconnect()

    def get_speed(self):
        """获得机器臂的运动速度"""
        return self.rtde_r.getActualTCPSpeed()

    def get_current_angle(self):
        """获得各个关节的角度,返回数组,依次为机座,肩部,肘部,手腕1 2 3"""
        # 获得弧度数组
        actual = np.array(self.rtde_r.getActualQ())
        # 转化为角度
        actual = actual * 180 / math.pi
        return actual

    def get_current_radian(self):
        """返回各个关节的弧度，返回数组,依次为机座,肩部,肘部,手腕1 2 3"""
        return self.rtde_r.getActualQ()

    # 获取为列表数据，返回旋转矢量，rad弧度, m
    def get_current_tcp(self):
        """获得XYZ RXRYRZ,XYZ单位是M,示教器上单位是mm.RXYZ和示教器上一致"""
        return self.rtde_r.getActualTCPPose()

    # 返回旋转矢量姿态rvec，return [x,y,z,rx,ry,rz], mm
    def get_current_tcp_sc(self, mm=False):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.host, self.port))
        data = tcp_socket.recv(1108)
        position = struct.unpack('!6d', data[444:492])
        tcp_socket.close()
        pos_list = np.asarray(position)
        # 单位转化,机械臂传过来的单位是米,我们需要毫米
        if mm:
            pos_list[0:3] = pos_list[0:3] * 1000
        return pos_list

    # 输出欧拉角姿态，rad弧度
    def get_current_tcp_rpy(self):
        """x, y, theta"""
        tcp = self.get_current_tcp()
        rpy = util.rotvec2rpy(tcp[3:])
        return np.asarray([tcp[0], tcp[1], tcp[2], rpy[0], rpy[1], rpy[2]])

    def get_current_pos_same_with_simulation(self):
        tcp = self.get_current_tcp()
        rpy = util.rv2rpy(tcp[3], tcp[4], tcp[5])
        return np.asarray([tcp[1], tcp[0], rpy[-1]])

    def move_up(self, z):
        """机械臂末端向上移动多少mm"""
        tcp = self.get_current_tcp()
        tcp[2] = tcp[2] + z / 1000
        self.rtde_c.moveL(tcp, speed=self.tool_vel, acceleration=self.tool_acc)

    def move_down(self, z):
        """机械臂末端向下移动多少mm"""
        tcp = self.get_current_tcp()
        tcp[2] = tcp[2] - z / 1000
        self.rtde_c.moveL(tcp, speed=self.tool_vel, acceleration=self.tool_acc)

    def reset(self, tool_vel=0.6, tool_acc=0.5):
        """机器臂恢复初始位置(初始位置q需要适应性调整）"""
        self.moveJ_Position_rpy(
            [-0.1, -0.4, 0.35, 180, 0, 45],
            tool_vel,
            tool_acc,
            degree=True
        )

    def moveJ_Angle(self, angles, tool_vel=0.8, tool_acc=0.5, degree=False):
        """机械臂关节角度移动"""
        if degree:
            angles[:] = np.radians(angles[:])
        self.rtde_c.moveJ(
            q=[
                angles[0],
                angles[1],
                angles[2],
                angles[3],
                angles[4],
                angles[5]
            ],
            speed=tool_vel,
            acceleration=tool_acc,
        )

    def moveJ_Position_rv(self, pose: list, tool_vel=0.8, tool_acc=0.5, degree=False, mm=False):
        """机械臂坐标位姿移动, 姿态输入角度"""
        move_pose = copy.deepcopy(pose)
        if degree:
            move_pose[3:] = np.radians(move_pose[3:])
        if mm:
            move_pose = np.array(move_pose)
            move_pose[0:3] = move_pose[0:3] / 1000
            move_pose = move_pose.tolist()
        self.rtde_c.moveJ_IK(
            pose=move_pose,
            speed=tool_vel,
            acceleration=tool_acc,
        )

    def moveJ_Position_rpy(self, pose: list, tool_vel=0.5, tool_acc=0.5, degree=False, mm=False):
        """机械臂坐标位姿移动, 姿态输入角度"""
        move_pose = copy.deepcopy(pose)
        if degree:
            move_pose[3:] = np.radians(move_pose[3:])
        if mm:
            move_pose = np.array(move_pose)
            move_pose[0:3] = move_pose[0:3] / 1000
            move_pose = move_pose.tolist()
        rpy = util.rpy2rotvec(move_pose[3:])
        move_pose[3:] = rpy
        print("[DEBUG]移动到：", move_pose)
        self.rtde_c.moveJ_IK(
            pose=move_pose,
            speed=tool_vel,
            acceleration=tool_acc,
        )

    ## robotiq85 gripper
    # get gripper position [0-255]  open:0 ,close:255
    def get_current_tool_pos(self):
        return self.gripper.get_current_position()

    def log_gripper_info(self):
        print(f"Pos: {str(self.gripper.get_current_position())}")

    def close_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        print("gripper had closed!")
        time.sleep(1.2)
        self.log_gripper_info()

    def open_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        print("gripper had opened!")
        time.sleep(1.2)
        self.log_gripper_info()

    def check_grasp(self):
        # if the robot grasp unsuccessfully ,then the gripper close
        return self.get_current_tool_pos() > 220

    def offset_gripper(self, pose, mm=False):
        if mm:
            tool_offset = np.array(self.tool_offset) * 1000
        else:
            tool_offset = self.tool_offset
        temp = [pose[i] + x for i, x in enumerate(tool_offset)]
        pose[:3] = temp
        return pose

    def plane_grasp(self, position, open_size=0.65, k_acc=0.8, k_vel=0.8, speed=255, force=125):
        rpy = [180, 0, 45]
        # 判定抓取的位置是否处于工作空间
        # for i in range(3):
        #     position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [-pi,pi]
        for i in range(3):
            if rpy[i] > 180:
                rpy[i] -= 360
            elif rpy[i] < -180:
                rpy[i] += 180

        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)'
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        self.reset()
        open_pos = int(-258 * open_size + 230)  # open size:0~0.85cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        print("gripper open size:")
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        print("[DEBUG] 预夹取位置：", pre_position)
        self.moveJ_Position_rpy(pre_position + rpy, 0.3, k_acc, degree=True)

        # Second，achieve grasp position
        self.moveJ_Position_rpy(position + rpy, 0.3, 0.6 * k_acc, degree=True)
        self.close_gripper(speed, force)
        self.moveJ_Position_rpy(pre_position + rpy, 0.6 * k_vel, 0.6 * k_acc, degree=True)
        if self.check_grasp():
            print("Check grasp fail! ")
            self.reset()
            return False

        # Third,put the object into box
        box_position = [0, -0.6, 0.5, 180, 0, 45]  # you can change me!
        self.moveJ_Position_rpy(box_position, k_vel, k_acc, degree=True)
        box_position[2] = 0.3  # down to the 20cm
        self.moveJ_Position_rpy(box_position, k_vel, k_acc, degree=True)
        self.open_gripper(speed, force)
        box_position[2] = 0.45
        self.moveJ_Position_rpy(box_position, k_vel, k_acc, degree=True)
        self.reset()
        print("grasp success!")
        return True

    def test(self):
        self.moveJ_Position_rpy([-0.15, -0.15, 0.6, 180, 0, 45], 0.6, 0.6, degree=True)


if __name__ == "__main__":
    # angle = [40, -140, 130, -80, -90, 110]
    # pose = [-0.05, -0.2, 0.5, 180, 0, 45]
    ros = RobotControl(is_use_gripper=True)
    ros.gripper.move_and_wait_for_pos(255, 255, 125)
    print("gripper open size:")
    ros.log_gripper_info()
    ros.close_gripper()
    ros.open_gripper()
    # box_position = [0, -0.6, 0.5, 180, 0, 45]  # you can change me!
    # box_position = [0, -600, 500, 180, 0, 45]  # you can change me!
    # ros.moveJ_Position_rpy(box_position, 0.1, 0.6, degree=True, mm=True)
    # box_position[2] = 300  # down to the 20cm
    # print(box_position)
    # ros.moveJ_Position_rpy(box_position, 0.1, 0.6,degree=True, mm=True)
    ros.reset()
    # ros.moveJ_Angle(angle, degree=True)
    del ros
    # 了解旋转矢量和rpy原理
    # 理解标定板应该在什么位置
