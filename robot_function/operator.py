import base64
import json
from io import BytesIO
from process_utils.convert_update import convert_new
from process_utils.grasp_process import run_grasp_inference
from hardware.robot.UR_control_rtde import RobotControl
from hardware.util import *
import cv2
import numpy as np
import pyrealsense2 as rs
import requests
from robot_function.LLM import LLM_Structured_Chatter, ImageRecognition, generate_image_prompt


class Operator:
    def __init__(self):
        # 手眼标定外参
        T_cam2end = np.loadtxt(
            "/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/calibration_data/T_cam2end.txt",
            delimiter=',')
        self.R_cam2end = T_cam2end[:3, :3]
        self.t_cam2end_vector = T_cam2end[:3, 3:4]

        self.cam_depth_scale = np.loadtxt(
            "/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/calibration_data/camera_depth_scale.txt",
            delimiter=',')
        self.cam_intrinsics_calib = np.loadtxt(
            "/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/calibration_data/camera_intrinsic_calib.txt",
            delimiter=',')
        self.tool_orientation = [180, 0, 45]
        self.end2base_pose = []
        self.robot = RobotControl('192.168.1.10', 30003, is_use_gripper=True)
        self.color_image = None
        self.depth_image = None
        self.position = None
        self.mask = None

    def display(self):
        pipeline = rs.pipeline()
        config = rs.config()

        # TODO 考虑要不要还640*480的，能不能保证质量的前提，减少计算量，加快运行速度
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        try:
            profile = pipeline.start(config)
            color_sensor = profile.get_device().query_sensors()[1]
            # color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            color_sensor.set_option(rs.option.exposure, 156.000)
            color_sensor.set_option(rs.option.brightness, 50)
            # 新增：创建对齐对象，将深度图与彩色图对齐
            align = rs.align(rs.stream.color)  # 对齐到彩色图像流
            frames = pipeline.wait_for_frames()
            if not frames:
                raise RuntimeError("No color frame captured")
            # 对齐帧
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            color_sensor.set_option(rs.option.saturation, 64)  # default 64
            color_sensor.set_option(rs.option.sharpness, 50)  # default 50
            if not color_frame or not depth_frame:
                raise RuntimeError("No frame captured")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imshow('color', color_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            self.color_image = color_image
            self.depth_image = depth_image
        finally:
            pipeline.stop()

    def pick(self, object_no: int):
        mask = self.get_mask_image(object_no)
        mask = np.asarray(mask).astype(np.uint8)
        t_target2cam, R_target2cam, width = run_grasp_inference(
            self.color_image,
            self.depth_image,
            mask
        )
        print(f"[DEBUG] Grasp预测结果 - 平移: {t_target2cam}, 旋转矩阵:\n{R_target2cam}")

        # 单位为mm，后面转换成m
        end2base_pose = self.robot.get_current_tcp_sc()
        self.end2base_pose = end2base_pose
        T_end2base = attitudeVectorToMatrix(end2base_pose)
        # print("[DEBUG] 官方api计算出对应的齐次矩阵:", T_ee2base)

        T_target2base = convert_new(
            t_target2cam,
            R_target2cam,
            self.t_cam2end_vector,
            self.R_cam2end,
            T_end2base
        )
        print("[DEBUG] 基坐标系抓取齐次矩阵:", T_target2base)

        base_pose = T2pose(T_target2base)
        base_pose = np.array(base_pose, dtype=float)
        print("[DEBUG] 最终抓取位姿是什么:", base_pose)
        # .............................................................
        # 预抓取计算, 将夹取位置的z轴方向移动0.15m，改变旋转矩阵z轴方向的参数
        pre_grasp_offset = 0.15
        pre_grasp_pose = np.array(base_pose, dtype=float).copy()
        pre_rotation_mat = cv2.Rodrigues(base_pose[3:])[0]
        z_axis = pre_rotation_mat[:, 2]
        pre_grasp_pose[:3] -= z_axis * pre_grasp_offset

        try:
            print(f"预抓取位姿: {pre_grasp_pose.tolist()}")
            self.robot.moveJ_Position_rv(pre_grasp_pose.tolist())

            print(f"实际抓取: {base_pose}")
            self.robot.moveJ_Position_rv(base_pose.tolist())

            print("闭合夹爪")
            self.robot.close_gripper(speed=255, force=50)
            # if self.robot.check_grasp(): raise RuntimeError(f"夹爪闭合失败")
            self.robot.moveJ_Position_rv(pre_grasp_pose.tolist())
        except RuntimeError as e:
            print(f"[ERROR] 运动异常: {str(e)}")
            self.robot.reset()
            self.robot.open_gripper()
        return self.robot.get_current_tcp_sc()

    def place(self, place_no: int):
        center = self.get_center(place_no)
        # center = [229, 480]
        u = center[0]
        v = center[1]
        click_z = self.depth_image[v][u] * self.cam_depth_scale
        click_x = np.multiply(u - self.cam_intrinsics_calib[0][2], click_z / self.cam_intrinsics_calib[0][0])
        click_y = np.multiply(v - self.cam_intrinsics_calib[1][2], click_z / self.cam_intrinsics_calib[1][1])

        click_point = np.asarray([click_x, click_y, click_z])
        click_point = click_point.reshape(3, 1)

        # get m_base2grasp
        pose_base2gripper = self.end2base_pose

        m_base2gripper = attitudeVectorToMatrix(pose_base2gripper, "")

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)

        pos_grasp2obj = np.dot(self.R_cam2end, click_point) + self.t_cam2end_vector
        pos_base2obj = np.dot(m_base2gripper[0:3, 0:3], pos_grasp2obj) + m_base2gripper[0:3, 3:]
        pos_base2obj = pos_base2obj.reshape(1, 3)
        target_position = pos_base2obj[0]

        pos = [target_position[0], target_position[1], target_position[2] + 0.12,
               self.tool_orientation[0],
               self.tool_orientation[1], self.tool_orientation[2]]
        pre_pos = [target_position[0], target_position[1], target_position[2] + 0.3,
                   self.tool_orientation[0],
                   self.tool_orientation[1], self.tool_orientation[2]]
        print("目标位姿：\n", pos)
        self.robot.moveJ_Position_rpy(pre_pos, degree=True)
        self.robot.moveJ_Position_rpy(pos, degree=True)
        self.robot.open_gripper(speed=255, force=125)
        self.robot.moveJ_Position_rpy(pre_pos, degree=True)
        self.robot.reset()
        self.end2base_pose = []
        return self.robot.get_current_tcp_sc()

    def init_sam(self):
        _, img_encoded = cv2.imencode('.jpg', self.color_image)
        img_bytes = BytesIO(img_encoded.tobytes())
        # 使用 requests 发送字节流
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post('http://47.117.124.84:4001/init', files=files)
        print(response.json()["message"])

    # TODO 利用GPT-4o判断是否完成抓取
    def check(self, image: str) -> int:
        result: ImageRecognition = LLM_Structured_Chatter.invoke(
            generate_image_prompt(
                "I have labeled a bright numeric ID atthe center for each visual object in the image.Please tell me the IDs for: The curved cable.",
                image,
            )
        )
        return int(result.id)

    def get_annotated_image(self):
        url = "http://47.117.124.84:4001/anno_img"
        response = requests.get(url)
        print(response.json()["message"])
        anno_json = response.json()["anno_img_list"]
        anno_img = json.loads(anno_json)
        annotated_image = np.asarray(anno_img).astype(np.uint8)
        cv2.imwrite("/home/lwc/python_script/URgrasp_agent/UR/URgrasp_agent/robot_function/anno_data/annotated_image.png", annotated_image)
        cv2.imshow("annotated image", annotated_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        _, buffer = cv2.imencode('.png', annotated_image)
        anno_img_base64 = base64.b64encode(buffer).decode('utf-8')
        return anno_img_base64

    def get_mask_image(self, object_no):
        url = f"http://47.117.124.84:4001/mask_no?object_no={object_no}"
        # data = {"object_no": object_no}  # 假设要获取对象编号为1的掩码列表
        response = requests.post(url)
        print(response.json()["message"])
        mask_json = response.json()["mask_list"]
        mask = json.loads(mask_json)
        return mask

    def get_center(self, place_no) -> tuple[int, int]:
        url = f"http://47.117.124.84:4001/center_no?object_no={place_no}"
        # data = {"object_no": place_no}  # 假设要获取对象编号为1的掩码列表
        response = requests.post(url)
        print(response.json()["message"])
        center_json = response.json()["place_center_list"]
        center = json.loads(center_json)
        return center


if __name__ == "__main__":
    a = Operator()
    a.display()
    a.init_sam()
    # a.get_center(0)
    # annotated_image = a.get_annotated_image()
    # annotated_image = np.asarray(annotated_image).astype(np.uint8)
    # cv2.imwrite("anno_img.png", annotated_image)
    # mask_image = a.get_mask_image(21)
    # mask_image = np.asarray(mask_image).astype(np.uint8)
    # cv2.imwrite("mask_img.png", mask_image)
    # a.pick(0)
    # gpt_4o(anno_i)
    # a.pick(3)
    # a.place(3)
