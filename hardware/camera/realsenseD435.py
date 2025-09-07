import os
import threading
import time
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseCamera:
    def __init__(self, output_dir="catch_result/photo"):
        self._display_thread = None
        self.last_depth_image = None
        self.last_color_image = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 相机内参
        self.intrisics = None
        # 获取设备产品线，用于设置支持分辨率
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        # 创建对齐对象（深度对齐颜色）
        self.align = rs.align(rs.stream.color)
        self.output_dir = output_dir
        self.is_running = False

        self.image_ready = threading.Condition(threading.Lock())  # 添加条件锁

        # 设置配置：RGB与深度流均为640x480
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.cfg = self.pipeline.start(self.config)
        # 创建深度处理管道，包括temporal和spatial滤波器
        self.depth_pipeline = rs.pipeline_profile()
        # 孔洞填充过滤器
        self.hole_filling = rs.hole_filling_filter()
        # 时间滤波器
        self.temporal = rs.temporal_filter()
        # 边缘滤波
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 8)
        # 创建深度处理管道，包括temporal和spatial滤波器
        self.depth_pipeline = rs.pipeline_profile()
        self.depth_filters = [self.hole_filling, self.temporal, self.spatial]

        self.colorizer = rs.colorizer()

    def start_display_and_capture(self):
        self.is_running = True
        # self.pipeline.start(self.config)
        self._display_thread = threading.Thread(target=self._display_and_capture)
        self._display_thread.start()

    def _display_and_capture(self):
        try:
            time.sleep(3)
            while self.is_running:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)

                # 获取并处理深度数据
                depth_frame = frames.get_depth_frame()
                for filter_ in self.depth_filters:
                    depth_frame = filter_.process(depth_frame)
                # 保存原深度图
                depth = np.asanyarray(depth_frame.get_data())

                # 在深度图像上应用颜色图
                depth_colormap = self.colorizer.colorize(depth_frame)
                depth_image = np.asanyarray(depth_colormap.get_data())

                # 获取彩色图像
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                # 因为opencv使用的是BGR,但是相机用的是RGB,所以要转换
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # 将RGB图像和深度图像堆叠在一起
                combined_image = np.hstack((color_image, depth_image))

                # 显示堆叠后的图像
                cv2.imshow("RGB & Depth Images", combined_image)

                with self.image_ready:  # 在这里获取锁
                    self.last_color_image = color_image
                    self.last_depth_image = depth

                    self.image_ready.notify_all()  # 通知所有等待的线程，图像已准备好

                cv2.waitKey(1)
        except Exception as e:
            print(f"Error during display and capture: {e}")
            self.is_running = False
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()

    def save_screenshot(self, photo_path) -> None:
        with self.image_ready:
            while not (
                    self.last_color_image is not None and self.last_depth_image is not None
            ):
                self.image_ready.wait()  # 等待图像准备好
            # 保存图片
            # 保存待识别的图片
            cv2.imwrite(photo_path, self.last_color_image)

    def stop(self):
        self.is_running = False

    # 获取一个获取相机内参的函数,并保存
    def get_intrinsics(self):
        rgb_profile = self.cfg.get_stream(rs.stream.color)
        raw_intrinsics = rgb_profile.as_video_stream_profile().intrinsics
        print("camera intrinsics", raw_intrinsics)
        # camera intrinsics form
        # [[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        intrinsics = np.array(
            [
                raw_intrinsics.fx,
                0,
                raw_intrinsics.ppx,
                0,
                raw_intrinsics.fy,
                raw_intrinsics.ppy,
                0,
                0,
                1,
            ]
        ).reshape(3, 3)
        return intrinsics

    def get_cam_depth_scale(self):
        depth_scale = self.cfg.get_device().first_depth_sensor().get_depth_scale()
        return depth_scale

    def get_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        # depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image


def test():
    PHOTO_PATH = "photo/"

    if not os.path.exists(PHOTO_PATH):
        os.makedirs(PHOTO_PATH)

    def delete_photo():
        for files in os.listdir(PHOTO_PATH):
            if files.endswith(".png"):
                os.remove(os.path.join(PHOTO_PATH, files))

    camera = RealSenseCamera()
    # 线程开始运行
    camera.start_display_and_capture()
    # 递增，拍照次数
    num = 0
    # 文件名,使用ASCII编码
    name_num = 97
    # 删除上次的拍照照片
    delete_photo()
    # 循环读取每一帧
    while num < 3:
        time.sleep(2)
        # 拍照操作
        camera.save_screenshot(f"photo/{chr(name_num)}.png")

        print("第", (num + 1), "张照片拍摄完成")
        print("-------------------------")
        # 循环标志+1    PHOTO_PATH = "photo/"

    if not os.path.exists(PHOTO_PATH):
        os.makedirs(PHOTO_PATH)

    def delete_photo():
        for files in os.listdir(PHOTO_PATH):
            if files.endswith(".png"):
                os.remove(os.path.join(PHOTO_PATH, files))

    camera = RealSenseCamera()
    # 线程开始运行
    camera.start_display_and_capture()
    # 递增，拍照次数
    num = 0
    # 文件名,使用ASCII编码
    name_num = 97
    # 删除上次的拍照照片
    delete_photo()
    # 循环读取每一帧
    while num < 3:
        time.sleep(2)
        # 拍照操作
        camera.save_screenshot(f"photo/{chr(name_num)}.png")

        print("第", (num + 1), "张照片拍摄完成")
        print("-------------------------")
        # 循环标志+1
        num += 1
        # 文件名ASCII码+1
        name_num += 1
        # camera.stop()
        num += 1
        # 文件名ASCII码+1
        name_num += 1
    # camera.stop()


if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.get_data()
    cv2.namedWindow('color')
    cv2.namedWindow('depth')
    while True:
        camera_color_img, camera_depth_img = cam.get_data()
        bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('color', bgr_data)
        cv2.imshow('depth', camera_depth_img)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
