import numpy as np


def convert_new(
        t_grasp_target2cam,  # GraspNet输出的平移向量（相机坐标系）
        R_grasp_target2cam,  # GraspNet输出的旋转矩阵（相机坐标系，3x3）
        t_cam2end,  # 手眼标定平移向量（相机->末端）
        R_cam2end,  # 手眼标定旋转矩阵（相机->末端）
        T_end2base  # 初始末端->基座
):
    """
    优化后的坐标系转换函数，主要改动：
    1. 修正坐标系转换链路顺序
    2. 处理GraspNet夹爪坐标系定义差异
    返回 [base_x, base_y, base_z, base_rx, base_ry, base_rz]
    """
    # ================== 坐标系对齐预处理 ==================

    # 修正GraspNet坐标系定义（X轴朝向 -> Z轴朝向），绕y转90, 符合我的夹爪
    R_adjust = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=np.float32)

    # 调整后的抓取姿态（相机坐标系）
    adjusted_rotation = R_grasp_target2cam @ R_adjust  # 右乘
    # adjusted_rotation = R_adjust @ R_grasp_target2cam # 左乘
    adjusted_translation = t_grasp_target2cam

    # ================== 坐标系转换链路 ==================
    # 1. 构造抓取位姿的齐次矩阵（相机坐标系）
    T_grasp_target2cam = np.eye(4)
    T_grasp_target2cam[:3, :3] = adjusted_rotation
    T_grasp_target2cam[:3, 3] = adjusted_translation

    # 2. 手眼标定矩阵（相机->末端）
    T_cam2end = np.eye(4)
    T_cam2end[:3, :3] = R_cam2end
    T_cam2end[:3, 3:] = t_cam2end

    # 3. 计算完整的转换链路：T_target2base = T_end2base @ T_cam2end @ T_grasp_target2cam
    T_target2base = T_end2base @ T_cam2end @ T_grasp_target2cam

    return T_target2base
