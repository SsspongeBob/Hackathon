import re
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# 旋转矢量转旋转矩阵
def rv2rm(rx, ry, rz):
    theta = np.linalg.norm([rx, ry, rz])
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta

    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    R = np.zeros((3, 3))
    R[0][0] = kx * kx * v + c
    R[0][1] = kx * ky * v - kz * s
    R[0][2] = kx * kz * v + ky * s

    R[1][0] = ky * kx * v + kz * s
    R[1][1] = ky * ky * v + c
    R[1][2] = ky * kz * v - kx * s

    R[2][0] = kz * kx * v - ky * s
    R[2][1] = kz * ky * v + kx * s
    R[2][2] = kz * kz * v + c

    return R


# 旋转矩阵转rpy
def rm2rpy(R):
    sy = np.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2][1], R[2][2])
        y = np.arctan2(-R[2][0], sy)
        z = np.arctan2(R[1][0], R[0][0])
    else:
        x = np.arctan2(-R[1][2], R[1][1])
        y = np.arctan2(-R[2][0], sy)
        z = 0

    return np.asarray([x, y, z])


# rpy转旋转矩阵
def rpy2rm(rpy):
    # Rx = np.zeros((3, 3), dtype=rpy.dtype)
    # Ry = np.zeros((3, 3), dtype=rpy.dtype)
    # Rz = np.zeros((3, 3), dtype=rpy.dtype)

    R0 = np.zeros((3, 3), dtype=float)

    thetaX = rpy[0]
    thetaY = rpy[1]
    thetaZ = rpy[2]

    cx = np.cos(thetaX)
    sx = np.sin(thetaX)

    cy = np.cos(thetaY)
    sy = np.sin(thetaY)

    cz = np.cos(thetaZ)
    sz = np.sin(thetaZ)

    R0[0][0] = cz * cy
    R0[0][1] = cz * sy * sx - sz * cx
    R0[0][2] = cz * sy * cx + sz * sx
    R0[1][0] = sz * cy
    R0[1][1] = sz * sy * sx + cz * cx
    R0[1][2] = sz * sy * cx - cz * sx
    R0[2][0] = -sy
    R0[2][1] = cy * sx
    R0[2][2] = cy * cx
    return R0


# 旋转矩阵转旋转矢量
def rm2rv(R):
    theta = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    K = (1 / (2 * np.sin(theta))) * np.asarray(
        [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    )
    r = theta * K
    return r


def rv2rpy(rx, ry, rz):
    R = rv2rm(rx, ry, rz)
    rpy = rm2rpy(R)
    return rpy


def rpy2rv(rpy):
    R = rpy2rm(rpy)
    rv = rm2rv(R)
    return rv


def rpy2rotating_vector(rpy):
    # rpy to R
    R = rpy2R(rpy)
    # R to rotating_vector
    return R2rotating_vector(R)


def rpy2R(rpy):  # [r,p,y] 单位rad
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rpy[0]), -np.sin(rpy[0])],
            [0, np.sin(rpy[0]), np.cos(rpy[0])],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(rpy[1]), 0, np.sin(rpy[1])],
            [0, 1, 0],
            [-np.sin(rpy[1]), 0, np.cos(rpy[1])],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(rpy[2]), -np.sin(rpy[2]), 0],
            [np.sin(rpy[2]), np.cos(rpy[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(rot_z, np.dot(rot_y, rot_x))
    return R


def R2rotating_vector(R):
    theta = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    print(f"theta:{theta}")
    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    return np.array([rx, ry, rz]) * theta


def R2rpy(R):
    # assert (isRotationMatrix(R))
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.atan2(R[2, 1], R[2, 2])
        y = np.atan2(-R[2, 0], sy)
        z = np.atan2(R[1, 0], R[0, 0])
    else:
        x = np.atan2(-R[1, 2], R[1, 1])
        y = np.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# sc
def rpy2rotvec(rpy):
    np.set_printoptions(suppress=True)
    rot = Rotation.from_euler('xyz', rpy)  # 假设旋转顺序为 xyz
    rotvec = rot.as_rotvec()
    return rotvec


def rotvec2rpy(rvec):
    np.set_printoptions(suppress=True)
    rot = Rotation.from_rotvec(rvec)  # 假设旋转顺序为 xyz
    rpy = rot.as_euler("xyz")
    return rpy


def HomogeneousMtr2RT(HomoMtr):
    """
    * @brief    齐次矩阵分解为旋转矩阵与平移矩阵
    * @note
    * @param	const Mat& HomoMtr  4*4齐次矩阵
    * @param	Mat& R              输出旋转矩阵
    * @param	Mat& T				输出平移矩阵
    * @return
    """
    R = HomoMtr[:3, :3]
    T = HomoMtr[:3, 3:4]
    return R, T


def R_T2HomogeneousMatrix(R, T):
    """
    * @brief   将旋转矩阵与平移向量合成为齐次矩阵
    * @note
    * @param   Mat& R   3*3旋转矩阵
    * @param   Mat& T   3*1平移矩阵
    * @return  Mat      4*4齐次矩阵
    """
    HomoMtr = np.zeros((4, 4), np.float32)

    if T.shape[0] == 1:
        T = T.T
    HomoMtr[:3, :3] = R
    HomoMtr[:3, 3:4] = T
    HomoMtr[3, 3] = 1
    return HomoMtr


def isRotatedMatrix(R):  # 旋转矩阵的转置矩阵是它的逆矩阵，逆矩阵 * 矩阵 = 单位矩阵
    """
    * @brief	检查是否是旋转矩阵
    * @note
    * @param
    * @param
    * @param
    * @return  true : 是旋转矩阵， false : 不是旋转矩阵
    """
    temp33 = R[:3, :3]  # 无论输入是几阶矩阵，均提取它的三阶矩阵
    shouldBeIdentity = np.matmul(temp33, temp33.T)  # 是旋转矩阵则乘积为单位矩阵
    # print(shouldBeIdentity)
    I = np.eye(3, dtype=shouldBeIdentity.dtype)

    return bool(cv2.norm(I, shouldBeIdentity) < 1e-6)


def eulerAngleToRotateMatrix(vec, seq):
    """
    * @brief   欧拉角RPY转换为旋转矩阵
    * @param    const std::string& seq  指定欧拉角的排列顺序；（机械臂的位姿类型有xyz,zyx,zyz几种，需要区分）
    * @param    const Mat& eulerAngle   欧拉角（1*3矩阵）, 角度值
    * @return   返回3*3旋转矩阵
    """
    # CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3) # 检查参数是否正确
    # eulerAngle /= (180 / CV_PI)  # 度转弧度
    # Matx13d m(eulerAngle)			# <double, 1, 3>
    eulerAngle = vec[0, 3:7].reshape(1, 3)
    if abs(eulerAngle[0, 0]) >= 10 or abs(eulerAngle[0, 1]) >= 10 or abs(eulerAngle[0, 2]) >= 10:
        eulerAngle /= (180 / np.pi)  # a=a/b  角度转弧度
    rx, ry, rz = eulerAngle[0, 0], eulerAngle[0, 1], eulerAngle[0, 2]  # 绕x轴旋转的角度   类推
    rxs = np.sin(rx)  # 角度正交分解
    rxc = np.cos(rx)
    rys = np.sin(ry)
    ryc = np.cos(ry)
    rzs = np.sin(rz)
    rzc = np.cos(rz)

    RotX = np.array([1, 0, 0, 0, rxc, -rxs, 0, rxs, rxc]).reshape(3, 3)
    RotY = np.array([ryc, 0, rys, 0, 1, 0, -rys, 0, ryc]).reshape(3, 3)
    RotZ = np.array([rzc, -rzs, 0, rzs, rzc, 0, 0, 0, 1]).reshape(3, 3)

    # 机器人学中的矩阵变化遵循左乘原则
    if seq == "zyx":
        rotMat = RotX @ RotY @ RotZ
    elif seq == "yzx":
        rotMat = RotX @ RotZ @ RotY
    elif seq == "zxy":
        rotMat = RotY @ RotX @ RotZ
    elif seq == "yxz":
        rotMat = RotZ @ RotX @ RotY
    elif seq == "xyz":
        rotMat = RotZ @ RotY @ RotX
    elif seq == "xzy":
        rotMat = RotY @ RotZ @ RotX
    else:
        print("Euler Angle Sequence string is wrong...")
    if not isRotatedMatrix(rotMat):  # 欧拉角特殊情况下会出现死锁
        print("Euler Angle convert to RotatedMatrix failed...")

    return rotMat


def attitudeVectorToMatrix(m, seq=""):
    """
    * @brief      将采集的原始数据转换为齐次矩阵（从机器人控制器中获得的）
    * @param	  Mat& m    1*6//1*10矩阵 ， 元素为： x,y,z,rx,ry,rz  or x,y,z, q0,q1,q2,q3,rx,ry,rz
    * @param	  string& seq         原始数据使用欧拉角表示时，坐标系的旋转顺序
    * @return	  返回转换完的齐次矩阵
    """
    # CV_Assert(m.total() == 6 | | m.total() == 10);
    # if m.cols == 1:# 转置矩阵为行矩阵
    #     m = m.T
    m = m.reshape(1, -1)
    r, t = m[0, 3:6], m[0, :3]
    t = t.T.reshape(3, -1)
    temp = np.eye(4, dtype=np.float64)  # 创建4*4的单位矩阵
    if seq == "":
        temp[:3, :3] = cv2.Rodrigues(r)[0]  # 罗德利斯转换,旋转矢量，欧拉角
    else:
        temp[:3, :3] = eulerAngleToRotateMatrix(m, seq)  # 欧拉角
    # 存入平移矩阵
    temp[:3, 3:4] = t
    return temp  # 返回转换结束的齐次矩阵


# T 4x4矩阵转换为pose
def T2pose(T):
    R, t = HomogeneousMtr2RT(T)
    rvec, _ = cv2.Rodrigues(R)
    return [t[0][0], t[1][0], t[2][0], rvec[0], rvec[1], rvec[2]]

def T2pose_rpy(T):
    R, t = HomogeneousMtr2RT(T)
    temp = Rotation.from_matrix(R)
    rvec = temp.as_euler("xyz")
    return [t[0][0], t[1][0], t[2][0], rvec[0], rvec[1], rvec[2]]

# 将机械臂末端的姿态向量转换为旋转矩阵和位移向量
def pose_vectors_to_end2base_transforms(pose_vectors):
    R_end2bases = []
    t_end2bases = []
    for pose_vector in pose_vectors:
        R_end2base = cv2.Rodrigues(pose_vector[3:])[0]
        t_end2base = pose_vector[:3]  # 位移向量
        R_end2bases.append(R_end2base)
        t_end2bases.append(t_end2base)
    return R_end2bases, t_end2bases


# 定义排序键函数
def sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    else:
        return 0


if __name__ == "__main__":
    # pose = [-0.05, -0.2, 0.2, 180.0, 0.0, 45.0]
    # rpy = [
    #     pose[3] / 180.0 * np.pi,
    #     pose[4] / 180.0 * np.pi,
    #     pose[5] / 180.0 * np.pi,
    # ]  # Roll, Pitch, Yaw in radians
    # rotvec = rpy2rotvec(rpy)
    # r2 = rpy2rotating_vector(rpy)
    # r3 = rpy2rv(rpy)
    # r4 = rotvec2rpy(rotvec)
    # r4 = np.rad2deg(r4)
    # print("rpy2rotvec:", rotvec)
    # print("rpy2rotating_vector:", r2)
    # print("rpy2rv:", r3)
    # print("rotvec2rpy:", r4)

    T = np.loadtxt("../calibration/calibration_data/data/T_cam2end.txt", delimiter=",")
    pose = T2pose(T)
    pose2 = T2pose_rpy(T)
    print(pose)
    print(pose2)
