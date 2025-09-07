# UR5grasp_demo:从标定到graspnet实现

涉及keyboard模块的python文件需要在root中运行
统一单位为m，弧度制，旋转矢量为主

# 一、安装

python == 3.10.6  
opencv版本:4.6.0(低于4.1.0没有camera_calibration函数)
PyTorch 1.7.1+ with CUDA 11.x/12.1   
cuda 11.8对应torch如下

```shell
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

**如果报错，调整numpy版本为：**

```shell
 pip install numpy == 1.26.4
```

## URcontrol

[URrtde](https://github.com/UniversalRobots/RTDE_Python_Client_Library/tree/main):最新版本用法与本库不兼容   
[ur_rtde](https://gitlab.com/sdurobotics/ur_rtde):本库使用的旧版安装方法

```shell
pip install ur_rtde
```

## realsense相机

下载[realsenseSDK](https://github.com/IntelRealSense/librealsense): SDK为可视化界面与代码运行无关,可以不下载    
安装[pyrealsense2](https://github.com/IntelRealSense/librealsense/tree/development/wrappers/python):

```sh
pip install pyrealsense2
```

## Graspnet

[graspnet baseline](https://github.com/graspnet/graspnet-baseline):
Get the code.

```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```

Install packages via Pip.

```bash
pip install -r requirements.txt
```

Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).

```bash
cd pointnet2
python setup.py install
```

Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).

```bash
cd knn
python setup.py install
```

[graspnetAPI](https://github.com/graspnet/graspnetAPI)：  
python3.10版本只能下载源码安装环境，同时需要修改setup中sklearn为scikit-learn
```shell
# pip
pip install graspnetAPI
# 源码下载
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## AnyGrasp

需要申请license  
[csdn安装指南](https://blog.csdn.net/weixin_63116759/article/details/146483943?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7ECtr-4-146483943-blog-137137558.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7ECtr-4-146483943-blog-137137558.235%5Ev43%5Epc_blog_bottom_relevance_base2&utm_relevant_index=8)  
[anygrasp官网](https://github.com/graspnet/anygrasp_sdk)  
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine#anaconda): 先于anygrasp安装

## SAM

[segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file)  
`python>=3.8`  
`pytorch>=1.7`   
`torchvision>=0.8`  
pip git指令安装

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

官方下载

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

以下**可选依赖项**对于掩码后处理、以 COCO 格式保存掩码、示例笔记本以及以 ONNX 格式导出模型是必要的。运行示例笔记本也需要 `jupyter`

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

模型参数下载url

```
# huge最大 至少6g以上
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
# large 中等 至少5.4g以上
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
# base 最小 至少5.4g以上
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

## GPT-4o

## TTS，STT

音频处理
安装pyaudio

```shell
# 安装pyaudio的前置环境，安装的系统环境里
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
# 卸载用 sudo apt-get remove lib...
pip install pyaudio
```

安装dashscope

```shell
pip install dashscope
```

# 二、使用

## 标定 calibration

先运行get_camera_intrinsic, 再运行UR_get_calibration_data, 再进行标定

## 夹取 src, run_graspnet

run_touch相机手动夹取
run_graspnet自动生成位姿

## Grasp_agent

# 三、DEBUG

## keyboard需要root权限

**UR_get_calibration_data.py运行报错**  
需要在root设置conda环境后,使用python指令运行

```shell
su
# 输入密码
# 设置conda
source /home/new_user/miniconda3/bin/activate
# 设置使之后每次进入root都会使用conda环境
conda init 
conda activate calibration_grasp
# 进入文件夹
cd <basedir>
# 运行
python UR_get_calibration_data.py
```