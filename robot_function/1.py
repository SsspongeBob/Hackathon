import numpy as np
import cv2

def generate_blank_mask_cv(width=1280, height=720):
    """
    生成黑白掩码（OpenCV格式）
    返回:
        mask (np.ndarray): 黑白掩码，形状 (height, width)，dtype=uint8
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255  # 全白掩码
    return mask

# 使用示例
mask = generate_blank_mask_cv()

# 显示掩码
cv2.imshow("Mask", mask)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 保存掩码
cv2.imwrite("anno_data/mask.png", mask)
print("黑白掩码已保存为 mask.png")
