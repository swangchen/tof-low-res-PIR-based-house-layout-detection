import numpy as np
import cv2
from sklearn.neighbors import KDTree

# 读取图片
img = cv2.imread("d1.jpg")

# 获取图片的宽度和高度
w = img.shape[1]
h = img.shape[0]

# 网格大小
grid_size = 8

# 计算每个格子的宽度和高度
width_step = w // grid_size
height_step = h // grid_size

# 循环遍历并处理每个格子
for i in range(grid_size):
    for j in range(grid_size):
        # 计算当前格子的左上角坐标
        left = j * width_step
        upper = i * height_step
        # 计算当前格子的右下角坐标
        right = (j + 1) * width_step
        lower = (i + 1) * height_step

        # 裁剪图片
        roi = img[upper:lower, left:right]

        # 将ROI转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 应用阈值
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 提取像素值
        pixels = thresh.flatten()

        # 创建KD树
        kd_tree = KDTree(pixels.reshape(-1, 1))

        # 这里可以进行更多的操作，例如查询最近的邻居等
        # ...

# 保存处理后的图片
cv2.imwrite("d1_processed.jpg", img)