import cv2

# 读取图片
img = cv2.imread("untitled.jpg")

# 获取图片的宽度和高度
h, w = img.shape[:2]

# 网格大小
grid_size = 8

# 计算每个格子的宽度和高度
width_step = w // grid_size
height_step = h // grid_size

# 创建一个副本，以免修改原图
img_with_grid = img.copy()

# 绘制水平线
for i in range(1, grid_size):
    y = i * height_step
    cv2.line(img_with_grid, (0, y), (w, y), (0, 0, 255), 1)  # 红色线，BGR格式

# 绘制垂直线
for j in range(1, grid_size):
    x = j * width_step
    cv2.line(img_with_grid, (x, 0), (x, h), (0, 0, 255), 1)  # 红色线，BGR格式

# 保存带有网格线的图片
cv2.imwrite("d1_with_grid.jpg", img_with_grid)