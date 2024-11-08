import matplotlib.pyplot as plt
import numpy as np

# 定义矩阵数据
d1 = np.array([
    [1, 1, 2, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2, 1, 2],
    [2, 1, 2, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 2, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 2, 1, 2]
])

d2 = np.array([
    [1, 0, 0, 0, 1, 1, 2, 1],
    [1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 1, 1],
    [1, 3, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

d3 = np.array([
    [1, 1, 1, 1, 0, 2, 1, 1],
    [1, 1, 1, 0, 0, 1, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 2],
    [1, 1, 1, 1, 2, 1, 1, 2],
    [1, 1, 2, 2, 2, 2, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 1, 2, 1]
])

d4 = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 2, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 2, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2, 1],
    [2, 1, 1, 2, 2, 2, 2, 2]
])

# 创建一个颜色映射，将 1 映射为白色，将 2 和 3 映射为黑色
cmap = plt.cm.gray
norm = plt.Normalize(1, 3)

# 绘制图像
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
matrices = [d1, d2, d3, d4]
titles = ['d1', 'd2', 'd3', 'd4']

for ax, matrix, title in zip(axes, matrices, titles):
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 定义原始矩阵
d1 = np.array([
    [1, 1, 2, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2, 1, 2],
    [2, 1, 2, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 2, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 2, 1, 2]
])

d2 = np.array([
    [1, 0, 0, 0, 1, 1, 2, 1],
    [1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 1, 1],
    [1, 3, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

d3 = np.array([
    [1, 1, 1, 1, 0, 2, 1, 1],
    [1, 1, 1, 0, 0, 1, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 2],
    [1, 1, 1, 1, 2, 1, 1, 2],
    [1, 1, 2, 2, 2, 2, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 1, 2, 1]
])

d4 = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 2, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 2, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2, 1],
    [2, 1, 1, 2, 2, 2, 2, 2]
])

# 将矩阵组合成c
top = np.hstack((d1, d2))  # 上半部分
bottom = np.hstack((d3, d4))  # 下半部分
c = np.vstack((top, bottom))

# 画图
plt.imshow(c, cmap='gray', vmin=0, vmax=3)
plt.colorbar()
plt.title("Matrix C Visualization")
plt.show()
