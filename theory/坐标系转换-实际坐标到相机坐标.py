import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 参数设置
location_x = -0.3
location_y = 0
height = 2.7
fov_l = 22.5  # 各侧面与法线的夹角（视场）
Aiming_tilt_deg = 32
Aiming_rotate_deg = 38
Patten_rotate_deg = 50

# 计算法线向量
normal_vector = np.array([
    np.cos(np.radians(Aiming_tilt_deg)) * np.cos(np.radians(Aiming_rotate_deg)),
    np.cos(np.radians(Aiming_tilt_deg)) * np.sin(np.radians(Aiming_rotate_deg)),
    -np.sin(np.radians(Aiming_tilt_deg))
])

# 根据视角和高度计算底面的边长
base_size = 2 * height * np.tan(np.radians(fov_l))

# 创建底面顶点相对于锥顶的初始坐标
base_points = np.array([
    [base_size / 2, base_size / 2, 0],
    [-base_size / 2, base_size / 2, 0],
    [-base_size / 2, -base_size / 2, 0],
    [base_size / 2, -base_size / 2, 0]
])

# 锥顶位置
apex_point = np.array([location_x, location_y, height])

# 定义旋转函数
def rotate(point, axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = cos_angle * np.eye(3) + \
                      sin_angle * np.array([[0, -axis[2], axis[1]],
                                            [axis[2], 0, -axis[0]],
                                            [-axis[1], axis[0], 0]]) + \
                      (1 - cos_angle) * np.outer(axis, axis)
    return point @ rotation_matrix.T

# 将底面点旋转至法线方向并平移到锥顶位置
rotated_base_points = np.array([rotate(p, normal_vector, np.radians(Patten_rotate_deg)) for p in base_points]) + apex_point
rotated_apex_point = rotate(apex_point, normal_vector, np.radians(Patten_rotate_deg))

# 计算每条边的向量方程，并找到 z=0 时的交点
intersection_points = []
for point in rotated_base_points:
    direction_vector = point - rotated_apex_point
    t = -rotated_apex_point[2] / direction_vector[2]
    intersect_x = rotated_apex_point[0] + t * direction_vector[0]
    intersect_y = rotated_apex_point[1] + t * direction_vector[1]
    intersection_points.append([intersect_x, intersect_y, 0])

# 绘制正四棱锥和交面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

# 绘制底面
base_faces = [rotated_base_points]
ax.add_collection3d(Poly3DCollection(base_faces, color='lightgray', edgecolor='black', linewidth=1, alpha=0.5))

# 绘制从顶点到底面顶点的边
for point in rotated_base_points:
    ax.plot([rotated_apex_point[0], point[0]],
            [rotated_apex_point[1], point[1]],
            [rotated_apex_point[2], point[2]], 'k-')

# 绘制 z = 0 平面的交线
intersection_points = np.array(intersection_points)
for i in range(len(intersection_points)):
    start_point = intersection_points[i]
    end_point = intersection_points[(i + 1) % len(intersection_points)]
    ax.plot([start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]], 'r--')

# 输出底面与 z=0 的交点
print("底面与 z=0 的交点坐标:")
for i, point in enumerate(intersection_points):
    print(f"交点 {i+1}: {point}")

# 设置坐标轴范围和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([location_x - base_size, location_x + base_size])
ax.set_ylim([location_y - base_size, location_y + base_size])
ax.set_zlim([0, height])

plt.show()
