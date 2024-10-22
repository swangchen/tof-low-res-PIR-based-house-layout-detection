import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import re
from modle import SimplifiedAlexNet
import argparse


def transform(transform_matrix, data):
    return np.matmul(transform_matrix, np.append(data, 1))[0:3]


def reconstruction_with_angles(point_c, x_angle=0, y_angle=0, z_angle=0):
    x_rad = np.pi * x_angle / 180
    y_rad = np.pi * y_angle / 180
    z_rad = np.pi * z_angle / 180
    roll_ang = x_rad  # x
    pitch_ang = y_rad  # y
    yaw_ang = z_rad  # z

    roll_rotation = np.array(
        [[1, 0, 0], [0, np.cos(roll_ang), -np.sin(roll_ang)], [0, np.sin(roll_ang), np.cos(roll_ang)]])
    pitch_rotation = np.array(
        [[np.cos(pitch_ang), 0, np.sin(pitch_ang)], [0, 1, 0], [-np.sin(pitch_ang), 0, np.cos(pitch_ang)]])

    yaw_rotation = np.array(
        [[np.cos(yaw_ang), -np.sin(yaw_ang), 0], [np.sin(yaw_ang), np.cos(yaw_ang), 0], [0, 0, 1]])

    # rotation = pitch_rotation @ roll_rotation
    rotation = pitch_rotation @ roll_rotation @ yaw_rotation

    # translation matrix
    translation = np.array([0, 0, 0])

    euclidean_transform = np.concatenate(
        (np.concatenate((rotation, translation.reshape([-1, 1])), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

    points_in_world = point_c.copy().astype(np.float64)
    for i in range(point_c.shape[0]):
        if np.any(np.isnan(point_c[-1])):
            continue
        points_in_world[i, 0:3] = transform(euclidean_transform, point_c[i, 0:3])

    # res = np.delete(points_in_world, np.isnan(points_in_world[:,0]), axis=0)

    return points_in_world


def create_mask(angle_degree=45, distance_pixel=1000):
    side_length = np.sqrt(2 - 2 * np.cos(np.pi * angle_degree / 180)) * distance_pixel

    # n = side_length / 14
    # pixel_size = n * 2

    pixel_size = side_length / 8
    half_pixel_size = pixel_size / 2

    height = np.sqrt(distance_pixel ** 2 - side_length ** 2 / 2)
    mask = np.arange(8) * 2 - 7
    test = np.zeros((8, 8, 5))

    for s in range(8):
        for t in range(8):
            test[s, t, 0] = mask[s] * half_pixel_size
            test[s, t, 1] = mask[t] * half_pixel_size
            test[s, t, 2] = height
            test[s, t, 3] = np.sqrt(test[s, t, 0] ** 2 + test[s, t, 1] ** 2 + height ** 2)
            test[s, t, 4] = pixel_size
    return test


def convert_from_img_to_camera_pixel(img_u, img_v, height_pixel, mask):
    point = mask[img_u, img_v] * height_pixel / mask[img_u, img_v, 2]
    return point


def convert_from_img_to_camera_matrix(height_m, angle=45, status_m=5 * np.ones([8, 8])):
    point_c = np.empty([0, 5])
    mask = create_mask(angle, 1000)

    for u in range(height_m.shape[0]):
        for v in range(height_m.shape[1]):
            if status_m[u, v] == 5 or status_m[u, v] == 9 or status_m[u, v] == 12:
                point = convert_from_img_to_camera_pixel(u, v, height_m[u, v], mask)
                point_c = np.append(point_c, point.reshape(1, -1), axis=0)
    return point_c


def cuatro_tof_construction(device, distance):
    point_cloud = convert_from_img_to_camera_matrix(distance)
    if device == 1:
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=51 - 90)
        point_cloud = reconstruction_with_angles(point_cloud, y_angle=-32)
        point_cloud[:, 0] = point_cloud[:, 0] + 50
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=0)
    elif device == 2:
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=39)
        point_cloud = reconstruction_with_angles(point_cloud, y_angle=-32)
        point_cloud[:, 0] = point_cloud[:, 0] + 50
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=90)
    elif device == 3:
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=39)
        point_cloud = reconstruction_with_angles(point_cloud, y_angle=-32)
        point_cloud[:, 0] = point_cloud[:, 0] + 50
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=270)
    else:
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=51 - 90)
        point_cloud = reconstruction_with_angles(point_cloud, y_angle=-32)
        point_cloud[:, 0] = point_cloud[:, 0] + 50
        point_cloud = reconstruction_with_angles(point_cloud, z_angle=180)

    return point_cloud[:, 0:3], point_cloud[:, 4]


# 1 Raw data processing
def raw_depth_processing(data):
    S0 = data['target_status'][0]
    S1 = data['target_status'][1]
    S2 = data['target_status'][2]
    R0 = data['reflectance'][0]
    R1 = data['reflectance'][1]
    R2 = data['reflectance'][2]
    filtered_depth = data['distance_mm'][0]
    # print(data['device'])
    for i in range(0, 8):
        for j in range(0, 8):
            if (R0[i][j] <= 5) and (S0[i][j] == 12):
                if (S1[i][j] == 5 or S1[i][j] == 6 or S1[i][j] == 9 or S1[i][j] == 10) and (R1[i][j] > 3):
                    filtered_depth[i][j] = data['distance_mm'][1][i][j]
                else:
                    if (S2[i][j] == 5 or S2[i][j] == 6 or S2[i][j] == 9 or S2[i][j] == 10) and (R2[i][j] > 3):
                        filtered_depth[i][j] = data['distance_mm'][2][i][j]
            if not (S1[i][j] == 5 or S1[i][j] == 6 or S1[i][j] == 9 or S1[i][j] == 10):
                if not (S2[i][j] == 5 or S2[i][j] == 6 or S2[i][j] == 9 or S2[i][j] == 10):
                    if not (S0[i][j] == 5 or S0[i][j] == 6 or S0[i][j] == 9 or S0[i][j] == 10):
                        filtered_depth[i][j] = 4500
    return {'device': data['device'], 'distance_mm': filtered_depth}

def load_labels_from_file(label_file_path):
    labels_dict = {}
    with open(label_file_path, 'r') as f:
        content = f.read()
        matches = re.findall(r"(L\d+_device\d+)=\s*(\[\[.*?\]\])", content, re.DOTALL)

        for label_name, matrix_str in matches:
            matrix_str = matrix_str.replace('\n', '').replace(' ', '')
            # 将字符串转换为 8x8 矩阵
            matrix = np.array(eval(matrix_str))
            labels_dict[label_name] = matrix

    return labels_dict


# 深度矩阵处理函数
def get_depth_matrices(data, device_id):
    depth_matrices = {}
    for j, entry in enumerate(data):
        if entry['device'] == device_id:
            distance = entry['distance_mm']
            depth_matrix, _ = cuatro_tof_construction(device_id, distance)
            depth_matrices[f'Device {device_id} (Frame {j})'] = depth_matrix
    return depth_matrices


def reshaped_matrix(path, device_id):
    data = np.load(path, allow_pickle=True)['data']
    processed_data = [raw_depth_processing(d) for d in data]
    depth_matrices = get_depth_matrices(processed_data, device_id)


    for key, matrix in depth_matrices.items():
        reshaped_matrix = matrix.reshape(8, 8, 3)

    return list(reshaped_matrix)


def print_point_cloud_matrix(data):
    # 定义固定的帧和设备组合
    frame_device_pairs = {
        9: 3,
        10: 2,
        8: 1,
        7: 4
    }

    # 按每个设备逐帧生成点云并打印矩阵
    frame_data = {}
    for j, device in frame_device_pairs.items():
        distance = data[j]['distance_mm']
        # 生成点云
        pc, _ = cuatro_tof_construction(device, distance)
        # 存储每个设备的点云矩阵
        frame_data[device] = pc.reshape(8, 8, 3)  # 将每个设备的点云重塑为8x8x3矩阵

    # 拼接成16x16x3矩阵并打印
    combined_matrix = np.block([
        [frame_data[1], frame_data[2]],
        [frame_data[3], frame_data[4]]
    ])
    print("Combined 16x16x3 point cloud matrix:\n", combined_matrix)





import numpy as np

# 加载数据文件
data = np.load("datas/20240704_130638_tof_rawdata.npz", allow_pickle=True)['data']

# 处理数据
processed_data = [raw_depth_processing(d) for d in data]

# 查看 processed_data 的整体结构
print("processed_data 类型:", type(processed_data))
print("processed_data 的长度:", len(processed_data))

# 检查每帧的结构
for i, frame in enumerate(processed_data):
    print(f"\nFrame {i}:")
    print("Keys:", frame.keys())
    for key, value in frame.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape:", value.shape)
        else:
            print(f"{key}: {value}")



