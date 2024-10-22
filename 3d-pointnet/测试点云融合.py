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

import numpy as np

# 加载并拼接每个设备的点云数据
def load_single_frame_point_cloud(file_info, frame_index=0):
    path = file_info['path']
    devices = file_info['devices']

    # 存储每个设备的点云数据
    device_point_clouds = []

    # 加载 .npz 文件
    data = np.load(path, allow_pickle=True)['data']  # 确保 .npz 文件包含 "data" 键

    for device_id in devices:
        # 提取指定帧的 'distance_mm' 数据
        distance = data[frame_index]['distance_mm']  # 假设 data 是按帧和设备结构化的数组
        point_cloud, _ = cuatro_tof_construction(device_id, distance)
        device_point_clouds.append(point_cloud.reshape(8, 8, 3))  # 重塑为 8x8x3

    # 按 [device1, device2], [device3, device4] 的布局拼接成 16x16x3
    combined_point_cloud = np.block([
        [device_point_clouds[0], device_point_clouds[1]],
        [device_point_clouds[2], device_point_clouds[3]]
    ])

    return combined_point_cloud  # 返回单帧的 16x16x3 点云

# 示例调用
file_info = {'path': 'datas/20240703_140339_tof_rawdata.npz', 'devices': [1, 2, 3, 4]}
combined_point_cloud = load_single_frame_point_cloud(file_info, frame_index=0)
print("Combined point cloud shape:", combined_point_cloud.shape)  # 输出为 (16, 16, 3)
