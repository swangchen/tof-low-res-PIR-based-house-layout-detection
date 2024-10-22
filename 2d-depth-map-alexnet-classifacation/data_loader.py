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


def get_pointnet_matrices_with_deviceid(data, device_id):
    depth_matrices = {}
    for j, entry in enumerate(data):
        if entry['device'] == device_id:
            distance = entry['distance_mm']
            depth_matrix, _ = cuatro_tof_construction(device_id, distance)
            depth_matrices[f'Device {device_id} (Frame {j})'] = depth_matrix
    return depth_matrices


def get_single_device_z(path, device_id):
    data = np.load(path, allow_pickle=True)['data']
    processed_data = [raw_depth_processing(d) for d in data]
    depth_matrices = get_pointnet_matrices_with_deviceid(processed_data, device_id)

    # 提取 z 轴数据，转换为 8x8 单通道矩阵
    z_matrices = {}
    for key, matrix in depth_matrices.items():
        reshaped_matrix = matrix.reshape(8, 8, 3)[:, :, 2]  # 提取 z 轴 (即第三个维度)
        z_matrices[key] = reshaped_matrix
        # print(f"{key} Reshaped Z-axis Depth Matrix:\n{reshaped_matrix}\n")

    return list(z_matrices.values())


# 加载标签文件并解析为字典
def load_labels_from_file(label_file_path):
    labels_dict = {}
    with open(label_file_path, 'r') as f:
        content = f.read()
        matches = re.findall(r"(L\d+_device\d+)=\s*(\[\[.*?\]\])", content, re.DOTALL)
        for label_name, matrix_str in matches:
            matrix_str = matrix_str.replace('\n', '').replace(' ', '')
            matrix = np.array(eval(matrix_str))  # 转换为 8x8 矩阵
            labels_dict[label_name] = matrix
    return labels_dict


# 加载并拼接每个设备的数据
def load_concat_data(file_device_mapping):
    all_data = []
    for file_info in file_device_mapping:
        path = file_info['path']
        devices = file_info['devices']

        # 加载并拼接每个设备的数据
        device_matrices = []
        min_length = float('inf')
        for device_id in devices:
            device_data = get_single_device_z(path, device_id=device_id)
            device_matrices.append(device_data)
            min_length = min(min_length, len(device_data))

        for i in range(min_length):
            concatenated_matrix = np.block([
                [device_matrices[0][i], device_matrices[1][i]],
                [device_matrices[2][i], device_matrices[3][i]]
            ])
            all_data.append(concatenated_matrix)
    return np.array(all_data)


def load_data_with_labels(file_device_mapping, labels_dict):
    all_data = []
    all_labels = []

    for file_info in file_device_mapping:
        path = file_info['path']
        version = file_info['version']
        devices = file_info['devices']

        # 获取当前路径的数据帧数
        device_matrices = []
        min_length = float('inf')

        # 每个 device 的实际 frame 数量
        for device_id in devices:
            device_data = get_single_device_z(path, device_id=device_id)
            device_matrices.append(device_data)
            min_length = min(min_length, len(device_data))  # 取得每个设备的最小帧数以同步

        # 生成拼接后的数据帧
        for i in range(min_length):
            combined_data = np.block([
                [device_matrices[0][i], device_matrices[1][i]],
                [device_matrices[2][i], device_matrices[3][i]]
            ])
            all_data.append(combined_data)

        # 生成相应的标签帧
        label_matrix_list = []
        for device_id in devices:
            label_key = f"{version}_device{device_id}"
            label_matrix = labels_dict.get(label_key, np.zeros((8, 8), dtype=int))
            label_matrix_list.append(label_matrix)

        combined_label = np.block([
            [label_matrix_list[0], label_matrix_list[1]],
            [label_matrix_list[2], label_matrix_list[3]]
        ])

        # 根据当前 path 的实际帧数生成相应数量的标签
        all_labels.extend([combined_label] * min_length)

    # 最后检查数据和标签长度是否一致
    assert len(all_data) == len(all_labels), f"Data and labels length mismatch: {len(all_data)} vs {len(all_labels)}"
    print(f"Data length: {len(all_data)}, Labels length: {len(all_labels)}")

    return np.array(all_data), np.array(all_labels)



# 定义数据集类
class DepthDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # 增加单通道维度 (N, 1, 16, 16)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 封装的数据加载函数，返回可用于训练和测试的数据集
def data_set(file_device_mapping, label_file):
    labels_dict = load_labels_from_file(label_file)
    device_data, labels = load_data_with_labels(file_device_mapping, labels_dict)
    dataset = DepthDataset(device_data, labels)
    return dataset


# 保存数据集到文件
def save_dataset(dataset, file_path='dataset.pt'):
    torch.save(dataset, file_path)
    print(f"Dataset saved to {file_path}")


# 加载数据集从文件
def load_washed_dataset(file_path='dataset.pt'):
    dataset = torch.load(file_path)
    print(f"Dataset loaded from {file_path}")
    return dataset


def print_labels(file_device_mapping, labels_dict):
    _, all_labels = load_data_with_labels(file_device_mapping, labels_dict)
    for i, label in enumerate(all_labels):
        print(f"\nLabel Matrix {i + 1}:\n{label}")


# 初始化数据集
def wash_and_save():
    file_device_mapping = [
        {'path': 'datas/20240703_140339_tof_rawdata.npz', 'version': 'L1', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_170056_tof_rawdata.npz', 'version': 'L2', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_162917_tof_rawdata.npz', 'version': 'L3', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_142922_tof_rawdata.npz', 'version': 'L4', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_105818_tof_rawdata.npz', 'version': 'L5', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_130638_tof_rawdata.npz', 'version': 'L6', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_103005_tof_rawdata.npz', 'version': 'L8', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_164136_tof_rawdata.npz', 'version': 'L9', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_164531_tof_rawdata.npz', 'version': 'L10', 'devices': [1, 2, 3, 4]},
    ]
    label_file = "label.txt"
    dataset = data_set(file_device_mapping, label_file)
    save_dataset(dataset, 'washed_labeled_data.pt')


def print_pinned_labels():
    file_device_mapping = [
        {'path': 'datas/20240703_140339_tof_rawdata.npz', 'version': 'L1', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_170056_tof_rawdata.npz', 'version': 'L2', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_162917_tof_rawdata.npz', 'version': 'L3', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240703_142922_tof_rawdata.npz', 'version': 'L4', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_105818_tof_rawdata.npz', 'version': 'L5', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_103005_tof_rawdata.npz', 'version': 'L8', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_164136_tof_rawdata.npz', 'version': 'L9', 'devices': [1, 2, 3, 4]},
        {'path': 'datas/20240704_164531_tof_rawdata.npz', 'version': 'L10', 'devices': [1, 2, 3, 4]},
    ]
    label_file = "label.txt"
    labels_dict = load_labels_from_file(label_file)

    for file_info in file_device_mapping:
        version = file_info['version']
        label_matrix_list = []

        for device_id in file_info['devices']:
            label_key = f"{version}_device{device_id}"
            label_matrix = labels_dict.get(label_key, np.zeros((8, 8), dtype=int))
            label_matrix_list.append(label_matrix)  # 保持8x8，不扩展

        combined_label = np.block([
            [label_matrix_list[0], label_matrix_list[1]],
            [label_matrix_list[2], label_matrix_list[3]]
        ])
        print(f"\nLabel Matrix for {file_info['path']}:\n{combined_label}")
        print(f"\nLabel Matrix for {file_info['path']}:\n{combined_label.shape}")


# print_pinned_labels()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, help="Name of the function to execute")
    args = parser.parse_args()

    # 检查并执行指定的函数
    if args.function == "wash_and_save":
        wash_and_save()
    elif args.function == "print_pinned_labels":
        wash_and_save()
    else:
        print("No valid function specified.")
