import numpy as np
import scipy


def transform(transform_matrix, data):
    return np.matmul(transform_matrix, np.append(data, 1))[0:3]


'''
Input:
    point_c: array [K,4]         K points in 3d world (K = 64)
    height:  int                 the height of the camera in the ceiling
    angle:   int                 the inclination angle of the camera around x axis (angle not rads)
Output:
    point_in_world [K,4]         the calibrated K points in 3d world (K = 64)

Considering the position and angle of the camera, rotate and transform the point clouds. 64 to 64
According to the left-hand rule, rotate first around the z-axis, then x-axis, y-axis
'''


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


'''
Input:
    point_c:         array [K, 3/4]     the ground in camera coordinate
    normal_vector:   array [3,]         the normal vector of the ground in camera coordinate
Output:
    points_in_world: array [K, 3/4]     the ground in world coordinate
Considering point_c can fit the ground whose normal vector is normal_vector in camera coordinate, the normal vector of
the ground in world coordinate should be [0,0,1]. Rotate the plane whose normal vector is normal_vector to a plane whose
normal plane is [0,0,1]
'''


def reconstruction_ground_with_normal_vector(point_c, normal_vector):
    ground_vector = np.array([0,0,1])
    rotate_axis = np.array([ground_vector[1]*normal_vector[2]-ground_vector[2]*normal_vector[1],
                            ground_vector[2]*normal_vector[0]-ground_vector[0]*normal_vector[2],
                            ground_vector[0]*normal_vector[1]-ground_vector[1]*normal_vector[0]])
    rotate_axis = rotate_axis/np.linalg.norm(rotate_axis)

    angle = np.degrees(np.arccos( (ground_vector@normal_vector) / (np.sqrt(np.sum(np.square(ground_vector))) * np.sqrt(np.sum(np.square(normal_vector)))) ))
    angle = np.pi * angle / 180
    rotate_axis_anti = np.array([[0, -rotate_axis[2], rotate_axis[1]],
                                 [rotate_axis[2], 0, -rotate_axis[0]],
                                 [-rotate_axis[1], rotate_axis[0], 0]])

    rotation = np.eye(3) + rotate_axis_anti*np.sin(angle) + rotate_axis_anti@rotate_axis_anti*(1-np.cos(angle))

    # translation matrix
    translation = np.array([0, 0, 0])

    euclidean_transform = np.concatenate(
        (np.concatenate((rotation, translation.reshape([-1, 1])), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

    points_in_world = point_c.copy()
    for i in range(point_c.shape[0]):
        points_in_world[i, 0:3] = transform(euclidean_transform, point_c[i, 0:3])

    return points_in_world


'''
Input:
    height_m: array [8, 8]      8 * 8 distance from target POINT to camera PLANE
    status_m: array [8, 8]      8 * 8 status
    camera_center: array ?
Output:
    point_c: array [number, 8, 8, 5]       point clouds * [x, y, height=z, Point2Point distance]

Convert data from image coordinate to camera coordinate
Origin of camera coordinate is at center of image coordinate
Z-positive direction is the same as the camera
'''


def convert_from_img_to_camera_matrix(height_m, angle=45, status_m=5 * np.ones([8, 8])):
    point_c = np.empty([0, 5])
    mask = create_mask(angle, 1000)

    for u in range(height_m.shape[0]):
        for v in range(height_m.shape[1]):
            if status_m[u, v] == 5 or status_m[u, v] == 9 or status_m[u, v] == 12:
                point = convert_from_img_to_camera_pixel(u, v, height_m[u, v], mask)
                point_c = np.append(point_c, point.reshape(1, -1), axis=0)
    return point_c


'''
Input:
    img_u, img_v: int, int      x and y index, [0:8]
    height_pixel: float         z = height, distance from target POINT to camera PLANE
    mask: array [8, 8, 5]       [x, y, height=z, Point2Point distance, pixel size]
Output:
    point: array [8, 8, 5]      [x, y, height=z, Point2Point distance, pixel size]

camera_pixel is [0, 0, 0], sensor is vertical to the center of  8 * 8 plane
point [x, y, height, _], at the z-positive direction of the camera
'''


def convert_from_img_to_camera_pixel(img_u, img_v, height_pixel, mask):
    point = mask[img_u, img_v] * height_pixel / mask[img_u, img_v, 2]
    return point


'''
Input: 
    distance_pixel: int or float    distance from target POINT to camera POINT
    angle_degree: int               vertical/horizontal detection volume(degree) of the ToF sensor, default is 45°
Output:
    mask: array [8, 8, 5]           [x, y, height=z, Point2Point distance, pixel_size]

the 8 * 8 point cloud at the same plane ( height is the same )
distance_pixel is the distance from corner point to the sensor (0,0 / 0,7 / 7,0 / 7,7)
distance_pixel is point to point
'''


def create_mask(angle_degree=45, distance_pixel=1000):
    side_length = np.sqrt(2 - 2*np.cos(np.pi*angle_degree/180)) * distance_pixel

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


'''
Input:
    pt: array [64, 4]       an array (64 * 4) or (64 * 3)
Output:
    params: array [3,]      the coefficient of the plane

fit the 64 points into a plane ax + by + c = z with the least squares
the loss function is sigma((ax+by+x-z)^2)
'''


def calculate_lsq(pt):
    def loss_func(params):
        return params[0] * pt[:, 0] + params[1] * pt[:, 1] + params[2] - pt[:, 2]

    params = np.random.random(3)
    params = scipy.optimize.leastsq(loss_func, params)[0]

    return params


def sq_loss(pt, params):
    pt = pt.reshape((1, -1))
    return np.sum(np.square(params[0] * pt[:, 0] + params[1] * pt[:, 1] + params[2] - pt[:, 2])) / pt.shape[0]


'''
Fit a plane by the point cloud with noises (outliers) using RANSAC
'''


def ransac(point_cloud, init_size, err_threshold, max_inliers, max_iteration):
    epoch = 0

    best_model = None
    best_point_cloud = None
    best_err = np.inf

    while epoch < max_iteration:
        temp_index = np.random.permutation(point_cloud.shape[0])
        temp_inliers = point_cloud[temp_index[0:init_size]]
        temp_outliers = point_cloud[temp_index[init_size::]]
        temp_model = calculate_lsq(temp_inliers)

        for pt in temp_outliers:
            if sq_loss(pt, temp_model) < err_threshold:
                temp_inliers = np.append(temp_inliers, pt.reshape((1, -1)), axis=0)

        if temp_inliers.shape[0] > max_inliers:
            current_model = calculate_lsq(temp_inliers)
            current_err = sq_loss(temp_inliers, current_model)

            if current_err < best_err:
                best_model = current_model
                best_err = current_err
                best_point_cloud = temp_inliers

        epoch += 1

    return best_model, best_point_cloud, best_err


def triple_tof_construction(device, distance):
    point_cloud = convert_from_img_to_camera_matrix(distance)
    point_cloud = reconstruction_with_angles(point_cloud, z_angle=45)
    point_cloud = reconstruction_with_angles(point_cloud, y_angle=-22.5)
    point_cloud[:,0] = point_cloud[:,0] + 40
    point_cloud = reconstruction_with_angles(point_cloud, z_angle=-device*120+60)
    return point_cloud[:,0:3]


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

    # point_cloud = convert_from_img_to_camera_matrix(distance)
    # point_cloud = reconstruction_with_angles(point_cloud, z_angle=45)
    # point_cloud = reconstruction_with_angles(point_cloud, y_angle=-33)
    # point_cloud[:, 0] = point_cloud[:, 0] + 50
    # if device == 1:
    #     point_cloud = reconstruction_with_angles(point_cloud, z_angle=0)
    # elif device == 2:
    #     point_cloud = reconstruction_with_angles(point_cloud, z_angle=90)
    # elif device == 4:
    #     point_cloud = reconstruction_with_angles(point_cloud, z_angle=180)
    # else:
    #     point_cloud = reconstruction_with_angles(point_cloud, z_angle=270)
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
    for i in range(0,8):
        for j in range(0,8):
            if (R0[i][j] <= 5) and (S0[i][j] == 12):
                if (S1[i][j] == 5 or S1[i][j] == 6 or S1[i][j] == 9 or S1[i][j] == 10) and (R1[i][j]>3):
                    filtered_depth[i][j] = data['distance_mm'][1][i][j]
                else:
                    if (S2[i][j] == 5 or S2[i][j] == 6 or S2[i][j] == 9 or S2[i][j] == 10) and (R2[i][j]>3):
                        filtered_depth[i][j] = data['distance_mm'][2][i][j]
            if  not (S1[i][j] == 5 or S1[i][j] == 6 or S1[i][j] == 9 or S1[i][j] == 10):
                if  not (S2[i][j] == 5 or S2[i][j] == 6 or S2[i][j] == 9 or S2[i][j] == 10):
                    if  not (S0[i][j] == 5 or S0[i][j] == 6 or S0[i][j] == 9 or S0[i][j] == 10):
                        filtered_depth[i][j] = 4500
    return {'device':data['device'], 'distance_mm':filtered_depth}


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import re


# 读取标签文件并解析为字典
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


# 加载数据并提取 z 轴信息
def data_loader(path, device_id):
    data = np.load(path, allow_pickle=True)['data']
    processed_data = [raw_depth_processing(d) for d in data]
    depth_matrices = get_depth_matrices(processed_data, device_id)

    # 提取 z 轴数据，转换为 8x8 单通道矩阵
    z_matrices = {}
    for key, matrix in depth_matrices.items():
        reshaped_matrix = matrix.reshape(8, 8, 3)[:, :, 2]  # 提取 z 轴 (即第三个维度)
        z_matrices[key] = reshaped_matrix
        # print(f"{key} Reshaped Z-axis Depth Matrix:\n{reshaped_matrix.shape}\n")

    return list(z_matrices.values())


def iou_loss(preds, targets, num_classes=5):
    iou = 0.0
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).float().sum()  # 交集
        union = (pred_inds | target_inds).float().sum()  # 并集

        if union == 0:
            iou += 1.0  # 防止空集情况下的除0错误
        else:
            iou += intersection / union
    return 1 - (iou / num_classes)  # 返回1减去平均IoU作为损失
# 自定义数据集
class DepthDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # 增加单通道维度 (C, H, W)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据和标签加载封装函数
def load_data_with_labels(file_device_mapping, labels_dict):
    all_data, all_labels = [], []

    for file_info in file_device_mapping:
        path = file_info['path']
        version = file_info['version']
        devices = file_info['devices']

        # 加载每个版本的4个设备的数据
        device_matrices = []
        min_length = float('inf')  # 记录每个设备的最小帧数
        for device_id in devices:
            device_data = data_loader(path, device_id=device_id)
            device_matrices.append(device_data)
            min_length = min(min_length, len(device_data))  # 更新最小长度

        # 检查是否有设备数据为空或长度不足
        if min_length == 0:
            print(f"Skipping {version} due to missing data in one of the devices.")
            continue  # 跳过没有数据的版本

        # 拼接每四个 8x8 单通道矩阵成 16x16 单通道矩阵
        for i in range(min_length):
            concatenated_matrix = np.block([
                [device_matrices[0][i], device_matrices[1][i]],
                [device_matrices[2][i], device_matrices[3][i]]
            ])
            all_data.append(concatenated_matrix)

            # 获取标签
            label_key = f"{version}_device{devices[0]}"  # 假设所有设备标签相同
            label_matrix = labels_dict.get(label_key, np.zeros((8, 8), dtype=int))
            full_label = np.repeat(np.repeat(label_matrix, 2, axis=0), 2, axis=1)  # 扩展到16x16
            all_labels.append(full_label)

    return np.array(all_data), np.array(all_labels)


# 读取标签文件
labels_dict = load_labels_from_file('label.txt')

# 初始化数据集
file_device_mapping = [
    {'path': '20240703_140339_tof_rawdata.npz', 'version': 'L1', 'devices': [1, 2, 3, 4]},
    {'path': '20240703_170056_tof_rawdata.npz', 'version': 'L2', 'devices': [1, 2, 3, 4]},
    {'path': '20240703_162917_tof_rawdata.npz', 'version': 'L3', 'devices': [1, 2, 3, 4]},
    {'path': '20240703_142922_tof_rawdata.npz', 'version': 'L4', 'devices': [1, 2, 3, 4]},
    {'path': '20240704_105818_tof_rawdata.npz', 'version': 'L5', 'devices': [1, 2, 3, 4]},
    {'path': '20240704_130638_tof_rawdata.npz', 'version': 'L6', 'devices': [1, 2, 3, 4]},
    {'path': '20240704_103005_tof_rawdata.npz', 'version': 'L8', 'devices': [1, 2, 3, 4]},
    {'path': '20240704_164136_tof_rawdata.npz', 'version': 'L9', 'devices': [1, 2, 3, 4]},
    {'path': '20240704_164531_tof_rawdata.npz', 'version': 'L10', 'devices': [1, 2, 3, 4]},
]

device_data, labels = load_data_with_labels(file_device_mapping, labels_dict)
dataset = DepthDataset(device_data, labels)

# 数据集分割
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 修改 SimplifiedAlexNet 模型以适应单通道输入
class SimplifiedAlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(SimplifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 初始化模型、损失函数和优化器
model = SimplifiedAlexNet(num_classes=5)
criterion = nn.CrossEntropyLoss()  # 使用 criterion 作为交叉熵损失的变量名
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 训练模型
epochs = 8
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算交叉熵损失
        ce_loss = criterion(outputs, targets)  # 使用 criterion 而非 cross_entropy_loss

        # 计算IoU损失
        _, preds = torch.max(outputs, 1)
        iou_loss_value = iou_loss(preds, targets)

        # 总损失 = 交叉熵损失 + IoU损失
        total_loss = ce_loss + iou_loss_value

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

# 保存模型
model_path = "simplified_alexnet_model_with_iou.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 从测试集中挑选一个样本进行预测
sample_data, sample_label = test_dataset[0]
sample_data = sample_data.unsqueeze(0)

# 预测
model.eval()
with torch.no_grad():
    output = model(sample_data)
    _, predicted = torch.max(output, 1)

# 输出预测结果和真实标签
print("Predicted Label:")
print(predicted.squeeze().numpy())  # 输出 16x16 的预测矩阵
print("Actual Label:")
print(sample_label.numpy())  # 输出 16x16 的真实标签矩阵

# 生成并可视化混淆矩阵和计算精确度
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

# 混淆矩阵可视化
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 输出分类报告，包括精确度、召回率和 F1 分数
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))