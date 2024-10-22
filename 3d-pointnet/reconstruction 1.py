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


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog
import tkinter as tk

# 1 Raw data processing
def raw_depth_processing(data):
    S0 = data['target_status'][0]
    S1 = data['target_status'][1]
    S2 = data['target_status'][2]
    R0 = data['reflectance'][0]
    R1 = data['reflectance'][1]
    R2 = data['reflectance'][2]
    filtered_depth = data['distance_mm'][0]
    print(data['device'])
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

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

def visualize_point_cloud(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 定义固定的帧和设备
    frame_device_pairs = {
        9: 3,
        10: 2,
        8: 1,
        7: 4
    }

    colors = ['r', 'g', 'b', 'y']  # 不同设备使用不同颜色
    for idx, (j, device) in enumerate(frame_device_pairs.items()):
        distance = data[j]['distance_mm']

        # 生成点云
        pc, _ = cuatro_tof_construction(device, distance)  # 只提取第一个数组

        # 绘制点云
        ax.scatter(pc[:, 0], pc[:, 1], 2800-pc[:, 2], color=colors[idx], label=f'Device {device} (Frame {j})')
    ax.set_zlim(0, 2800)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    fnames = filedialog.askopenfilenames(initialdir=".", title="Select file",
                                         filetypes=(('npz', '*.npz'), ("all files", "*.*")))
    if fnames:
        print(fnames)
        data = np.load(fnames[0], allow_pickle=True)['data']

        # 清洗数据
        processed_data = [raw_depth_processing(d) for d in data]

        # 使用第一、二、三、四个设备的点云
        visualize_point_cloud(processed_data)

    else:
        print("No file selected.")









# def visualize_point_cloud(data):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#
#     j = 7  # 选择第 3 帧数据
#     device = data[j]['device']
#     distance = data[j]['distance_mm'][0]
#
#     # 生成点云
#     pc, _ = cuatro_tof_construction(device, distance)  # 只提取第一个数组
#
#     # 绘制点云
#     ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],  label=f'Device {device}')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.show()
#
# if __name__ == '__main__':
#     root = tk.Tk()
#     root.withdraw()
#     fnames = filedialog.askopenfilenames(initialdir=".", title="Select file",
#                                          filetypes=(('npz', '*.npz'), ("all files", "*.*")))
#     if fnames:
#         print(fnames)
#         data = np.load(fnames[0], allow_pickle=True)['data']
#         visualize_point_cloud(data)
#     else:
#         print("No file selected.")
