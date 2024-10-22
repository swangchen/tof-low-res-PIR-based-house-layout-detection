import OpenEXR
import Imath
import numpy as np


def load_exr_to_point_cloud(file_path):
    exr_file = OpenEXR.InputFile(file_path)

    # 获取并打印所有通道信息
    header = exr_file.header()
    print("Channels available in EXR file:", list(header['channels'].keys()))

    # 选择一个有效的通道 (这里假设使用 'R' 通道)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    try:
        z_data = exr_file.channel('R', pt)  # 使用 'R' 代替 'Z'
    except KeyError:
        print("No valid depth channel found in EXR file.")
        return None

    # 转换数据格式
    dw = header['dataWindow']
    width, height = (dw.max.x - dw.min.x + 1), (dw.max.y - dw.min.y + 1)
    z = np.frombuffer(z_data, dtype=np.float32).reshape(height, width)

    # 转换为点云格式
    point_cloud = np.array([[x, y, z[y, x]] for y in range(height) for x in range(width)])
    point_cloud = point_cloud[~np.isnan(point_cloud[:, 2])]  # 删除无效点
    return point_cloud


# 加载 EXR 文件
file_path = "L2_point_cloud_1.exr"
point_cloud = load_exr_to_point_cloud(file_path)
if point_cloud is not None:
    print("Point cloud shape:", point_cloud.shape)
