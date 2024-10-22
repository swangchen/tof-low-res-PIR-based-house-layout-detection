import OpenEXR
import Imath
import numpy as np


def read_exr(image_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(image_path)

    # 检查 EXR 文件中的通道名称
    print("Available channels:", exr_file.header()['channels'].keys())

    # 获取文件的头信息以读取尺寸
    header = exr_file.header()
    width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1

    # 尝试使用名为 'R' 的通道代替 'Z' （请替换为您在 Available channels 中看到的深度通道名称）
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', float_type)  # 将 'R' 替换为实际的深度通道名称

    # 将读取的深度数据转换为 NumPy 数组
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = np.reshape(depth, (height, width))  # 重新整形为图像尺寸

    return depth


# 使用示例
image_path = "point_cloud.exr"
depth_data = read_exr(image_path)

# 输出数组的形状和部分内容
print("Depth data shape:", depth_data.shape)
print("Sample depth values:", depth_data[0:5, 0:5])  # 打印前5行5列的值
