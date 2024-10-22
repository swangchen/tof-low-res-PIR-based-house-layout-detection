import bpy
import os
import time
import numpy as np
import mathutils  # 导入 mathutils

# 设置渲染输出路径
output_path = "C:/Users/670329832/Desktop/blender based point cloud detection/point_cloud_{}.exr"


# 相机参数函数
def camara_para(id):
    if id == 1:
        location_x, location_y, location_z = -0.3, 0, 2.7
        fov_l_deg = 22.5
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 38
        Patten_rotate_deg = 50
    elif id == 2:
        location_x, location_y, location_z = -0.3, 0, 2.7
        fov_l_deg = 22.5
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 142
        Patten_rotate_deg = 40
    elif id == 3:
        location_x, location_y, location_z = 0.3, 0, 2.7
        fov_l_deg = 22.5
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 218
        Patten_rotate_deg = 50
    elif id == 4:
        location_x, location_y, location_z = 0.3, 0, 2.7
        fov_l_deg = 22.5
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = -38
        Patten_rotate_deg = 40

    # 计算位置和 FOV
    location = (location_x, location_y, location_z)
    fov = fov_l_deg

    # 计算光轴向量
    normal_vector = np.array([
        np.cos(np.radians(Aiming_tilt_deg)) * np.cos(np.radians(Aiming_rotate_deg)),
        np.cos(np.radians(Aiming_tilt_deg)) * np.sin(np.radians(Aiming_rotate_deg)),
        -np.sin(np.radians(Aiming_tilt_deg))
    ])

    # 计算绕光轴旋转的四元数
    patten_quaternion = mathutils.Quaternion(normal_vector, np.radians(Patten_rotate_deg))
    final_vector = patten_quaternion @ mathutils.Vector(normal_vector)

    return location, final_vector, fov


# 设置渲染配置
def setup_render():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'


# 创建并配置相机
def create_camera(id):
    location, final_vector, fov = camara_para(id)
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = f"CustomCamera_{id}"

    # 设置相机指向
    target_name = f"Target_{id}"
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location + final_vector)
    target = bpy.context.object
    target.name = target_name

    # 添加约束以使相机指向目标
    camera.constraints.new(type='TRACK_TO')
    camera.constraints['Track To'].target = target
    camera.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    camera.constraints['Track To'].up_axis = 'UP_Y'

    # 设置 FOV
    bpy.data.cameras[camera.data.name].lens_unit = 'FOV'
    bpy.data.cameras[camera.data.name].angle = np.radians(fov)
    return camera


# 渲染并保存
def render_with_camera(camera, id):
    bpy.context.scene.camera = camera  # 设置当前摄像机
    filepath = output_path.format(id)  # 设置不同文件名
    bpy.context.scene.render.filepath = filepath

    bpy.ops.render.render(write_still=True)  # 执行渲染

    # 确保文件生成后再加载
    while not os.path.exists(filepath):
        time.sleep(0.1)  # 每 100 毫秒检查一次文件是否存在

    print(f"EXR file for camera {id} successfully saved at {filepath}.")


# 主执行函数
def main():
    setup_render()
    for id in range(1, 5):
        camera = create_camera(id)
        render_with_camera(camera, id)


# 运行
main()
