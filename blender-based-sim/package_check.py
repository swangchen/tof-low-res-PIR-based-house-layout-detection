import bpy
import numpy as np


## 清空场景
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False)

# 创建沙发和茶几
def create_sofa():
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -2, 0.5))
    sofa_body = bpy.context.object
    sofa_body.name = 'Sofa Body'
    sofa_body.scale = (1.5, 0.5, 0.5)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.5, 1.25))
    sofa_back = bpy.context.object
    sofa_back.name = 'Sofa Back'
    sofa_back.scale = (1.5, 0.1, 0.5)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(1.6, -2, 0.75))
    armrest_right = bpy.context.object
    armrest_right.name = 'Armrest Right'
    armrest_right.scale = (0.1, 0.5, 0.25)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.6, -2, 0.75))
    armrest_left = bpy.context.object
    armrest_left.name = 'Armrest Left'
    armrest_left.scale = (0.1, 0.5, 0.25)


def create_table():
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 1, 0.4))
    table_top = bpy.context.object
    table_top.name = 'Table Top'
    table_top.scale = (0.7, 0.5, 0.05)
    for x, y in [(-0.6, 0.8), (0.6, 0.8), (-0.6, 1.2), (0.6, 1.2)]:
        bpy.ops.mesh.primitive_cube_add(size=0.1, location=(x, y, 0.2))
        leg = bpy.context.object
        leg.name = 'Table Leg'
        leg.scale = (0.05, 0.05, 0.4)


## 创建场景中的沙发和茶几
# create_sofa()
# create_table()


# 参数
location_x = -0.3
location_y = 0
height = 2.7
fov_l_deg = 22.5
Aiming_tilt_deg = 32
Aiming_rotate_deg = 38
Patten_rotate_deg = 40

# 计算视场角（FOV）
fov_rad = np.radians(fov_l_deg)

# 计算摄像头位置
location = (location_x, location_y, height)

# 计算旋转角度
rotation_z = np.radians(Patten_rotate_deg)
rotation_y = np.radians(-Aiming_tilt_deg)
rotation_x = np.radians(Aiming_rotate_deg)

rotation = (rotation_x, rotation_y, rotation_z)


# 设置摄像头
def setup_camera(location, rotation, fov):
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.name = "CustomCamera"
    bpy.data.cameras[camera.data.name].lens_unit = 'FOV'
    bpy.data.cameras[camera.data.name].angle = np.radians(fov)
    return camera


# 根据参数设置摄像头
camera = setup_camera(location=location, rotation=rotation, fov=fov_l_deg)



# 直接设置 3D 环境中的物体和传感器位置

# 生成点云，直接读取深度图中的值作为点云数据
def generate_8x8_point_cloud(camera):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.camera = camera
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.filepath = "C:/Users/670329832/Desktop/point_cloud.exr"
    bpy.ops.render.render(write_still=True)
    depth_data = bpy.data.images.load("C:/Users/670329832/Desktop/point_cloud.exr").pixels[:]

    width, height = bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y
    step_x, step_y = width // 8, height // 8  # 每个区域的像素步长

    point_cloud = []
    zone_number = 1
    for i in range(8):
        for j in range(8):
            # 每个区域的最小深度值，直接作为点云 Z 值
            region_depth = min([
                depth_data[(y * width + x) * 4] for y in range(i * step_y, (i + 1) * step_y)
                for x in range(j * step_x, (j + 1) * step_x)
            ])
            point_cloud.append((i, j, region_depth, f"Zone{zone_number}"))
            zone_number += 1

    return point_cloud
