import bpy
import os
import time
import numpy as np
import mathutils

## 清空场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


# 创建沙发和茶几
def create_sofa():
    # 创建沙发主体
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    sofa_body = bpy.context.object
    sofa_body.name = 'Sofa Body'
    sofa_body.scale = (1.4, 0.5, 0.4)

    # 创建沙发靠背
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.475, 0.225))
    sofa_back = bpy.context.object
    sofa_back.name = 'Sofa Back'
    sofa_back.scale = (2.0, 0.45, 0.85)

    # 创建右侧扶手
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.85, 0, 0.1))
    armrest_right = bpy.context.object
    armrest_right.name = 'Armrest Right'
    armrest_right.scale = (0.3, 0.5, 0.6)

    # 创建左侧扶手
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.85, 0, 0.1))
    armrest_left = bpy.context.object
    armrest_left.name = 'Armrest Left'
    armrest_left.scale = (0.3, 0.5, 0.6)

    # 将沙发的所有部分加入到一个集合
    bpy.ops.object.select_all(action='DESELECT')
    sofa_body.select_set(True)
    sofa_back.select_set(True)
    armrest_right.select_set(True)
    armrest_left.select_set(True)

    bpy.ops.object.join()  # 将所有部分合并成一个对象
    bpy.context.object.name = 'Sofa'

    # 设置组合后沙发位置
    bpy.context.object.location = (-0.5, -0.75, 0.2)


# 运行函数以创建沙发
create_sofa()

import bpy
import os
import time
import numpy as np
import mathutils

# 设置渲染输出路径
output_path = "C:/Users/670329832/Desktop/blender based point cloud detection/point_cloud_{}.exr"


# 相机参数函数
def camara_para(id):
    if id == 1:
        location_x, location_y, location_z = -0.3, 0, 2.7
        fov_l_deg = 45
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 38
        Patten_rotate_deg = 50
    elif id == 2:
        location_x, location_y, location_z = -0.3, 0, 2.7
        fov_l_deg = 45
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 142
        Patten_rotate_deg = 40
    elif id == 3:
        location_x, location_y, location_z = 0.3, 0, 2.7
        fov_l_deg = 45
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = 218
        Patten_rotate_deg = 50
    elif id == 4:
        location_x, location_y, location_z = 0.3, 0, 2.7
        fov_l_deg = 45
        Aiming_tilt_deg = 32
        Aiming_rotate_deg = -38
        Patten_rotate_deg = 40

    location = (location_x, location_y, location_z)
    fov = fov_l_deg

    return location, Aiming_tilt_deg, Aiming_rotate_deg, Patten_rotate_deg, fov


# 设置渲染配置
def setup_render():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'


# 创建并配置相机
def create_camera(id):
    location, Aiming_tilt_deg, Aiming_rotate_deg, Patten_rotate_deg, fov = camara_para(id)

    # 添加相机
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = f"CustomCamera_{id}"

    # 设置俯仰角和方位角
    camera.rotation_euler = (np.radians(Aiming_tilt_deg), 0, np.radians(Aiming_rotate_deg))

    # 旋转视锥体
    camera.rotation_euler.rotate_axis('Z', np.radians(Patten_rotate_deg))

    # 设置 FOV
    bpy.data.cameras[camera.data.name].lens_unit = 'FOV'
    bpy.data.cameras[camera.data.name].angle = np.radians(fov)

    # 标注地面视角投影
    draw_ground_projection(camera, location, fov, Aiming_tilt_deg, Aiming_rotate_deg)
    return camera


# 绘制视角在地面的投影
def draw_ground_projection(camera, location, fov, tilt, rotate):
    # 计算投影边界点
    projection_distance = 10  # 投影距离（可调整）
    half_width = np.tan(np.radians(fov / 2)) * projection_distance

    # 相机的旋转方向
    rotation_matrix = camera.matrix_world.to_3x3()

    # 四个边界点的局部坐标
    boundary_points = [
        mathutils.Vector((half_width, 0, -projection_distance)),
        mathutils.Vector((-half_width, 0, -projection_distance))
    ]

    # 转换为世界坐标并投影到地面
    world_points = [(rotation_matrix @ point) + mathutils.Vector(location) for point in boundary_points]

    # 添加地面投影的边界线
    for i in range(len(world_points) - 1):
        start, end = world_points[i], world_points[i + 1]
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=(start - end).length, location=(start + end) / 2)
        line = bpy.context.object
        line.rotation_euler = (0, 0, np.arctan2(end.y - start.y, end.x - start.x))
        line.name = f"Ground_Projection_Line_{i}"


# 渲染并保存
def render_with_camera(camera, id):
    bpy.context.scene.camera = camera  # 设置当前摄像机
    filepath = output_path.format(id)  # 设置不同文件名
    bpy.context.scene.render.filepath = filepath

    bpy.ops.render.render(write_still=True)  # 执行渲染

    # 确保文件生成后再加载
    while not os.path.exists(filepath):
        time.sleep(0.1)

    print(f"EXR file for camera {id} successfully saved at {filepath}.")


# 主执行函数
def main():
   setup_render()
   for id in range(1, 5):
       camera = create_camera(id)
       render_with_camera(camera, id)

# 运行
main()


for id in range(1, 5):
    camera = create_camera(id)




