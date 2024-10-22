import numpy as np
import matplotlib.pyplot as plt

# Parameters
location_x = -0.3
location_y = 0
height = 2.7
fov_l_deg = 22.5
Aiming_tilt_deg = 32
Aiming_rotate_deg = 38
Patten_rotate_deg = 40

# Convert field of view to radians
fov_rad = np.radians(fov_l_deg)

# Calculate normal vector based on aiming angles
normal_vector = np.array([
    np.cos(np.radians(Aiming_tilt_deg)) * np.cos(np.radians(Aiming_rotate_deg)),
    np.cos(np.radians(Aiming_tilt_deg)) * np.sin(np.radians(Aiming_rotate_deg)),
    -np.sin(np.radians(Aiming_tilt_deg))
])

# Initial vertex vectors (before rotation)
v = 1
l1_o = np.array([v * np.tan(fov_rad), v * np.tan(fov_rad), -np.sqrt(2) * v * np.tan(fov_rad)])
l2_o = np.array([-v * np.tan(fov_rad), v * np.tan(fov_rad), -np.sqrt(2) * v * np.tan(fov_rad)])
l3_o = np.array([v * np.tan(fov_rad), -v * np.tan(fov_rad), -np.sqrt(2) * v * np.tan(fov_rad)])
l4_o = np.array([-v * np.tan(fov_rad), -v * np.tan(fov_rad), -np.sqrt(2) * v * np.tan(fov_rad)])

# Normalize the rotation axis
axis = normal_vector / np.linalg.norm(normal_vector)
theta = np.radians(Patten_rotate_deg)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
one_minus_cos = 1 - cos_theta

# Rotation matrix based on Rodrigues' rotation formula
R = np.array([
    [cos_theta + axis[0]**2 * one_minus_cos, axis[0] * axis[1] * one_minus_cos - axis[2] * sin_theta, axis[0] * axis[2] * one_minus_cos + axis[1] * sin_theta],
    [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_theta, cos_theta + axis[1]**2 * one_minus_cos, axis[1] * axis[2] * one_minus_cos - axis[0] * sin_theta],
    [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_theta, axis[2] * axis[1] * one_minus_cos + axis[0] * sin_theta, cos_theta + axis[2]**2 * one_minus_cos]
])

# Apply rotation matrix R to obtain new rotated vertices
l1_rotated = R @ l1_o
l2_rotated = R @ l2_o
l3_rotated = R @ l3_o
l4_rotated = R @ l4_o

# Calculate the intersection of the rotated vertices with the z=0 plane
def intersection_with_z0(vertex, location_x, location_y):
    if vertex[2] != 0:
        scale = -height / vertex[2]  # Factor to scale vertex to z=0
        return vertex[0] * scale + location_x, vertex[1] * scale + location_y, 0
    else:
        return None  # Point already lies on z=0

# Calculate intersections for each vertex
l1_intersection = intersection_with_z0(l1_rotated, location_x, location_y)
l2_intersection = intersection_with_z0(l2_rotated, location_x, location_y)
l3_intersection = intersection_with_z0(l3_rotated, location_x, location_y)
l4_intersection = intersection_with_z0(l4_rotated, location_x, location_y)

# Print results
print("Intersection points at z=0:")
print("L1:", l1_intersection)
print("L2:", l2_intersection)
print("L3:", l3_intersection)
print("L4:", l4_intersection)

# Plot the points
intersections = [l1_intersection, l2_intersection, l3_intersection, l4_intersection]
x_coords = [point[0] for point in intersections if point is not None]
y_coords = [point[1] for point in intersections if point is not None]

plt.figure(figsize=(6, 6))
plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], marker='o')  # Closing the loop
plt.scatter(x_coords, y_coords, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Intersection Points at z=0 Plane")
plt.grid(True)
plt.show()
