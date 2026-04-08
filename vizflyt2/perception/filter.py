from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d

# --- Load original PLY with all properties ---
ply = PlyData.read('splat.ply')
vertex_data = ply['vertex'].data  # structured numpy array

# --- Bounding box filter ---
x_min, x_max = -6, 0
y_min, y_max = -10, 2
z_min, z_max = -1.25, 1

mask = (
    (vertex_data['x'] >= x_min) & (vertex_data['x'] <= x_max) &
    (vertex_data['y'] >= y_min) & (vertex_data['y'] <= y_max) &
    (vertex_data['z'] >= z_min) & (vertex_data['z'] <= z_max)
)

filtered_data = vertex_data[mask]

# --- Save filtered PLY with all properties intact ---
filtered_vertex = PlyElement.describe(filtered_data, 'vertex')
PlyData([filtered_vertex], text=False).write('room_filtered_full.ply')

# --- Create Open3D point cloud for visualization ---
pcd = o3d.geometry.PointCloud()
points = np.vstack((filtered_data['x'], filtered_data['y'], filtered_data['z'])).T
pcd.points = o3d.utility.Vector3dVector(points)
# Visualize
o3d.visualization.draw_geometries([pcd])