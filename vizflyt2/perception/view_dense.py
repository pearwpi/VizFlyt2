import numpy as np
import open3d as o3d

VOXEL_SIZE = 0.05

density = np.load("density_grid.npy")

print("Loaded density grid:", density.shape)

threshold = density.max() * 0.000001

print("Density threshold:", threshold)

total_voxels = density.size

# occupied voxels
inds = np.argwhere(density > threshold)

print("Points:", len(inds))
print("Point percent: %", len(inds)/total_voxels*100)

points = inds.astype(np.float32)
points *= VOXEL_SIZE


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
#pcd.paint_uniform_color([0, 0, 0])
o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud("density_points.ply", pcd)
