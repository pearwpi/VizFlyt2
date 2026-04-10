import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.ndimage import distance_transform_edt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, required=True)
args = parser.parse_args()

voxelsize = args.voxel_size


def create_density():

    FILE_PATH = "output.csv"
    #in meters, smaller is higher detail
    VOXEL_SIZE = voxelsize
    #get rid of weak gaussians
    ALPHA_THRESHOLD = 0.01
    #occupancy threshold, the higher the value the less sensitive occupance

    df = pd.read_csv(FILE_PATH)

    means = df[['x','y','z']].values
    scales = df[['scale_0','scale_1','scale_2']].values
    rots   = df[['rot_0','rot_1','rot_2','rot_3']].values
    alphas = df['opacity'].values

    # Convert log-scales toreal scales
    scales = np.exp(scales)

    print(f"Loaded {len(means)} Gaussians")

    def build_sigma_inv(scale, quat):
        R = Rotation.from_quat(quat).as_matrix()
        S = np.diag(scale)
        Sigma = R @ S @ S.T @ R.T
        return np.linalg.inv(Sigma)


    #build grid
    xmin, ymin, zmin = means.min(axis=0) - 0.5
    xmax, ymax, zmax = means.max(axis=0) + 0.5

    xs = np.arange(xmin, xmax, VOXEL_SIZE)
    ys = np.arange(ymin, ymax, VOXEL_SIZE)
    zs = np.arange(zmin, zmax, VOXEL_SIZE)

    density = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float32)

    print(f"Grid size: {density.shape}")


    for i in range(len(means)):
        if i % 1000 == 0:
            print(f"Processing Gaussian {i}/{len(means)}, {i/len(means)*100:.2f}%")

        alpha = alphas[i]
        if alpha < ALPHA_THRESHOLD:
            continue

        mu = means[i]
        scale = scales[i]
        quat = rots[i]

        Sigma_inv = build_sigma_inv(scale, quat)

        # influence radius
        r = 3 * np.max(scale)

        # voxel bounds
        ix_min = int((mu[0]-r - xmin)/VOXEL_SIZE)
        ix_max = int((mu[0]+r - xmin)/VOXEL_SIZE)
        iy_min = int((mu[1]-r - ymin)/VOXEL_SIZE)
        iy_max = int((mu[1]+r - ymin)/VOXEL_SIZE)
        iz_min = int((mu[2]-r - zmin)/VOXEL_SIZE)
        iz_max = int((mu[2]+r - zmin)/VOXEL_SIZE)

        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        iz_min = max(0, iz_min)

        ix_max = min(len(xs), ix_max)
        iy_max = min(len(ys), iy_max)
        iz_max = min(len(zs), iz_max)

        # Create voxel block (vectorized)
        X, Y, Z = np.meshgrid(
            xs[ix_min:ix_max],
            ys[iy_min:iy_max],
            zs[iz_min:iz_max],
            indexing='ij'
        )

        points = np.stack([X, Y, Z], axis=-1)

        d = points - mu 
        temp = d @ Sigma_inv
        mahal = np.sum(temp * d, axis=-1)

        values = np.exp(-0.5 * mahal)

        density[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += alpha * values

    np.save("density_grid.npy", density)

    print("Density Field Created.")


def specify_occupancy():
    satisfied = False    

    density = np.load("density_grid.npy")
    print("Loaded density grid:", density.shape)
    
    while( not satisfied):

        percentile = float(input("What percent (0-100) of density would you like to count as occupied: "))


        threshold = np.percentile(density, percentile)

        print("Density threshold:", threshold)

        total_voxels = density.size

        # occupied voxels
        inds = np.argwhere(density > threshold)

        print("Points:", len(inds))
        print("Point percent: %", len(inds)/total_voxels*100)

        points = inds.astype(np.float32)
        points *= voxelsize


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        #pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd])

        ans = input("Satisfied? (y/n)").lower()
        if ans == 'y':
            satisfied = True

    o3d.io.write_point_cloud("density_points.ply", pcd)
    return threshold


def create_esdf(threshold):
    density = np.load("density_grid.npy")

    # 0 = occupied, 1 = free
    grid = (density <= threshold).astype(np.uint8)

    esdf = distance_transform_edt(grid).astype(np.float32)

    print("ESDF shape:", esdf.shape)
    print("ESDF max value:", esdf.max())
    print("ESDF min value:", esdf.min())
    print("ESDF mean value:", esdf.mean())
    np.save("esdf_grid.npy", esdf)

if __name__ == "__main__":
    create_density()
    final_threshold = specify_occupancy()  
    create_esdf(final_threshold)