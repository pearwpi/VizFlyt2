import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

FILE_PATH = "output.csv"
#in meters, smaller is higher detail
VOXEL_SIZE = 0.0125
#get rid of weak gaussians
ALPHA_THRESHOLD = 0.01
#occupancy threshold, the higher the value the less sensitive occupance
PERCENTILE_THRESHOLD = 99.5


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


tau = np.percentile(density, PERCENTILE_THRESHOLD)

occupancy = density > tau

print(f"Occupancy ratio: {occupancy.mean()}")

np.save("occupancy_grid.npy", occupancy)
np.save("density_grid.npy", density)

print("Done.")