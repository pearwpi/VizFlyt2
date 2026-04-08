import numpy as np
from scipy.ndimage import distance_transform_edt

density = np.load("density_grid.npy")

threshold = density.max() * 1e-6

# 0 = occupied, 1 = free
grid = (density <= threshold).astype(np.uint8)

esdf = distance_transform_edt(grid).astype(np.float32)

print("ESDF shape:", esdf.shape)
print("ESDF max value:", esdf.max())
print("ESDF min value:", esdf.min())
print("ESDF mean value:", esdf.mean())
np.save("esdf_grid.npy", esdf)
