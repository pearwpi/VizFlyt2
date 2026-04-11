import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.ndimage import distance_transform_edt
import argparse
from plyfile import PlyElement
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, required=True)
parser.add_argument("--ply_path", type=str, required=True)
args = parser.parse_args()

voxelsize = args.voxel_size
PLY_PATH = args.ply_path

def run_clipper():

    ply = PlyData.read(PLY_PATH)
    vertex_data = ply['vertex'].data

    #stack all points
    all_points_full = np.vstack((
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z'])).T

    
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(all_points_full)
    
    #Downsample for rendering
    pcd_down = pcd_full.voxel_down_sample(voxel_size=voxelsize * 2)
    #convert downsampled for UI interaction
    all_points = np.asarray(pcd_down.points)

    #initialize bounds as max for original ply
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    bounds = {
        "x_min": mins[0], "x_max": maxs[0],
        "y_min": mins[1], "y_max": maxs[1],
        "z_min": mins[2], "z_max": maxs[2],
    }

    #initialize app
    app = gui.Application.instance
    app.initialize()

    #create windoq
    window = app.create_window("Clipper", 1100, 800)

    #render section
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    
    #gui section
    panel = gui.Vert(10)
    window.add_child(panel)

    pcd = o3d.geometry.PointCloud()
    pcd_outside = o3d.geometry.PointCloud()

    #rendering material
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0

    def update():
        #mask applied to downsampled points
        mask = (
            (all_points[:, 0] >= bounds["x_min"]) & (all_points[:, 0] <= bounds["x_max"]) &
            (all_points[:, 1] >= bounds["y_min"]) & (all_points[:, 1] <= bounds["y_max"]) &
            (all_points[:, 2] >= bounds["z_min"]) & (all_points[:, 2] <= bounds["z_max"])
        )

        inside = all_points[mask]
        outside = all_points[~mask]

        if len(inside) == 0:
            return

        pcd.points = o3d.utility.Vector3dVector(inside)
        pcd.paint_uniform_color([.9, .1, .1])  # inside

        pcd_outside.points = o3d.utility.Vector3dVector(outside)
        pcd_outside.paint_uniform_color([0.7, 0.7, 0.7])  # outside

        #re-redner scene
        scene.scene.clear_geometry()
        scene.scene.add_geometry("outside", pcd_outside, mat)
        scene.scene.add_geometry("inside", pcd, mat)

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[bounds["x_min"], bounds["y_min"], bounds["z_min"]],
            max_bound=[bounds["x_max"], bounds["y_max"], bounds["z_max"]],
        )
        bbox.color = (1.0, 1.0, 0.0)  # yellow
        scene.scene.add_geometry("bbox", bbox, mat)

        scene.setup_camera(
            60,
            o3d.geometry.AxisAlignedBoundingBox(min_bound=mins, max_bound=maxs),
            all_points.mean(axis=0)
        )

    def save():
        #apply bounds to full data
        mask = (
            (vertex_data['x'] >= bounds["x_min"]) & (vertex_data['x'] <= bounds["x_max"]) &
            (vertex_data['y'] >= bounds["y_min"]) & (vertex_data['y'] <= bounds["y_max"]) &
            (vertex_data['z'] >= bounds["z_min"]) & (vertex_data['z'] <= bounds["z_max"])
        )

        #filter & save full data
        filtered = vertex_data[mask]

        PlyData([PlyElement.describe(filtered, 'vertex')]).write("filtered.ply")

        #compute and display bounds from filered data
        x = filtered['x']
        y = filtered['y']
        z = filtered['z']

        env_min = np.array([x.min(), y.min(), z.min()])
        env_max = np.array([x.max(), y.max(), z.max()])

        print("\nSaved filtered.ply")
        print("env_min =", env_min)
        print("env_max =", env_max)
        
        np.save("env_min.npy", env_min)
        np.save("env_max.npy", env_max)

        window.close()

        return "filtered.ply", env_min, env_max

    #create location for saved data
    result = {"data": None}

    #save stored data
    def save_wrapper():
        result["data"] = save()

    #create floating point sliders
    def slider(name, axis):
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(mins[axis], maxs[axis])
        s.double_value = bounds[name]

        #update bounds depending on slider movement
        def on_change(v):
            bounds[name] = v
            update()

        #add ui elements
        s.set_on_value_changed(on_change)
        panel.add_child(gui.Label(name))
        panel.add_child(s)

    slider("x_min", 0)
    slider("x_max", 0)
    slider("y_min", 1)
    slider("y_max", 1)
    slider("z_min", 2)
    slider("z_max", 2)

    btn = gui.Button("Save & Continue")
    btn.set_on_clicked(save_wrapper)
    panel.add_child(btn)

    def layout(ctx):
        #make scene fill full window and panel on the left
        r = window.content_rect
        scene.frame = r
        panel.frame = gui.Rect(r.x, r.y, 200, r.height)

    window.set_on_layout(layout)
    update()
    app.run()

    return result["data"]

def create_density(path, env_min, env_max):

    FILE_PATH = path
    #in meters, smaller is higher detail
    VOXEL_SIZE = voxelsize
    #get rid of weak gaussians
    ALPHA_THRESHOLD = 0.01
    #occupancy threshold, the higher the value the less sensitive occupance

    # df = pd.read_csv(FILE_PATH)

    ply = PlyData.read(FILE_PATH)
    vertex = ply['vertex'].data

    means = np.vstack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ]).T

    scales = np.vstack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ]).T

    rots = np.vstack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ]).T

    alphas = vertex['opacity']

    # Convert log-scales toreal scales
    scales = np.exp(scales)

    print(f"Loaded {len(means)} Gaussians")

    def build_sigma_inv(scale, quat):
        R = Rotation.from_quat(quat).as_matrix()
        S = np.diag(scale)
        Sigma = R @ S @ S.T @ R.T
        return np.linalg.inv(Sigma)

    xmin, ymin, zmin = env_min
    xmax, ymax, zmax = env_max

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


def specify_occupancy(voxelsize, env_min):

    density = np.load("density_grid.npy")
    print("Loaded density grid:", density.shape)

    #only consider voxels with non zero density
    nonzero = density[density > 0]

    total_voxels = density.size
    total_nonzero = len(nonzero)

    print("Total voxels:", total_voxels)
    print("Nonzero voxels:", total_nonzero)

    #setup gui
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("Density Viewer", 1100, 800)

    #make scene & panel
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    panel = gui.Vert(10)
    window.add_child(panel)

    #point cloud and material
    pcd = o3d.geometry.PointCloud()

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0

    #display stats
    stats_label = gui.Label("")
    panel.add_child(stats_label)

    state = {
        "percentile": 50.0,
        "threshold": 0.0
    }

    def update():
        percentile = state["percentile"]

        threshold = np.percentile(nonzero, percentile)
        state["threshold"] = threshold

        inds = np.argwhere(density > threshold)
        values = density[density > threshold]

        points = inds.astype(np.float32) * voxelsize + env_min
        
        pcd.points = o3d.utility.Vector3dVector(points)

        #color gradiant
        z_vals = points[:, 2]

        z_min = env_min[2]
        z_max = env_min[2] + density.shape[2] * voxelsize

        # normalize to [0,1]
        t = (z_vals - z_min) / (z_max - z_min + 1e-8)

        # create rainbow (HSV → RGB approximation)
        colors = np.zeros((len(t), 3))

        colors[:, 0] = np.clip(1.5 - np.abs(4*t - 3), 0, 1)  # red
        colors[:, 1] = np.clip(1.5 - np.abs(4*t - 2), 0, 1)  # green
        colors[:, 2] = np.clip(1.5 - np.abs(4*t - 1), 0, 1)  # blue

        pcd.colors = o3d.utility.Vector3dVector(colors)

        scene.scene.clear_geometry()
        scene.scene.add_geometry("pcd", pcd, mat)

        #update ui
        stats_label.text = (
            f"Percentile: {percentile:.2f}\n"
            f"Threshold: {threshold:.6f}\n"
            f"Points: {len(points)}\n"
            f"Nonzero kept: {len(points)/total_nonzero*100:.2f}%\n"
            f"Grid kept: {len(points)/total_voxels*100:.2f}%"
        )

    update()

    #Add slider
    slider = gui.Slider(gui.Slider.DOUBLE)
    slider.set_limits(0, 100)
    slider.double_value = state["percentile"]

    def on_slider(val):
        state["percentile"] = val
        update()

    slider.set_on_value_changed(on_slider)

    panel.add_child(gui.Label("Density Percentile"))
    panel.add_child(slider)

    #Save
    def save_and_close():
        inds = np.argwhere(density > state["threshold"])
        origin = env_min
        points = inds.astype(np.float32) * voxelsize + origin

        pcd_save = o3d.geometry.PointCloud()
        pcd_save.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud("density_points.ply", pcd_save)

        print("\nSaved density_points.ply")
        print("Final threshold:", state["threshold"])

        window.close()

    btn = gui.Button("💾 Save & Exit")
    btn.set_on_clicked(save_and_close)
    panel.add_child(btn)

    #create layout
    def on_layout(ctx):
        r = window.content_rect
        scene.frame = r
        panel.frame = gui.Rect(r.x, r.y, 250, r.height)

    window.set_on_layout(on_layout)

    #Set camera

    min_bound= [0, 0, 0]
    max_bound= env_min + np.array(density.shape) * voxelsize
    center = (min_bound + max_bound)/ 2
    scene.setup_camera(
        60,
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        ),
        max_bound / 2,
    )

    app.run()

    return state["threshold"]

def create_esdf(threshold, env_min, env_max):
    density = np.load("density_grid.npy")

    grid = (density > threshold).astype(np.uint8)  # invert meaning
    grid = 1 - grid  # make 0 = occupied, 1 = free

    esdf = distance_transform_edt(grid).astype(np.float32)

    print("ESDF shape:", esdf.shape)
    print("ESDF max value:", esdf.max())
    print("ESDF min value:", esdf.min())
    print("ESDF mean value:", esdf.mean())
    print("env_min =", env_min)
    print("env_max =", env_max)

    np.save("esdf_grid.npy", esdf)

if __name__ == "__main__":
    path, env_min, env_max = run_clipper()
    create_density(path, env_min, env_max)
    final_threshold = specify_occupancy(voxelsize, env_min)  
    create_esdf(final_threshold, env_min, env_max)