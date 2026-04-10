import numpy as np
import open3d as o3d
import sys
from collision_detection_esdf import collisionDetection
import argparse

def run_demo(ply_path: str, voxel_size float):

    cd = collisionDetection(
        ply_path = ply_path,
        esdf_path = "esdf_grid.npy",
        collision_threshold = 0.01,
        drone_radius = 0.3,
        num_points = 12,
        voxel_size = voxel_size
        )

    env = cd.ply

    drone = o3d.geometry.TriangleMesh.create_sphere(radius=cd.drone_radius)
    drone.compute_vertex_normals()
    drone.paint_uniform_color([0.2, 1.0, 0.2])
    
    # init_pos = ([2, 2, 2])
    # pos = np.array([init_pos[0] / voxel_size, init_pos[1] / voxel_size, init_pos[2] / voxel_size])
    pos = np.array([212, 400, 136], dtype=int)
    step = 1

    free_radius = cd.get_closest_obstacle_distance(pos)
    print(f"Initial position: {pos * cd.voxel_size}, ESDF distance: {free_radius}")

    free = o3d.geometry.TriangleMesh.create_sphere(radius=free_radius)
    free = o3d.geometry.LineSet.create_from_triangle_mesh(free)
    free.paint_uniform_color([1.0, 0, 1.0])
    free.translate(pos * cd.voxel_size, relative=False) 
    
    drone.translate(pos * cd.voxel_size, relative=False)  

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Collision Demo (WASDRF , Q)")
    vis.add_geometry(env)
    vis.add_geometry(drone)
    vis.add_geometry(free)
    
    vc = vis.get_view_control()

    def update_camera():
        lookat = pos * cd.voxel_size
        front = np.array([0.0, -.5, 0.3])
        up = np.array([0.0, 0.0, 1.0])

        vc.set_lookat(lookat.tolist())
        vc.set_front(front.tolist())
        vc.set_up(up.tolist())
        vc.set_zoom(0.2)

    update_camera()

    def update_color(hit: bool):
        drone.paint_uniform_color([1.0, 0.2, 0.2] if hit else [0.2, 1.0, 0.2])
        


    def attempt_move(d):
        nonlocal pos, free

        new_pos = pos + d

        if cd.check_out_of_bounds(new_pos):
            print(f"Out of bounds: {new_pos}")
            return False

        dist = cd.get_closest_obstacle_distance(new_pos)
        
        hit = cd.check_collision(new_pos)

        if hit:
            update_color(True)
            vis.update_geometry(drone)
            print(f"COLLISION: BLOCKED AT {new_pos * cd.voxel_size}, ESDF distance: {dist}")
            free.paint_uniform_color([1.0, 0, 0])
            vis.update_geometry(free)
            return False
        
        update_color(False)

        vis.remove_geometry(free, reset_bounding_box=False)

        free_radius = dist

        free = o3d.geometry.TriangleMesh.create_sphere(radius=free_radius)
        free = o3d.geometry.LineSet.create_from_triangle_mesh(free)
        free.paint_uniform_color([1.0, 0, 1.0])
       
        free.translate(new_pos * cd.voxel_size, relative=False)

        vis.add_geometry(free)

        pos = new_pos

        drone.translate(pos * cd.voxel_size, relative=False)

        vis.update_geometry(drone)

        update_camera()

        print(f"Moved to {pos * cd.voxel_size}, ESDF distance: {free_radius}")
        return False

    vis.register_key_callback(ord("W"), lambda v: attempt_move(np.array([ step, 0, 0])))
    vis.register_key_callback(ord("S"), lambda v: attempt_move(np.array([-step, 0, 0])))
    vis.register_key_callback(ord("A"), lambda v: attempt_move(np.array([0,  step, 0])))
    vis.register_key_callback(ord("D"), lambda v: attempt_move(np.array([0, -step, 0])))
    vis.register_key_callback(ord("R"), lambda v: attempt_move(np.array([0, 0,  step])))
    vis.register_key_callback(ord("F"), lambda v: attempt_move(np.array([0, 0, -step])))
    vis.register_key_callback(ord("Q"), lambda v: (vis.close(), False)[1])

    print("Controls: W/S (±X), A/D (±Y), R/F (±Z), Q quit")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision demo with ESDF")

    parser.add_argument("ply_path", type=str, help="Path to point cloud (.ply)")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size")

    args = parser.parse_args()

    run_demo(args.ply_path, args.voxel_size)