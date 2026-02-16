import numpy as np
import open3d as o3d
import sys
from collision_detection import collisionDetection

def run_demo(ply_path: str):
    cd = collisionDetection(
        ply_path = ply_path,
        collision_threshold = 0.01,
        drone_radius = 0.03,
        num_points = 12,
        )

    env = cd.occupancy_grid

    #Create drone
    drone = o3d.geometry.TriangleMesh.create_sphere(radius=cd.drone_radius)
    drone.compute_vertex_normals()
    drone.paint_uniform_color([0.2, 1.0, 0.2])

    pos = np.array([0.0, 0.0, 0.0], dtype=float)
    step = 0.05

    def set_drone_center(p):
        drone.translate(p - drone.get_center(), relative=True)

    set_drone_center(pos)

    #render point cloud
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Collision Demo (WASDRF , Q)")
    vis.add_geometry(env)
    vis.add_geometry(drone)
    
    vc = vis.get_view_control()
    vc.set_lookat(pos.tolist()) 
    #set along x axis, and z as up
    vc.set_front([1.0, 0.0, 0.0])
    vc.set_up([0.0, 0.0, 1.0])   
    vc.set_zoom(.1) 

    #update color for when hit
    def update_color(hit: bool):
        drone.paint_uniform_color([1.0, 0.2, 0.2] if hit else [0.2, 1.0, 0.2])

    #movement mechanics and calling collision check
    def attempt_move(d):
        nonlocal pos
        new_pos = pos + d

        hit = cd.check_collision(new_pos)
        if hit:
            update_color(True)
            vis.update_geometry(drone)
            print(f"COLLISION: BLOCKED AT {new_pos}")
            return False
        
        update_color(False)

        pos = new_pos
        set_drone_center(pos)
        vis.update_geometry(drone)
        print(f"Moved to {pos}")
        return False

    vis.register_key_callback(ord("W"), lambda v: attempt_move(np.array([ step, 0.0, 0.0])))
    vis.register_key_callback(ord("S"), lambda v: attempt_move(np.array([-step, 0.0, 0.0])))
    vis.register_key_callback(ord("A"), lambda v: attempt_move(np.array([0.0,  step, 0.0])))
    vis.register_key_callback(ord("D"), lambda v: attempt_move(np.array([0.0, -step, 0.0])))
    vis.register_key_callback(ord("R"), lambda v: attempt_move(np.array([0.0, 0.0,  step])))
    vis.register_key_callback(ord("F"), lambda v: attempt_move(np.array([0.0, 0.0, -step])))
    vis.register_key_callback(ord("Q"), lambda v: (vis.close(), False)[1])

    print("Controls: W/S (±X), A/D (±Y), R/F (±Z), Q quit")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    run_demo(sys.argv[1])       
