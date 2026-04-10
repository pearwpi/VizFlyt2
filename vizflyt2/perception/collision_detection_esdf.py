import numpy as np
import open3d as o3d

class collisionDetection:
    def __init__(
        self,
        #path to occupancy grid
        ply_path: str,
        collision_threshold = 0.01,
        drone_radius = 0.01,
        #resolution of collision sphere
        num_points=10,
        env_min = None,
        env_max = None,
        voxel_size = 0.05,
    ):
        self.voxel_size = voxel_size
        #Load environment
        self.ply = o3d.io.read_point_cloud(ply_path)
        self.esdf = np.load("esdf_grid.npy")

        self.collision_threshold = collision_threshold
        self.drone_radius = drone_radius
        self.num_points = num_points

        if env_min is None:
            env_min = np.load("env_min.npy")
        if env_max is None:
            env_max = np.load("env_max.npy")

        #must import constants
        self.x_min_env, self.y_min_env, self.z_min_env = env_min
        self.x_max_env, self.y_max_env, self.z_max_env = env_max

    def metric_to_index(self, position):
        origin = np.array([
            self.x_min_env,
            self.y_min_env,
            self.z_min_env
        ])

        idx = ((position - origin) / self.voxel_size).astype(int)

        # prevent crashes
        idx = np.clip(idx, [0,0,0], np.array(self.esdf.shape) - 1)

        return idx
        
    def check_collision(self, position):
        idx = self.metric_to_index(position)
        return self.esdf[idx[0], idx[1], idx[2]] * self.voxel_size <= self.drone_radius

    def check_out_of_bounds(self, position):
        if position[0] < self.x_min_env or position[0] > self.x_max_env:
            return True
        
        if position[1] < self.y_min_env or position[1] > self.y_max_env:
            return True
        
        if position[2] < self.z_min_env or position[2] > self.z_max_env:
            return True

        return False

    def get_closest_obstacle_distance(self, position):
        idx = self.metric_to_index(position)
        return self.esdf[idx[0], idx[1], idx[2]] * self.voxel_size

    def get_minimum_boundary_distance(self, position):
        
        x_dist = position[0] - self.x_min_env - self.drone_radius

        y_dist = position[1] - self.y_min_env - self.drone_radius

        z_dist = position[2] - self.z_min_env - self.drone_radius

        if(abs(position[0] - self.x_min_env) > position[0] - self.x_max_env):
            x_dist = abs(position[0] - self.x_max_env) - self.drone_radius

        if(abs(position[1] - self.y_min_env) > position[1] - self.y_max_env):
            y_dist = abs(position[1] - self.y_max_env) - self.drone_radius

        if(abs(position[2] - self.y_min_env) > position[1] - self.y_max_env):
            z_dist = abs(position[2] - self.y_max_env) - self.drone_radius

        return x_dist, y_dist, z_dist