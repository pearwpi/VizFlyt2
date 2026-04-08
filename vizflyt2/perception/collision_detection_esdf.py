from tokenize import String

import numpy as np
import open3d as o3d

class collisionDetection:
    def __init__(
        self,
        #path to occupancy grid
        ply_path: String,
        esdf_path: "esdf_grid.npy",
        collision_threshold = 0.01,
        drone_radius = 0.01,
        #resolution of collision sphere
        num_points=10,
        env_min = [-20, -20, -20],
        env_max = [51, 51, 51],
        voxel_size = 0.05,
    ):
        self.voxel_size = 0.05
        #Load environment
        self.ply = o3d.io.read_point_cloud(ply_path)
        self.esdf = np.load(esdf_path)

        self.collision_threshold = collision_threshold
        self.drone_radius = drone_radius
        self.num_points = num_points


        #must import constants
        self.x_min_env, self.y_min_env, self.z_min_env = env_min
        self.x_max_env, self.y_max_env, self.z_max_env = env_max

    
    def check_collision(self, position):
       
        return self.esdf[position[0], position[1], position[2]] * self.voxel_size <= self.drone_radius

    def check_out_of_bounds(self, position):
        position = position * self.voxel_size
        if position[0] < self.x_min_env or position[0] > self.x_max_env:
            return True
        
        if position[1] < self.y_min_env or position[1] > self.y_max_env:
            return True
        
        if position[2] < self.z_min_env or position[2] > self.z_max_env:
            return True

        return False

    def get_closest_obstacle_distance(self, position):
        return self.esdf[position[0], position[1], position[2]] * self.voxel_size


    def get_minimum_boundary_distance(self, position):
        
        x_dist = self.position[0] - self.x_min_env - self.drone_radius

        y_dist = self.position[1] - self.y_min_env - self.drone_radius

        z_dist = self.position[2] - self.z_min_env - self.drone_radius

        if(abs(position[0] - self.x_min_env) > self.position[0] - self.x_max_env):
            x_dist = abs(self.position[0] - self.x_max_env) - self.drone_radius

        if(abs(position[1] - self.y_min_env) > self.position[1] - self.y_max_env):
            y_dist = abs(self.position[1] - self.y_max_env) - self.drone_radius

        if(abs(position[1] - self.y_min_env) > self.position[1] - self.y_max_env):
            z_dist = abs(self.position[2] - self.y_max_env) - self.drone_radius

        return x_dist, y_dist, z_dist