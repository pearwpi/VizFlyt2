import numpy as np
import open3d as o3d

class collisionDetection:
    def __init__(
        self,
        #path to occupancy grid
        ply_path: String,
        collision_threshold = 0.01,
        drone_radius = 0.01,
        #resolution of collision sphere
        num_points=10,
        env_min = [0, 0, 0],
        env_max = [1, 1, 1]
    ):
        #Load environment
        self.occupancy_grid = o3d.io.read_point_cloud(ply_path)

        if not self.occupancy_grid.has_points():
            raise RuntimeError("Occupancy PLY file is empty or not loaded correctly")

        self.collision_threshold = collision_threshold
        self.drone_radius = drone_radius
        self.num_points = num_points

        #precompute sphere surface
        #evenly spaced values from 0(representing south pole) - pi(representing north pole
        phi = np.linspace(0, np.pi, self.num_points)
        #create theta, evenly spaced values from 0 - 2pi, azimuth angle
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        phi, theta = np.meshgrid(phi, theta)

        self.sphere_template = np.vstack((
            #x = r*sin(phi)*cos(theta)
            self.drone_radius * np.sin(phi) * np.cos(theta),
            #y = r*sin(phi)*sin(phi)
            self.drone_radius * np.sin(phi) * np.sin(theta),
            #z = r*cos(phi)
            self.drone_radius * np.cos(phi)
            )).reshape(3, -1).T

        #must import constants
        self.x_min_env, self.y_min_env, self.z_min_env = env_min
        self.x_max_env, self.y_max_env, self.z_max_env = env_max

    
    def check_collision(self, position):

        #world relative sphere points = local sphere points + position
        sphere_points = self.sphere_template + position.reshape(1, 3)

        #put coordinates on o3d format
        drone_pcd = o3d.geometry.PointCloud()
        drone_pcd.points = o3d.utility.Vector3dVector(sphere_points)

        #distance = minimum |drone point - enviornment point| for all drone points
        distances = np.array(
                self.occupancy_grid.compute_point_cloud_distance(drone_pcd)
                )

        if distances.size == 0:
            return False

        return np.min(distances) < self.collision_threshold

    def check_out_of_bounds(self, position):

        if position[0] < self.x_min_env or position[0] > self.x_max_env:
            return True
        
        if position[1] < self.y_min_env or position[1] > self.y_max_env:
            return True
        
        if position[2] < self.z_min_env or position[1] > self.z_max_env:
            return True

        return False

    def get_closest_obstacle_distance(self, position):

        #world relative sphere points = local sphere points + position
        sphere_points = self.sphere_template + position.reshape(1, 3)

        #put coordinates on o3d format
        drone_pcd = o3d.geometry.PointCloud()
        drone_pcd.points = o3d.utility.Vector3dVector(sphere_points)

        #distance = minimum |drone point - enviornment point| for all drone points
        distances = np.array(
                self.occupancy_grid.compute_point_cloud_distance(drone_pcd)
                )

        return np.min(distances) 


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