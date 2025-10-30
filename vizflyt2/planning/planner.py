"""
Vision-based Obstacle Avoidance Planner

This module provides reactive planning using potential fields on depth images.
Uses attractive forces (forward motion) and repulsive forces (from free space)
to navigate around obstacles.

Classes:
    PotentialFieldPlanner: Reactive planner using depth images for obstacle avoidance
    
Functions:
    calculate_velocity: Simple centroid tracking velocity calculation
    thresholding: Depth image thresholding for obstacle detection
    calculate_free_direction_cc: Find direction to largest free space
    calculate_free_direction_cc_boundary: Find free direction in image boundaries
"""

import os
import cv2 as cv
import numpy as np
import math
from typing import Tuple, Optional, Dict
from .base import ReactiveVisualPlanner


def calculate_velocity(object_centroid, frame_size, v_forward=1.0):
    """
    Calculate velocity commands based on object centroid position.
    
    Args:
        object_centroid: (cx, cy) pixel coordinates of tracked object
        frame_size: (width, height) of the frame
        v_forward: Forward velocity (default: 1.0)
    
    Returns:
        [forward, lateral, vertical] velocity commands
    """
    frame_width, frame_height = frame_size
    cx, cy = object_centroid

    # Additional parameter for sensitivity scaling
    sensitivity_scaling = 1.1  # Scales the lateral movement to respond to object shifts smoothly

    # Calculate normalized offset of object from frame center (-1 to 1)
    frame_center_x = frame_width / 2
    offset_x = (cx - frame_center_x) / frame_center_x

    # added new
    frame_center_y = frame_height / 2
    offset_y = (cy - frame_center_y) / frame_center_y

    # Scale the lateral movement by avoidance_distance and sensitivity_scaling
    lateral_movement = offset_x * 3.5 * sensitivity_scaling

    print('Lateral movement ', lateral_movement)
    return [v_forward, lateral_movement, 0]

# Function to create directories
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")
    else:
        print(f"Exists: {path}")



def thresholding(gray, threshold=60):
    """
    Apply threshold to depth image to detect obstacles.
    
    Args:
        gray: Grayscale depth image
        threshold: Threshold value (default: 60)
    
    Returns:
        Thresholded mask where values below threshold are set to zero
    """
    # print("Before thresholding:", np.min(gray), np.max(gray))
    
    # Create a mask where values below the threshold are set to zero
    mask = np.where(gray < threshold, 0, gray)
    
    # print("After thresholding:", np.min(mask), np.max(mask))
    return mask


def calculate_neighborhood(mask, center_x, center_y, neighborhood_size=(40, 40)):
    """
    Extract a small neighborhood centered on the image center.

    :param mask: Thresholded depth mask.
    :param center_x: X-coordinate of the center of the neighborhood.
    :param center_y: Y-coordinate of the center of the neighborhood.
    :param neighborhood_size: Tuple specifying the height and width of the neighborhood.
    :return: Extracted neighborhood region.
    """
    h, w = mask.shape
    half_h, half_w = neighborhood_size[0] // 2, neighborhood_size[1] // 2
    x_min = max(0, center_x - half_w)
    x_max = min(w, center_x + half_w)
    y_min = max(0, center_y - half_h)
    y_max = min(h, center_y + half_h)
    return mask[y_min:y_max, x_min:x_max]


def calculate_Zclose_and_Z0(mask, neighborhood, delta=-10):
    """
    Calculate Z_close and Z_0 from the neighborhood.

    :param mask: Thresholded depth mask.
    :param neighborhood: Extracted neighborhood region.
    :param delta: Safety margin for Z_0 calculation.
    :return: Z_close (maximum intensity), Z_0 (minimum intensity > Z_close + delta).
    """
    Z_close = np.max(neighborhood)
    unsafe_regions = neighborhood[neighborhood > (Z_close + delta)]
    Z_0 = np.min(unsafe_regions) if unsafe_regions.size > 0 else Z_close
    return Z_close, Z_0

def visualize_neighborhood_on_mask(
    mask,
    center_x,
    center_y,
    arrow_2d,
    neighborhood_size,
    free_space_cx,
    free_space_cy
):
    """
    Create a debug visualization by:
      - Drawing a rectangle around the neighborhood of interest.
      - Drawing an arrow (arrow_2d) from the image center.
      - Drawing a circle at (free_space_cx, free_space_cy).
      - (Extended) Drawing the largest free contour in blue, plus its centroid in white.

    :param mask: 2D thresholded depth mask (white=obstacle, black=free).
    :param center_x: X-coordinate of the image center.
    :param center_y: Y-coordinate of the image center.
    :param arrow_2d: (dx, dy) vector for navigation arrow in *image* coordinates.
    :param neighborhood_size: (height, width) of the rectangle around (center_x, center_y).
    :param free_space_cx: X-coordinate of free-space centroid to plot.
    :param free_space_cy: Y-coordinate of free-space centroid to plot.
    :return: BGR image with the drawn rectangle, arrow, centroid circle, and largest free contour.
    """

    # Make a BGR copy of the mask for drawing
    visualization = mask.copy()
    visualization = cv.cvtColor(visualization, cv.COLOR_GRAY2BGR)

    # 1) Neighborhood rectangle
    half_h, half_w = neighborhood_size[0] // 2, neighborhood_size[1] // 2
    x_min = max(0, center_x - half_w)
    x_max = min(mask.shape[1], center_x + half_w)
    y_min = max(0, center_y - half_h)
    y_max = min(mask.shape[0], center_y + half_h)

    cv.rectangle(
        visualization,
        (x_min, y_min),
        (x_max, y_max),
        (0, 0, 255),  # red rectangle
        thickness=2
    )

    # 2) Draw the navigation arrow
    draw_force_arrow_on_visualization(visualization, arrow_2d)

    # 3) Draw a circle at the free-space centroid
    cx_int, cy_int = int(free_space_cx), int(free_space_cy)
    cv.circle(visualization, (cx_int, cy_int), 4, (0, 255, 0), -1)

    # 4) (Extended) Find & draw the largest free-space contour in blue
    #    We'll re-invert mask to find free space as white=255
    free_space = np.where(mask == 0, 255, 0).astype(np.uint8)
    contours, _ = cv.findContours(free_space, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(visualization, [largest_contour], -1, (255, 0, 0), 2)

        # Compute the contour's centroid and draw it
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cxx = int(M["m10"] / M["m00"])
            cyy = int(M["m01"] / M["m00"])
            cv.circle(visualization, (cxx, cyy), 4, (255, 255, 255), -1)

    return visualization

def draw_force_arrow_on_visualization(
    visualization,
    arrow_2d, 
    arrow_scale=30,
    arrow_color=(0, 0, 255)
):
    """
    Draw a 2D arrow on 'visualization' to represent a navigation vector.

    :param visualization: BGR image on which to draw.
    :param arrow_2d: 2D vector (dx, dy) in image coordinates.
    :param arrow_scale: Scaling factor for the arrow length.
    :param arrow_color: (B, G, R) color of the arrow.
    """
    h, w, _ = visualization.shape
    center_x, center_y = w // 2, h // 2  # Draw arrow from the image center

    # Normalize (dx, dy) so we can scale consistently
    norm = np.linalg.norm(arrow_2d)
    if norm > 1e-6:
        arrow_2d_normalized = arrow_2d / norm
    else:
        arrow_2d_normalized = np.zeros_like(arrow_2d)

    # Scale the arrow
    arrow_dx = int(arrow_scale * arrow_2d_normalized[0])
    # In many image conventions, +dy goes down, so we invert for upward:
    arrow_dy = int(-arrow_scale * arrow_2d_normalized[1])

    start_point = (center_x, center_y)
    end_point = (center_x + arrow_dx, center_y + arrow_dy)

    cv.arrowedLine(
        visualization,
        start_point,
        end_point,
        color=arrow_color,
        thickness=2,
        tipLength=0.3
    )

def calculate_free_direction_cc(mask):
    """
    Calculate the free direction (v_free) based on the largest free-space region
    (instead of the nearest to center).

    :param mask: Thresholded depth mask, where obstacles=255, free=0.
    :return: A tuple: (direction_2D, centroid_x, centroid_y)
             

    direction_2D is the 2D unit vector [dx, dy] from the image center
                 to the largest free-space contour's centroid.
        centroid_x, centroid_y are the coordinates of that largest free-space
                   contour's centroid in the image. If no free region is found, returns( [0,0], 0, 0 )."""

    h, w = mask.shape
    center_x, center_y = w // 2, h // 2

    # Convert the mask so free=255 and obstacles=0
    # (this makes free regions white, which is typical for contour detection)
    free_space = np.where(mask == 0, 255, 0).astype(np.uint8)

    # Find contours of free regions
    contours, _ = cv.findContours(free_space, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No free region found
        return np.zeros(2, dtype=np.float32), 0.0, 0.0

    # Identify the largest contour based on area
    largest_contour = max(contours, key=cv.contourArea)

    # Compute centroid of the largest contour
    M = cv.moments(largest_contour)
    if M["m00"] == 0:  # Avoid division by zero
        return np.zeros(2, dtype=np.float32), 0.0, 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    direction = np.array([cx - center_x, cy - center_y], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction /= norm
    else:
        direction = np.zeros_like(direction)

    return direction, cx, cy


################################ Band-Region implementation ################################
def calculate_free_direction_cc_boundary(mask, band_size=50):
    """
    Finds the largest free-space contour *only* in the boundary bands (top, bottom,
    left, right) of 'mask', and returns a 2D unit direction vector [dx, dy]
    from the image center to that boundary contour's centroid.

    :param mask: 2D numpy array, obstacles=255, free=0
    :param band_size: Thickness of each boundary band in pixels
    :return: (direction_2D, centroid_x, centroid_y)
             where direction_2D = [dx, dy] is a normalized vector
             pointing from image center -> largest boundary free-space contour.
    """
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2

    # 1) Convert 'free' regions (mask==0) to white=255
    free_space = np.where(mask == 0, 255, 0).astype(np.uint8)

    # 2) Create a "boundary mask" that is 255 in the boundary bands, 0 in the interior
    boundary_mask = np.zeros((h, w), dtype=np.uint8)

    # Top band
    boundary_mask[0:band_size, :] = 255
    # Bottom band
    boundary_mask[h-band_size:h, :] = 255
    # Left band
    boundary_mask[:, 0:band_size] = 255
    # Right band
    boundary_mask[:, w-band_size:w] = 255

    # 3) Keep only free space that lies in those boundary bands
    boundary_free_space = cv.bitwise_and(free_space, boundary_mask)

    # 4) Find contours in boundary_free_space
    contours, _ = cv.findContours(boundary_free_space, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No free space in the boundary => no direction
        return np.zeros(2, dtype=np.float32), 0.0, 0.0

    # 5) Largest contour by area
    largest_contour = max(contours, key=cv.contourArea)

    # 6) Compute centroid
    M = cv.moments(largest_contour)
    if M["m00"] == 0:
        return np.zeros(2, dtype=np.float32), 0.0, 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 7) Direction from center to that centroid
    direction = np.array([cx - center_x, cy - center_y], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction /= norm
    else:
        direction = np.zeros_like(direction)

    return direction, cx, cy
################################################################################################

def potential_field_planner(
    depth,
    step_size,  # Move non-default argument up
    path_dir,  # Move non-default argument up
    threshold_mask_path,  # Move non-default argument up
    counter,  # Move non-default argument up
    delta,
    neighborhood_size,
    z_step_size,
    safety_radius
):
    """
    Potential Field Planner:
      - Attractive force: purely forward in Y.
      - Repulsive force: derived from the connected components of free space.
      - Weights (w) change based on proximity to obstacles (Z_close, Z_0).

    :param depth: Thresholded depth mask (2D numpy array). 
    :param path_dir: Path to output directory.
    :param counter: Frame counter done via timestamp.
    :param delta: Safety margin for Z_0.
    :param neighborhood_size: Size of the region around the image center.
    :param step_size: Step size for drone motion.
    :param z_step_size: Step size in Z-direction for drone motion.
    :param safety_radius: Safety radius for movement scaling.
    :return: Updated coordinates and collision count.
    """
    
    # 1) Thresholding
    obstacles = thresholding(depth)

    # 2) Center of the image
    h, w = depth.shape
    center_x, center_y = w // 2, h // 2

    # 3) Compute Z_close, Z_0 from the local neighborhood
    neighborhood = calculate_neighborhood(obstacles, center_x, center_y, neighborhood_size)
    Z_close, Z_0 = calculate_Zclose_and_Z0(obstacles, neighborhood, delta)

    # 4) Weight calculation based on proximity
    w = 1 / (1 + math.exp(-Z_0 / Z_close)) if Z_close > 0 else 0

    # 5) Define forces
    #    a) Attractive force purely along Y (forward)
    attractive_force = np.array([0.0, 1.0, 0.0], dtype=float)

    ################# Band Repulsion #################
    free_dir_2d, free_space_cx, free_space_cy = calculate_free_direction_cc_boundary(
        obstacles.astype(np.uint8), 
        band_size=50  # You can adjust this as needed
    )

    
    #    b) Repulsive force: connected components direction
    #       Convert 2D to 3D => x (left-right), z (inverted up-down)
    # free_dir_2d, free_space_cx, free_space_cy = calculate_free_direction_cc(obstacles.astype(np.uint8))
    repulsive_force = np.array([free_dir_2d[0], 0.0, -free_dir_2d[1]], dtype=float)


    # 6) Weighted combination
    print(f"Weights:\n Forward (att) : {1 - w:.2f}\n CC-based rep  : {w:.2f}")
    total_force = (1 - w) * attractive_force + w * repulsive_force

    # 7) Apply force to drone’s position
    forward_vel = step_size
    lateral_vel = 8.5*step_size * safety_radius * total_force[0]
    vertical_vel = z_step_size * safety_radius * total_force[2]

    velocity_commands = [forward_vel, lateral_vel, vertical_vel]

    print('Velocity commands ----- ', velocity_commands)

    # 8) Visualize: show the neighborhood highlight + arrow
    neighborhood_image = visualize_neighborhood_on_mask(obstacles, center_x, center_y, total_force, neighborhood_size, free_space_cx, free_space_cy)
    depth_bgr = cv.cvtColor(depth, cv.COLOR_GRAY2BGR)
    all_outputs = np.concatenate((depth_bgr, neighborhood_image), axis=1)
    #print("*"*51)
    print(f"Image shape, Height: {all_outputs.shape[0]}, Width: {all_outputs.shape[1]}")
    #print("*"*51)
    # all_outputs = np.concatenate((np.dstack((depth, depth, depth)), neighborhood_image), axis=1)

    write_output(
        threshold_mask_path,
        f"Frame{str(counter)}.png",
        all_outputs
    )
    return velocity_commands, neighborhood_image


def write_output(path, filename, output):
    """Write visualization output to file."""
    cv.imwrite(os.path.join(path, filename), output)


class PotentialFieldPlanner(ReactiveVisualPlanner):
    """
    Reactive planner for vision-based obstacle avoidance using potential fields.
    
    Maps: depth_image → velocity commands
    
    Uses attractive (forward) and repulsive (free space) forces for navigation.
    Automatically detects free space via connected component analysis on depth.
    
    Args:
        step_size: Forward step size (default: 0.5)
        z_step_size: Vertical step size (default: 0.2)
        safety_radius: Scaling factor for lateral/vertical motion (default: 1.0)
        delta: Safety margin for Z_0 calculation (default: -10)
        neighborhood_size: Size of region around image center (default: (40, 40))
        band_size: Boundary band thickness for free space detection (default: 50)
        threshold: Depth threshold for obstacle detection (default: 60)
        output_dir: Directory for saving visualizations (default: None)
    
    Example:
        ```python
        planner = PotentialFieldPlanner(step_size=0.5, safety_radius=1.0)
        
        for step in range(1000):
            depth = get_depth_image()  # From camera/sensor
            action = planner.compute_action(depth_image=depth)
            velocity = action['velocity']  # [forward, lateral, vertical]
            
            # Send to dynamics
            dynamics.set_control(velocity)
            dynamics.step()
            planner.step()
        ```
    """
    
    def __init__(
        self,
        step_size: float = 0.5,
        z_step_size: float = 0.2,
        safety_radius: float = 1.0,
        delta: float = -10,
        neighborhood_size: Tuple[int, int] = (40, 40),
        band_size: int = 50,
        threshold: int = 60,
        output_dir: Optional[str] = None,
        verbose: bool = False
    ):
        super().__init__()
        self.step_size = step_size
        self.z_step_size = z_step_size
        self.safety_radius = safety_radius
        self.delta = delta
        self.neighborhood_size = neighborhood_size
        self.band_size = band_size
        self.threshold = threshold
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory if specified
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def compute_action(
        self, 
        depth_image: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
        save_visualization: bool = False,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute velocity commands from depth image.
        
        Args:
            depth_image: 2D depth image (obstacles have higher values)
            rgb_image: RGB image (not used, for interface compatibility)
            save_visualization: Whether to save visualization to output_dir
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Dictionary with:
                'velocity': [forward, lateral, vertical] velocity commands
                'info': Optional dict with debug information and visualization
        """
        if depth_image is None:
            raise ValueError("PotentialFieldPlanner requires depth_image")
        
        # reduce 1 channel to 2d array
        if len(depth_image.shape) == 3 and depth_image.shape[2] == 1:
            depth_image = depth_image[:, :, 0]
        
        # Threshold the depth image
        obstacles = thresholding(depth_image, self.threshold)
        
        # Get image center
        h, w = depth_image.shape
        center_x, center_y = w // 2, h // 2
        
        # Compute Z_close and Z_0 from neighborhood
        neighborhood = calculate_neighborhood(obstacles, center_x, center_y, self.neighborhood_size)
        Z_close, Z_0 = calculate_Zclose_and_Z0(obstacles, neighborhood, self.delta)
        
        # Weight calculation based on proximity
        w = 1 / (1 + math.exp(-Z_0 / Z_close)) if Z_close > 0 else 0
        
        # Attractive force (purely forward in Y)
        attractive_force = np.array([0.0, 1.0, 0.0], dtype=float)
        
        # Repulsive force from boundary free space
        free_dir_2d, free_space_cx, free_space_cy = calculate_free_direction_cc_boundary(
            obstacles.astype(np.uint8), 
            band_size=self.band_size
        )
        
        # Convert 2D to 3D: x (left-right), z (inverted up-down)
        repulsive_force = np.array([free_dir_2d[0], 0.0, -free_dir_2d[1]], dtype=float)
        
        # Weighted combination
        if self.verbose:
            print(f"Weights:\n Forward (att) : {1 - w:.2f}\n CC-based rep  : {w:.2f}")
        total_force = (1 - w) * attractive_force + w * repulsive_force
        
        # Convert to velocity commands
        forward_vel = self.step_size
        lateral_vel = 8.5 * self.step_size * self.safety_radius * total_force[0]
        vertical_vel = -1 * self.z_step_size * self.safety_radius * total_force[2]
        
        velocity = np.array([forward_vel, lateral_vel, vertical_vel])
        
        if self.verbose:
            print('Velocity commands ----- ', velocity)
        
        # Generate visualization
        info = {}
        if save_visualization or self.output_dir is not None:
            neighborhood_image = visualize_neighborhood_on_mask(
                obstacles, center_x, center_y, total_force, 
                self.neighborhood_size, free_space_cx, free_space_cy
            )
            depth_bgr = cv.cvtColor(depth_image, cv.COLOR_GRAY2BGR)
            visualization = np.concatenate((depth_bgr, neighborhood_image), axis=1)
            
            info['visualization'] = visualization
            info['weights'] = {'attractive': 1-w, 'repulsive': w}
            info['forces'] = {'attractive': attractive_force, 'repulsive': repulsive_force}
            
            # Save if output directory is set
            if self.output_dir is not None:
                write_output(
                    self.output_dir,
                    f"Frame{str(self.step_count)}.png",
                    visualization
                )
        
        return {
            'velocity': velocity,
            'info': info
        }
    
    def plan(self, depth_image: np.ndarray, save_visualization: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Legacy interface for backward compatibility.
        
        Args:
            depth_image: 2D depth image
            save_visualization: Whether to save visualization
        
        Returns:
            Tuple of (velocity_commands, visualization_image)
        """
        result = self.compute_action(depth_image=depth_image, save_visualization=save_visualization)
        visualization = result['info'].get('visualization', None) if result['info'] else None
        return result['velocity'], visualization

