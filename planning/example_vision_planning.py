"""
Example: Using PotentialFieldPlanner for vision-based obstacle avoidance

This example demonstrates reactive planning using depth images.
The planner uses potential fields to navigate around obstacles.
"""

import numpy as np
from planning import PotentialFieldPlanner

# Example 1: Basic usage with synthetic depth image
print("=" * 60)
print("Example 1: Basic Potential Field Planning")
print("=" * 60)

# Create a synthetic depth image (640x480) with an obstacle
depth = np.zeros((480, 640), dtype=np.uint8)
# Add an obstacle in the center (high depth values = close obstacles)
depth[200:280, 280:360] = 150  # Central obstacle

# Create planner
planner = PotentialFieldPlanner(
    step_size=0.5,
    z_step_size=0.2,
    safety_radius=1.0,
    threshold=60
)

# Get velocity commands
velocity, viz = planner.plan(depth, save_visualization=False)
print(f"Velocity commands: forward={velocity[0]:.3f}, lateral={velocity[1]:.3f}, vertical={velocity[2]:.3f}")

# Example 2: With output directory for saving visualizations
print("\n" + "=" * 60)
print("Example 2: Planning with Visualization Saving")
print("=" * 60)

# Create planner with output directory
planner_with_viz = PotentialFieldPlanner(
    step_size=0.5,
    output_dir="./planning_output"
)

# Simulate a sequence of depth images
for i in range(5):
    # Create depth image with moving obstacle
    depth = np.zeros((480, 640), dtype=np.uint8)
    obstacle_x = 280 + i * 20
    depth[200:280, obstacle_x:obstacle_x+80] = 150
    
    velocity, _ = planner_with_viz.plan(depth)
    print(f"Frame {i}: vel=[{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")

print("Visualizations saved to ./planning_output/")

# Example 3: Integration with dynamics
print("\n" + "=" * 60)
print("Example 3: Integration with Dynamics Module")
print("=" * 60)

try:
    from dynamics import PointMassDynamics
    
    # Create dynamics model
    dynamics = PointMassDynamics(
        control_mode='velocity',
        dt=0.1
    )
    
    # Create planner
    planner = PotentialFieldPlanner(step_size=0.5)
    
    # Simulation loop
    print("\nSimulating vision-based navigation:")
    for step in range(10):
        # Get current position
        state = dynamics.get_state()
        pos = state['position']
        
        # Create synthetic depth image based on position
        # (In real scenario, this would come from a camera/sensor)
        depth = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
        
        # Plan velocity commands
        velocity, _ = planner.plan(depth, save_visualization=False)
        
        # Send to dynamics
        dynamics.set_control(velocity)
        dynamics.step()
        
        print(f"Step {step}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"vel=[{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
    
    print("\nIntegration successful!")
    
except ImportError:
    print("Dynamics module not available. Skipping integration example.")

# Example 4: Custom parameters
print("\n" + "=" * 60)
print("Example 4: Custom Planner Parameters")
print("=" * 60)

custom_planner = PotentialFieldPlanner(
    step_size=0.8,           # Faster forward speed
    z_step_size=0.3,         # More vertical motion
    safety_radius=1.5,       # More aggressive avoidance
    delta=-15,               # Different safety margin
    neighborhood_size=(60, 60),  # Larger neighborhood
    band_size=80,            # Wider boundary bands
    threshold=70             # Different obstacle threshold
)

depth = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
velocity, _ = custom_planner.plan(depth)
print(f"Custom planner velocity: {velocity}")

# Example 5: Using helper functions directly
print("\n" + "=" * 60)
print("Example 5: Direct Use of Helper Functions")
print("=" * 60)

from planning import (
    calculate_velocity,
    thresholding,
    calculate_free_direction_cc
)

# Object tracking velocity
object_centroid = (400, 300)  # Object at (400, 300) in 640x480 frame
frame_size = (640, 480)
tracking_vel = calculate_velocity(object_centroid, frame_size, v_forward=1.0)
print(f"Tracking velocity: {tracking_vel}")

# Depth thresholding
depth = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
thresholded = thresholding(depth, threshold=60)
print(f"Thresholded image shape: {thresholded.shape}")

# Free space direction
free_dir, cx, cy = calculate_free_direction_cc(thresholded)
print(f"Free direction: {free_dir}, centroid: ({cx:.1f}, {cy:.1f})")

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
