"""
Complete Integration Example: Perception + Planning + Dynamics

This example shows the full pipeline:
1. Splat renderer generates depth images
2. Planning module computes velocity commands from depth
3. Dynamics executes the commands
4. Loop continues with updated position

Run this to see the complete system in action.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from planning import PotentialFieldPlanner
from dynamics import PointMassDynamics

# Try to import perception (may not be available)
try:
    from perception.splat_render import SplatRenderer
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False
    print("Note: Splat renderer not available, using synthetic depth images")

print("=" * 70)
print("COMPLETE INTEGRATION: Perception → Planning → Dynamics")
print("=" * 70)

# ============================================================================
# Setup
# ============================================================================

print("\n1. Setting up components...")

# Initial state
initial_state = {
    'position': np.array([0., 0., -50.]),      # Start at origin, 50m altitude
    'velocity': np.array([0., 0., 0.]),         # Start stationary
    'orientation_rpy': np.array([0., 0., 0.])   # Level orientation
}

# Dynamics (velocity control mode)
dynamics = PointMassDynamics(
    initial_state=initial_state,
    control_mode='velocity'
)
print(f"   ✓ Dynamics initialized: {dynamics.control_mode} mode")

# Planning (reactive obstacle avoidance)
planner = PotentialFieldPlanner(
    step_size=2.0,          # 2 m/s forward speed
    safety_radius=1.5,      # Aggressive avoidance
    threshold=60,           # Obstacle detection threshold
    verbose=False           # Quiet mode
)
print(f"   ✓ Planner initialized: step_size={planner.step_size} m/s")

# Perception (if available)
if HAS_RENDERER:
    # You would initialize with your config files
    # renderer = SplatRenderer("config.yml", "camera.json")
    # print("   ✓ Renderer initialized")
    pass
else:
    print("   ✓ Using synthetic depth generator")

# ============================================================================
# Synthetic depth image generator (simulates perception)
# ============================================================================

def generate_synthetic_depth(position, step):
    """
    Generate synthetic depth image based on current position.
    Simulates obstacles appearing in the environment.
    
    Args:
        position: Current drone position [x, y, z]
        step: Current simulation step
    
    Returns:
        depth: (480, 640) depth image, obstacles = high values
    """
    depth = np.zeros((480, 640), dtype=np.uint8)
    
    # Add obstacles based on position
    x, y, z = position
    
    # Obstacle 1: Wall ahead if x > 20m
    if x > 20:
        depth[150:330, 250:390] = 120  # Central obstacle
    
    # Obstacle 2: Left wall if y < -10m
    if y < -10:
        depth[:, :200] = 100
    
    # Obstacle 3: Right wall if y > 10m  
    if y > 10:
        depth[:, 440:] = 100
    
    # Obstacle 4: Periodic obstacles
    if step % 50 < 25 and x > 10:
        depth[200:280, 150:230] = 90
    
    return depth

# ============================================================================
# Main simulation loop
# ============================================================================

print("\n2. Running simulation loop...")
print("   (Perception → Planning → Dynamics)\n")

num_steps = 100
dt = 0.1

# Storage for logging
trajectory = []

for step in range(num_steps):
    # -----------------------------------------------------------------------
    # 1. PERCEPTION: Get depth image
    # -----------------------------------------------------------------------
    state = dynamics.get_state()
    position = state['position']
    orientation_rpy = state['orientation_rpy']
    
    if HAS_RENDERER:
        # Real renderer (if available)
        # position_render, orientation_quat = dynamics.get_render_params()
        # result = renderer.render(position_render, orientation_quat)
        # depth = result['depth']
        pass
    else:
        # Synthetic depth
        depth = generate_synthetic_depth(position, step)
    
    # -----------------------------------------------------------------------
    # 2. PLANNING: Compute velocity command from depth
    # -----------------------------------------------------------------------
    action = planner.compute_action(depth_image=depth)
    velocity_cmd = action['velocity']
    
    # -----------------------------------------------------------------------
    # 3. DYNAMICS: Execute command
    # -----------------------------------------------------------------------
    controls = {'velocity': velocity_cmd}
    dynamics.step(controls, dt)
    planner.step()
    
    # -----------------------------------------------------------------------
    # 4. LOGGING
    # -----------------------------------------------------------------------
    trajectory.append({
        'step': step,
        'position': position.copy(),
        'velocity': state['velocity'].copy(),
        'velocity_cmd': velocity_cmd.copy(),
        'has_obstacle': np.max(depth) > planner.threshold
    })
    
    # Print progress
    if step % 10 == 0:
        obs_str = "OBSTACLE" if trajectory[-1]['has_obstacle'] else "clear"
        print(f"   Step {step:3d}: pos=[{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f}] "
              f"vel=[{velocity_cmd[0]:5.2f}, {velocity_cmd[1]:5.2f}, {velocity_cmd[2]:5.2f}] "
              f"({obs_str})")

# ============================================================================
# Analysis
# ============================================================================

print("\n3. Simulation complete!")
print("-" * 70)

# Convert to arrays
positions = np.array([t['position'] for t in trajectory])
velocities = np.array([t['velocity'] for t in trajectory])
commands = np.array([t['velocity_cmd'] for t in trajectory])

# Statistics
total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
num_obstacles = sum([t['has_obstacle'] for t in trajectory])

print(f"Total distance traveled: {total_distance:.2f} m")
print(f"Average speed: {avg_speed:.2f} m/s")
print(f"Steps with obstacles: {num_obstacles}/{num_steps}")
print(f"Final position: [{positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f}]")

# ============================================================================
# Optional: Visualization
# ============================================================================

try:
    import matplotlib.pyplot as plt
    
    print("\n4. Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: 2D trajectory
    ax = axes[0, 0]
    obstacle_indices = [i for i, t in enumerate(trajectory) if t['has_obstacle']]
    clear_indices = [i for i, t in enumerate(trajectory) if not t['has_obstacle']]
    
    ax.plot(positions[clear_indices, 0], positions[clear_indices, 1], 'b.-', 
            label='Clear path', markersize=4)
    ax.plot(positions[obstacle_indices, 0], positions[obstacle_indices, 1], 'r.',
            label='Avoiding obstacle', markersize=6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2D Trajectory (Top View)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # Plot 2: Altitude over time
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Altitude (m, NED)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity commands
    ax = axes[1, 0]
    ax.plot(commands[:, 0], 'r-', label='Forward', linewidth=2)
    ax.plot(commands[:, 1], 'g-', label='Lateral', linewidth=2)
    ax.plot(commands[:, 2], 'b-', label='Vertical', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Commands from Planner')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Speed over time
    ax = axes[1, 1]
    speeds = np.linalg.norm(velocities, axis=1)
    ax.plot(speeds, 'k-', linewidth=2)
    ax.fill_between(range(len(speeds)), speeds, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Total Speed')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integration_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved plot to integration_results.png")
    
    # Show plot
    # plt.show()

except ImportError:
    print("\n4. Matplotlib not available, skipping visualization")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE!")
print("=" * 70)
print("\nWhat happened:")
print("  1. Dynamics model tracked position and velocity")
print("  2. Synthetic depth images simulated obstacles")
print("  3. Planner computed velocity commands from depth")
print("  4. Dynamics executed the commands")
print("  5. Loop repeated for", num_steps, "steps")
print("\nThis is the full pipeline working together!")
print("=" * 70)
