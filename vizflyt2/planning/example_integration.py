"""
Complete Integration Example: Perception + Planning + Dynamics

This example demonstrates the full VizFlyt2 pipeline:
1. Perception: Depth images (real renderer or synthetic simulation)
2. Planning: PotentialFieldPlanner computes velocity commands from depth
3. Dynamics: PointMassDynamics executes the commands
4. Visualization: Frame-by-frame and animated outputs

Mode Selection:
- By default, uses SYNTHETIC depth images (no Gaussian Splat needed)
- To use REAL Gaussian Splat rendering:
  1. Train a splatfacto model using Nerfstudio
  2. Update SPLAT_CONFIG_PATH and CAMERA_JSON_PATH below
  3. Ensure nerfstudio is installed: pip install nerfstudio

Outputs generated in outputs/integration/:
- integration_summary.png: 4-panel summary (trajectory, altitude, velocities, speed)
- frames/frame_XXXX.png: Individual frames showing depth + trajectory
- integration_animation.gif: Animated visualization of the simulation
- integration_animation.mp4: Video version (if ffmpeg available)

Dependencies:
- matplotlib (required for plots)
- imageio (optional, for animation)
- nerfstudio (optional, for real rendering)

Run from planning directory:
    python example_integration.py

Example renderer paths:
    SPLAT_CONFIG_PATH = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
    CAMERA_JSON_PATH = "../splats/cam_settings.json"
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work from planning/ directory
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import cv2
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
# Configuration
# ============================================================================

# Renderer configuration - update these paths to match your setup
# These are the same paths used in perception examples

# Boxes
# SPLAT_CONFIG_PATH = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
# CAMERA_JSON_PATH = "../splats/boxes.json"

# Racing
# SPLAT_CONFIG_PATH = "../splats/washburn-env6-itr0-1fps/washburn-env6-itr0-1fps_nf_format/splatfacto/2025-03-06_201843/config.yml"
# CAMERA_JSON_PATH = "../splats/racing.json"

# Science
SPLAT_CONFIG_PATH = "../splats/saranga_rebuttal_colmap_splat/saranga_rebuttal_colmap/splatfacto/2025-10-26_191918/config.yml"
CAMERA_JSON_PATH = "../splats/science.json"

# ============================================================================
# Setup
# ============================================================================

print("\n1. Setting up components...")

# Initial state
initial_state = {
    'position': np.array([0., 0., 0]),      # Start at origin
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
    step_size=0.02,          # 5 cm/s forward speed
    safety_radius=0.3,      #  
    threshold=240,           # Obstacle detection threshold
    neighborhood_size=(60, 60),  # Neighborhood size for potential field (px)
    verbose=False           # Quiet mode
)
print(f"   ✓ Planner initialized: step_size={planner.step_size} m/s")

# Perception (optional - uses synthetic depth if not available)
renderer = None
if HAS_RENDERER:
    try:
        renderer = SplatRenderer(SPLAT_CONFIG_PATH, CAMERA_JSON_PATH)
        print("   ✓ Renderer initialized")
    except Exception as e:
        print(f"   ⚠ Could not initialize renderer: {e}")
        exit()


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

num_steps = 750
dt = 0.1

# Storage for logging
trajectory = []
depth_images = []  # Store depth images for visualization
rgb_images = []    # Store RGB images for visualization
debug_images = []  # Store potential field debug visualizations

for step in range(num_steps):
    # -----------------------------------------------------------------------
    # 1. PERCEPTION: Get depth image
    # -----------------------------------------------------------------------
    state = dynamics.get_state()
    position = state['position']
    orientation_rpy = state['orientation_rpy']
    
    if renderer is not None:
        # Real renderer (if available and initialized)
        print(f"   Step {step:3d}: Rendering depth and RGB")
        position_render, orientation_render = dynamics.get_render_params()
        result = renderer.render(position_render, orientation_render)
        depth = result['depth_raw']
        rgb = result['rgb']
    else:
        # Synthetic depth and RGB
        print(f"   Step {step:3d}: Generating synthetic depth and RGB")
        depth = generate_synthetic_depth(position, step)
        # Create synthetic RGB (simple gradient based on depth)
        rgb = np.stack([depth, depth*0.8, depth*0.6], axis=-1).astype(np.uint8)

    # Handle depth image shape - squeeze if (h, w, 1)
    if len(depth.shape) == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(axis=2)
    
    depth_images.append(depth.copy())
    rgb_images.append(rgb.copy())
    
    # -----------------------------------------------------------------------
    # 2. PLANNING: Compute velocity command from depth
    # -----------------------------------------------------------------------

    # depth needs to be relative (not metric) for planner
    depth_rel = np.clip((np.max(depth) - depth) / np.max(depth) * 255, 0, 255).astype(np.uint8)
    action = planner.compute_action(depth_image=depth_rel, save_visualization=True)
    velocity_cmd = action['velocity']
    
    # Store debug visualization if available
    debug_viz = action.get('info', {}).get('visualization', None)
    debug_images.append(debug_viz)
    
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
# ============================================================================
# Optional: Visualization
# ============================================================================

try:
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print("\n4. Generating visualizations...")
    
    # Create output directory
    output_dir = Path('outputs/integration')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Summary plots
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: 2D trajectory
    ax = axes[0, 0]
    obstacle_indices = [i for i, t in enumerate(trajectory) if t['has_obstacle']]
    clear_indices = [i for i, t in enumerate(trajectory) if not t['has_obstacle']]
    
    ax.plot(positions[clear_indices, 0], positions[clear_indices, 1], 'b.-', 
            label='Clear path', markersize=4)
    ax.plot(positions[obstacle_indices, 0], positions[obstacle_indices, 1], 'r.',
            label='Avoiding obstacle', markersize=6)
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=10, label='End')
    ax.set_xlabel('X - North (m)')
    ax.set_ylabel('Y - East (m)')
    ax.set_title('2D Trajectory (Top View)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # Plot 2: Altitude over time
    ax = axes[0, 1]
    ax.plot(-positions[:, 2], 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Altitude (m, AGL)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.legend()
    
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
    summary_path = output_dir / 'integration_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved summary plot to {summary_path}")
    plt.close()
    
    # ========================================================================
    # Frame-by-frame visualization
    # ========================================================================
    print("\n5. Generating frame-by-frame visualization...")
    
    # Sample every N frames to avoid too many images
    frame_sample_rate = 5
    sampled_frames = range(0, len(trajectory), frame_sample_rate)
    
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    for i in sampled_frames:
        fig = plt.figure(figsize=(24, 5))
        
        # Panel 1: RGB image
        ax1 = plt.subplot(1, 4, 1)
        if len(rgb_images[i].shape) == 3:
            # RGB image
            ax1.imshow(cv2.cvtColor(rgb_images[i], cv2.COLOR_BGR2RGB))
        else:
            # Grayscale
            ax1.imshow(rgb_images[i], cmap='gray')
        ax1.set_title(f'RGB - Step {i}', fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Depth image
        ax2 = plt.subplot(1, 4, 2)
        depth_display = depth_images[i]
        # Auto-scale depth for better visualization
        depth_min, depth_max = np.min(depth_display), np.max(depth_display)
        im = ax2.imshow(depth_display, cmap='gray', vmin=depth_min, vmax=depth_max)
        ax2.set_title(f'Depth - {"OBSTACLE" if trajectory[i]["has_obstacle"] else "Clear"}',
                     color='red' if trajectory[i]["has_obstacle"] else 'green',
                     fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Depth', fraction=0.046, pad=0.04)
        
        # Panel 3: Potential Field Debug
        ax3 = plt.subplot(1, 4, 3)
        if debug_images[i] is not None:
            # Debug visualization from planner
            debug_rgb = cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB)
            ax3.imshow(debug_rgb)
            ax3.set_title('Potential Field Debug', fontweight='bold')
        else:
            # Show depth if no debug available
            ax3.imshow(depth_display, cmap='gray')
            ax3.set_title('No Debug Available', fontweight='bold')
        ax3.axis('off')
        
        # Panel 4: Trajectory plot
        ax4 = plt.subplot(1, 4, 4)
        
        # Plot full trajectory in gray
        ax4.plot(positions[:, 0], positions[:, 1], 'gray', alpha=0.3, linewidth=1, label='Full path')
        
        # Plot past trajectory
        if i > 0:
            past_positions = positions[:i+1]
            past_obstacle = [trajectory[j]['has_obstacle'] for j in range(i+1)]
            
            for j in range(len(past_positions)-1):
                color = 'red' if past_obstacle[j] else 'blue'
                ax4.plot(past_positions[j:j+2, 0], past_positions[j:j+2, 1], 
                        color=color, linewidth=2, alpha=0.7)
        
        # Current position
        curr_pos = positions[i]
        ax4.plot(curr_pos[0], curr_pos[1], 'go', markersize=15, 
                label=f'Current ({curr_pos[0]:.1f}, {curr_pos[1]:.1f})')
        
        # Velocity vector
        vel_cmd = trajectory[i]['velocity_cmd']
        scale = 2.0
        ax4.arrow(curr_pos[0], curr_pos[1], 
                 vel_cmd[0]*scale, vel_cmd[1]*scale,
                 head_width=.1, head_length=0.1, fc='orange', ec='orange',
                 linewidth=2, label='Velocity cmd')
        
        ax4.set_xlabel('X - North (m)')
        ax4.set_ylabel('Y - East (m)')
        ax4.set_title(f'Trajectory - Step {i}/{len(trajectory)-1}')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.axis('equal')
        
        # Add velocity info
        info_text = f"Velocity: [{vel_cmd[0]:.2f}, {vel_cmd[1]:.2f}, {vel_cmd[2]:.2f}] m/s\n"
        info_text += f"Speed: {np.linalg.norm(vel_cmd):.2f} m/s\n"
        info_text += f"Altitude: {-curr_pos[2]:.1f} m"
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        frame_path = frames_dir / f'frame_{i:04d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if i % (frame_sample_rate * 4) == 0:
            print(f"   Saved frame {i}/{len(trajectory)-1}")
    
    print(f"   ✓ Saved {len(sampled_frames)} frames to {frames_dir}")
    
    # ========================================================================
    # Create video/GIF
    # ========================================================================
    print("\n6. Creating animation...")
    
    try:
        import imageio.v2 as imageio
        
        # Load all frames
        frame_files = sorted(frames_dir.glob('frame_*.png'))
        frames = [imageio.imread(f) for f in frame_files]
        
        # Ensure all frames have consistent shape (normalize to RGB with same dimensions)
        if len(frames) > 0:
            # Get target shape from first frame
            first_frame = frames[0]
            if len(first_frame.shape) == 3 and first_frame.shape[2] == 4:
                first_frame = first_frame[:, :, :3]
            elif len(first_frame.shape) == 2:
                first_frame = np.stack([first_frame]*3, axis=-1)
            target_shape = first_frame.shape
            
            normalized_frames = [first_frame]
            for frame in frames[1:]:
                # Convert RGBA to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                # Ensure RGB (some might be grayscale)
                elif len(frame.shape) == 2:
                    frame = np.stack([frame]*3, axis=-1)
                
                # Resize if dimensions don't match
                if frame.shape != target_shape:
                    frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
                    # Ensure it's 3-channel after resize
                    if len(frame.shape) == 2:
                        frame = np.stack([frame]*3, axis=-1)
                
                normalized_frames.append(frame)
            frames = normalized_frames
        
        # Save as GIF
        gif_path = output_dir / 'integration_animation.gif'
        imageio.mimsave(gif_path, frames, duration=0.2, loop=0)
        print(f"   ✓ Saved animation to {gif_path}")
        
        # Try to save as MP4 if ffmpeg is available
        try:
            mp4_path = output_dir / 'integration_animation.mp4'
            imageio.mimsave(mp4_path, frames, fps=5)
            print(f"   ✓ Saved video to {mp4_path}")
        except Exception as e:
            print(f"   ⚠ Could not create MP4 (ffmpeg may not be installed): {e}")
    
    except ImportError:
        print("   ⚠ imageio not available, skipping animation creation")
        print("   Install with: pip install imageio")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    
except ImportError as e:
    print(f"\n4. Visualization libraries not available: {e}")
    print("   Install with: pip install matplotlib imageio")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE!")
print("=" * 70)
print("\nWhat happened:")
print("  1. Dynamics model tracked position and velocity")
print("  2. Synthetic depth images simulated obstacles")
print("  3. Planner computed velocity commands from depth")
print("  4. Dynamics executed the commands")
print("  5. Loop repeated for", num_steps, "steps")
print("\nOutputs created:")
print("  📊 Summary plot: outputs/integration/integration_summary.png")
print("  🎞️  Frame images: outputs/integration/frames/")
print("  🎬 Animation GIF: outputs/integration/integration_animation.gif")
print("  🎥 Video (MP4): outputs/integration/integration_animation.mp4")
print("\nThis is the full VizFlyt2 pipeline working together!")
print("=" * 70)
