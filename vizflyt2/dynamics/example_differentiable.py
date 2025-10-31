"""
Example: Differentiable Quadrotor Dynamics

Demonstrates the PyTorch-based differentiable dynamics model with:
- Gradient flow through physics
- Realistic controller delay and filtering
- Drag effects
- Attitude computation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from vizflyt2.dynamics import DifferentiableQuadrotorDynamics

# ============================================================================
# Example 1: Basic Usage (Compatible with VizFlyt2 Interface)
# ============================================================================

print("=" * 70)
print("Example 1: Basic Simulation with Differentiable Dynamics")
print("=" * 70)

# Initialize dynamics
dynamics = DifferentiableQuadrotorDynamics(
    initial_state={
        'position': np.array([0., 0., -50.]),
        'velocity': np.array([2., 0., 0.]),
        'orientation_rpy': np.array([0., 0., 0.]),
    },
    dt=0.1,
    mass=1.0,
    drag_quad=0.02,
    ctrl_delay_sec=0.06,
    ctrl_alpha=0.25,
    action_is_normalized=False,
)

# Simulate
num_steps = 100
dt = 0.1
positions = []
velocities = []
attitudes = []

for i in range(num_steps):
    # Simple constant acceleration command
    controls = {
        'acceleration': np.array([0., 0., 1.]),  # Slight upward thrust
        'frame': 'world'
    }
    
    state = dynamics.step(controls, dt)
    
    positions.append(state['position'].copy())
    velocities.append(state['velocity'].copy())
    attitudes.append(state['orientation_rpy'].copy())
    
    if i % 20 == 0:
        print(f"Step {i:3d}: pos={state['position']}, vel={state['velocity']}")

positions = np.array(positions)
velocities = np.array(velocities)
attitudes = np.array(attitudes)

print(f"\nFinal position: {positions[-1]}")
print(f"Final velocity: {velocities[-1]}")

# ============================================================================
# Example 2: Gradient-Based Optimization
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: Gradient-Based Trajectory Optimization")
print("=" * 70)

# Reset
dynamics.reset()

# Define a simple optimization problem: reach target position
target_pos = torch.tensor([20., 10., -45.], device=dynamics.device, dtype=dynamics.dtype)

# Initialize action sequence (learnable parameters)
n_opt_steps = 50
actions = torch.zeros(n_opt_steps, 3, device=dynamics.device, dtype=dynamics.dtype, requires_grad=True)

optimizer = torch.optim.Adam([actions], lr=0.5)

# Optimize
n_iterations = 100
for iteration in range(n_iterations):
    optimizer.zero_grad()
    
    # Reset to initial state
    p0 = torch.tensor([0., 0., -50.], device=dynamics.device, dtype=dynamics.dtype)
    v0 = torch.tensor([0., 0., 0.], device=dynamics.device, dtype=dynamics.dtype)
    att0 = torch.tensor([0., 0., 0.], device=dynamics.device, dtype=dynamics.dtype)
    dynamics.physics.reset(p0, v0, att0)
    
    # Simulate with current action sequence
    final_pos = None
    for t in range(n_opt_steps):
        result = dynamics.physics.step(
            action_cmd=actions[t],
            heading_xy=None,
            add_gravity=True,
            frame='world'
        )
        final_pos = result['p_next']
    
    # Loss: distance to target
    loss = torch.norm(final_pos - target_pos)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration:3d}: loss={loss.item():.4f}, final_pos={final_pos.detach().cpu().numpy()}")

print(f"\nTarget position: {target_pos.cpu().numpy()}")
print(f"Reached position: {final_pos.detach().cpu().numpy()}")
print(f"Error: {torch.norm(final_pos - target_pos).item():.4f} m")

# ============================================================================
# Example 3: Body-Frame Control
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Body-Frame Control Commands")
print("=" * 70)

# Reset
dynamics.reset({
    'position': np.array([0., 0., -50.]),
    'velocity': np.array([0., 0., 0.]),
    'orientation_rpy': np.array([0., 0., np.pi/4]),  # 45 degree yaw
})

# Apply forward thrust in body frame
for i in range(50):
    controls = {
        'acceleration': np.array([5., 0., 0.]),  # Forward in body frame
        'frame': 'body'
    }
    state = dynamics.step(controls, dt)
    
    if i % 10 == 0:
        print(f"Step {i:3d}: pos={state['position']}, att(deg)={np.degrees(state['orientation_rpy'])}")

print(f"\nFinal position (should move NE at 45°): {state['position']}")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "=" * 70)
print("Creating Visualization...")
print("=" * 70)

# Simple trajectory plot from Example 1
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position
axes[0, 0].plot(positions[:, 0], label='X (North)')
axes[0, 0].plot(positions[:, 1], label='Y (East)')
axes[0, 0].plot(positions[:, 2], label='Z (Down)')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].set_title('Position over Time')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Velocity
axes[0, 1].plot(velocities[:, 0], label='Vx')
axes[0, 1].plot(velocities[:, 1], label='Vy')
axes[0, 1].plot(velocities[:, 2], label='Vz')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Velocity (m/s)')
axes[0, 1].set_title('Velocity over Time')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Attitude
axes[1, 0].plot(np.degrees(attitudes[:, 0]), label='Roll')
axes[1, 0].plot(np.degrees(attitudes[:, 1]), label='Pitch')
axes[1, 0].plot(np.degrees(attitudes[:, 2]), label='Yaw')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].set_title('Attitude over Time')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 3D Trajectory
ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label='End')
ax_3d.set_xlabel('X (North)')
ax_3d.set_ylabel('Y (East)')
ax_3d.set_zlabel('Z (Down)')
ax_3d.set_title('3D Trajectory')
ax_3d.legend()

plt.tight_layout()
plt.savefig('differentiable_dynamics_example.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to differentiable_dynamics_example.png")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
