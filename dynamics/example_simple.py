"""
Simple dynamics examples showing velocity and acceleration modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from dynamics import PointMassDynamics


def velocity_mode_example():
    """Velocity mode - direct kinematic control."""
    print("\nVelocity Mode - Kinematic Circle")
    print("-" * 40)
    
    dynamics = PointMassDynamics(
        initial_state={
            'position': np.array([0.0, 0.0, -10.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation_rpy': np.array([0.0, 0.0, 0.0])
        },
        control_mode='velocity'
    )
    
    trajectory = []
    dt = 0.01
    
    for i in range(1000):
        t = dynamics.get_time()
        
        # Circular motion
        vx = -10 * np.sin(0.5 * t)
        vy = 10 * np.cos(0.5 * t)
        
        dynamics.step({'velocity': np.array([vx, vy, 0.0])}, dt)
        
        if i % 10 == 0:
            trajectory.append(dynamics.get_state()['position'].copy())
    
    trajectory = np.array(trajectory)
    print(f"✓ Final position: [{trajectory[-1,0]:.1f}, {trajectory[-1,1]:.1f}, {trajectory[-1,2]:.1f}]")
    return trajectory


def acceleration_mode_example():
    """Acceleration mode - altitude hold with gravity."""
    print("\nAcceleration Mode - Altitude Hold")
    print("-" * 40)
    
    dynamics = PointMassDynamics(
        initial_state={
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([5.0, 0.0, 0.0]),
            'orientation_rpy': np.array([0.0, 0.0, 0.0])
        },
        control_mode='acceleration',
        gravity=True
    )
    
    trajectory = []
    dt = 0.05
    
    for i in range(500):
        state = dynamics.get_state()
        print(state)
        
        # Simple altitude controller
        altitude = -state['position'][2]
        az = -9.81 - .75 * (10.0 - altitude) - .75 * state['velocity'][2]  # Cancel gravity + correction
        
        dynamics.step({'acceleration': np.array([0.0, 0.0, az])}, dt)
        
        if i % 5 == 0:
            trajectory.append(state['position'].copy())
    
    trajectory = np.array(trajectory)
    print(f"✓ Final altitude: {-trajectory[-1,2]:.1f}m")
    print(f"✓ Final speed: {dynamics.get_speed():.1f}m/s")
    return trajectory


if __name__ == '__main__':
    print("="*50)
    print("Simple Point-Mass Dynamics Examples")
    print("="*50)
    
    traj_vel = velocity_mode_example()
    traj_acc = acceleration_mode_example()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(traj_vel[:, 0], traj_vel[:, 1], 'b-', linewidth=2)
    ax1.plot(traj_vel[0, 0], traj_vel[0, 1], 'go', markersize=10)
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_title('Velocity Mode (Kinematic)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    ax2.plot(traj_acc[:, 0], -traj_acc[:, 2], 'r-', linewidth=2)
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Acceleration Mode (with Gravity)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path('outputs/dynamics_simple.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved to {save_path}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)
