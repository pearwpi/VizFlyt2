"""
Example: Simple trajectory planning.

Shows how to use trajectory primitives and the planner.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from planning import TrajectoryPlanner, circle_trajectory, figure8_trajectory


def example_primitives():
    """Use trajectory primitives directly."""
    print("\nTrajectory Primitives")
    print("-" * 40)
    
    # Circle
    traj = circle_trajectory(
        center=np.array([0., 0., -50.]),
        radius=20.0,
        duration=10.0
    )
    print(f"✓ Circle: {len(traj['time'])} points over {traj['time'][-1]:.1f}s")
    
    # Figure-8
    traj = figure8_trajectory(
        center=np.array([0., 0., -50.]),
        size=15.0,
        duration=20.0
    )
    print(f"✓ Figure-8: {len(traj['time'])} points over {traj['time'][-1]:.1f}s")
    
    return traj


def example_planner():
    """Use the TrajectoryPlanner class."""
    print("\nTrajectory Planner")
    print("-" * 40)
    
    planner = TrajectoryPlanner(dt=0.01)
    
    # Plan a circle
    planner.plan_circle(
        center=np.array([0., 0., -50.]),
        radius=25.0,
        duration=15.0
    )
    
    print(f"✓ Planned circle: {planner.get_duration():.1f}s duration")
    
    # Get state at specific time
    state = planner.get_state_at_time(5.0)
    print(f"  Position at t=5s: {state['position']}")
    
    # Iterate through trajectory
    planner.reset()
    positions = []
    while not planner.is_complete():
        state = planner.get_next_state()
        if state is not None:
            positions.append(state['position'])
    
    positions = np.array(positions)
    print(f"✓ Iterated through {len(positions)} waypoints")
    
    return positions


def example_waypoints():
    """Plan trajectory through waypoints."""
    print("\nWaypoint Planning")
    print("-" * 40)
    
    planner = TrajectoryPlanner(dt=0.01)
    
    waypoints = [
        np.array([0., 0., -50.]),
        np.array([50., 0., -50.]),
        np.array([50., 50., -30.]),
        np.array([0., 50., -30.]),
        np.array([0., 0., -50.])
    ]
    
    planner.plan_waypoints(waypoints, speeds=[10., 10., 10., 10.])
    
    print(f"✓ Planned {len(waypoints)} waypoints")
    print(f"  Duration: {planner.get_duration():.1f}s")
    
    return planner.get_trajectory()


def example_with_dynamics():
    """Use planning with dynamics."""
    print("\nPlanning + Dynamics")
    print("-" * 40)
    
    try:
        from dynamics import PointMassDynamics
        
        planner = TrajectoryPlanner(dt=0.01)
        planner.plan_circle(
            center=np.array([0., 0., -50.]),
            radius=20.0,
            duration=10.0
        )
        
        # Initialize dynamics
        initial_state = {
            'position': np.array([20., 0., -50.]),
            'velocity': np.array([0., 12.6, 0.]),
            'orientation_rpy': np.array([0., 0., np.pi/2])
        }
        
        dynamics = PointMassDynamics(
            initial_state=initial_state,
            control_mode='velocity'
        )
        
        # Follow planned trajectory
        planner.reset()
        actual_positions = []
        
        while not planner.is_complete():
            planned = planner.get_next_state()
            if planned is None:
                break
            
            # Command dynamics to follow planned velocity
            dynamics.step({'velocity': planned['velocity']}, dt=0.01)
            actual_positions.append(dynamics.get_state()['position'].copy())
        
        actual_positions = np.array(actual_positions)
        print(f"✓ Executed trajectory with dynamics")
        print(f"  Traveled {len(actual_positions)} steps")
        
        return actual_positions
        
    except ImportError:
        print("  (Dynamics module not available, skipping)")
        return None


def plot_examples(fig8_traj, circle_pos, waypoint_traj, dynamics_pos):
    """Plot all examples."""
    fig = plt.figure(figsize=(14, 10))
    
    # Figure-8
    ax = fig.add_subplot(221)
    ax.plot(fig8_traj['position'][:, 0], fig8_traj['position'][:, 1], 'b-', linewidth=2)
    ax.plot(fig8_traj['position'][0, 0], fig8_traj['position'][0, 1], 'go', markersize=10)
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_title('Figure-8 Primitive')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Circle from planner
    ax = fig.add_subplot(222)
    ax.plot(circle_pos[:, 0], circle_pos[:, 1], 'r-', linewidth=2)
    ax.plot(circle_pos[0, 0], circle_pos[0, 1], 'go', markersize=10)
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_title('Circle (TrajectoryPlanner)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Waypoints
    ax = fig.add_subplot(223, projection='3d')
    ax.plot(waypoint_traj['position'][:, 0], 
            waypoint_traj['position'][:, 1], 
            -waypoint_traj['position'][:, 2], 'g-', linewidth=2)
    ax.plot([waypoint_traj['position'][0, 0]], 
            [waypoint_traj['position'][0, 1]], 
            [-waypoint_traj['position'][0, 2]], 'go', markersize=10)
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Waypoint Trajectory')
    
    # Dynamics
    if dynamics_pos is not None:
        ax = fig.add_subplot(224)
        ax.plot(dynamics_pos[:, 0], dynamics_pos[:, 1], 'm-', linewidth=2)
        ax.plot(dynamics_pos[0, 0], dynamics_pos[0, 1], 'go', markersize=10)
        ax.set_xlabel('North (m)')
        ax.set_ylabel('East (m)')
        ax.set_title('Planning + Dynamics')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    
    save_path = Path('outputs/planning_examples.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved to {save_path}")


if __name__ == '__main__':
    print("=" * 50)
    print("Simple Trajectory Planning Examples")
    print("=" * 50)
    
    fig8_traj = example_primitives()
    circle_pos = example_planner()
    waypoint_traj = example_waypoints()
    dynamics_pos = example_with_dynamics()
    
    plot_examples(fig8_traj, circle_pos, waypoint_traj, dynamics_pos)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
