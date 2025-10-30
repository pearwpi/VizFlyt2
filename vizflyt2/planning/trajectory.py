"""
Trajectory-following planner.

Generates smooth trajectories using primitives and outputs reference
velocity commands for trajectory tracking.
"""

import numpy as np
from typing import Optional, Dict
from .primitives import (
    line_trajectory,
    circle_trajectory,
    figure8_trajectory,
    spiral_trajectory,
    waypoint_trajectory
)
from .base import TrajectoryPlanner as BaseTrajectoryPlanner


class TrajectoryPlanner(BaseTrajectoryPlanner):
    """
    Trajectory-following planner using pre-computed primitives.
    
    Plans trajectories using primitives (line, circle, figure-8, etc.) and
    outputs reference velocity commands at each timestep.
    
    Args:
        dt: Time step for trajectory sampling (default: 0.01)
    
    Example:
        ```python
        planner = TrajectoryPlanner(dt=0.01)
        planner.plan_circle(center=np.array([0., 0., -50.]), radius=20., duration=10.)
        
        planner.reset()
        while not planner.is_complete():
            action = planner.compute_action()
            velocity = action['velocity']
            position_ref = action['position']  # Optional reference
            
            # Send to dynamics
            dynamics.set_control(velocity)
            dynamics.step()
            planner.step()
        ```
    """
    
    def __init__(self, dt: float = 0.01):
        super().__init__()
        self.dt = dt
    
    def compute_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Get next trajectory waypoint as control action.
        
        Args:
            **kwargs: Not used, for interface compatibility
        
        Returns:
            Dictionary with:
                'velocity': Reference velocity at current time
                'position': Reference position at current time (optional)
                'time': Current time in trajectory
        """
        if self.trajectory is None or self.is_complete():
            return {
                'velocity': np.zeros(3),
                'position': np.zeros(3),
                'time': 0.0
            }
        
        idx = self.current_index
        return {
            'velocity': self.trajectory['velocity'][idx],
            'position': self.trajectory['position'][idx],
            'time': self.trajectory['time'][idx]
        }
    
    def get_next_state(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get next state and advance trajectory (legacy interface).
        
        Returns:
            Dictionary with 'position', 'velocity', 'time' or None if complete
        """
        if self.is_complete():
            return None
        
        action = self.compute_action()
        self.current_index += 1
        self.step()
        return action if action['time'] is not None else None
    
    def get_state_at_time(self, t: float) -> Optional[Dict[str, np.ndarray]]:
        """
        Get trajectory state at specific time (legacy interface).
        
        Args:
            t: Time in seconds
        
        Returns:
            Dictionary with 'position', 'velocity', 'time' or None
        """
        if self.trajectory is None:
            return None
        
        # Find closest time index
        time_array = self.trajectory['time']
        idx = np.argmin(np.abs(time_array - t))
        
        return {
            'position': self.trajectory['position'][idx],
            'velocity': self.trajectory['velocity'][idx],
            'time': self.trajectory['time'][idx]
        }
    
    def get_state_at_index(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Get trajectory state at specific index."""
        if self.trajectory is None or idx >= len(self.trajectory['time']):
            return None
        
        return {
            'position': self.trajectory['position'][idx],
            'velocity': self.trajectory['velocity'][idx],
            'time': self.trajectory['time'][idx]
        }
    
    def get_duration(self) -> float:
        """
        Get total duration of the planned trajectory.
        
        Returns:
            Duration in seconds, or 0.0 if no trajectory is planned
        """
        if self.trajectory is None or len(self.trajectory['time']) == 0:
            return 0.0
        return self.trajectory['time'][-1]
    
    # Planning methods
    
    def plan_line(self, start: np.ndarray, end: np.ndarray, duration: float):
        """Plan straight line trajectory."""
        self.trajectory = line_trajectory(start, end, duration, self.dt)
        self.current_index = 0
        return self
    
    def plan_circle(
        self,
        center: np.ndarray,
        radius: float,
        duration: float,
        angular_velocity: float = None
    ):
        """Plan circular trajectory."""
        self.trajectory = circle_trajectory(
            center, radius, duration, self.dt, angular_velocity
        )
        self.current_index = 0
        return self
    
    def plan_figure8(self, center: np.ndarray, size: float, duration: float):
        """Plan figure-8 trajectory."""
        self.trajectory = figure8_trajectory(center, size, duration, self.dt)
        self.current_index = 0
        return self
    
    def plan_spiral(
        self,
        center: np.ndarray,
        radius_start: float,
        radius_end: float,
        height_change: float,
        duration: float
    ):
        """Plan spiral trajectory."""
        self.trajectory = spiral_trajectory(
            center, radius_start, radius_end, height_change, duration, self.dt
        )
        self.current_index = 0
        return self
    
    def plan_waypoints(
        self,
        waypoints: np.ndarray,
        speeds: Optional[np.ndarray] = None
    ):
        """
        Plan trajectory through waypoints.
        
        Args:
            waypoints: List/array of waypoint positions [x, y, z]
            speeds: Speed for each segment (m/s), defaults to 10 m/s for all
        """
        self.trajectory = waypoint_trajectory(waypoints, speeds, self.dt)
        self.current_index = 0
        return self
