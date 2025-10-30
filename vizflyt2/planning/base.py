"""
Base classes for planning modules.

All planners inherit from BasePlanner and implement compute_action().
This supports both reactive planners (perception → action) and 
trajectory-following planners.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BasePlanner(ABC):
    """
    Base class for all planners.
    
    Planners take in state/perception and output control actions.
    This supports:
    - Reactive planners: depth_image → velocity commands
    - RL agents: observation → action  
    - Trajectory followers: time → reference velocity
    - Hybrid planners: combine multiple strategies
    
    The primary interface is compute_action() which returns control commands.
    """
    
    def __init__(self):
        """Initialize the planner."""
        self._step_count = 0
    
    @abstractmethod
    def compute_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute control action given current state/perception.
        
        Args:
            **kwargs: Planner-specific inputs (e.g., depth_image, state, time)
        
        Returns:
            Dictionary with control outputs, e.g.:
            {
                'velocity': np.array([vx, vy, vz]),  # Primary control
                'angular_velocity': np.array([wx, wy, wz]),  # Optional
                'position': np.array([x, y, z]),  # Optional reference
                'info': {...}  # Optional debug info
            }
        """
        pass
    
    def reset(self):
        """Reset planner state."""
        self._step_count = 0
    
    def step(self):
        """Increment internal step counter."""
        self._step_count += 1
    
    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count


class ReactiveVisualPlanner(BasePlanner):
    """
    Base class for reactive planners using visual perception.
    
    These planners map: perception (depth/rgb) → control action
    Examples: potential fields, learned policies, rule-based navigation
    """
    
    @abstractmethod
    def compute_action(self, depth_image: Optional[np.ndarray] = None, 
                      rgb_image: Optional[np.ndarray] = None,
                      **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute action from visual perception.
        
        Args:
            depth_image: Depth image (H, W) if available
            rgb_image: RGB image (H, W, 3) if available
            **kwargs: Additional planner-specific inputs
        
        Returns:
            Dictionary with at minimum 'velocity' key
        """
        pass


class TrajectoryPlanner(BasePlanner):
    """
    Base class for trajectory-following planners.
    
    These planners follow pre-computed trajectories and output
    reference commands at each timestep.
    """
    
    def __init__(self):
        super().__init__()
        self.trajectory = None
        self.current_index = 0
        self.dt = 0.01
    
    def is_complete(self) -> bool:
        """Check if trajectory is complete."""
        if self.trajectory is None:
            return True
        return self.current_index >= len(self.trajectory['time'])
    
    def get_trajectory(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the full planned trajectory."""
        return self.trajectory
    
    def reset(self):
        """Reset to start of trajectory."""
        super().reset()
        self.current_index = 0
