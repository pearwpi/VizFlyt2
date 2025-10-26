"""
Base class for dynamics models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class DynamicsModel(ABC):
    """
    Base class for aircraft dynamics models.
    
    All dynamics models should:
    1. Maintain state (position, velocity, orientation, etc.)
    2. Accept control inputs
    3. Integrate equations of motion
    4. Provide state for rendering
    
    Coordinate System:
    - NED (North-East-Down) frame for positions
    - Body frame for velocities and forces
    - Roll-Pitch-Yaw (Euler angles) for orientation
    """
    
    def __init__(self, initial_state: Dict[str, np.ndarray]):
        """
        Initialize the dynamics model.
        
        Args:
            initial_state: Dictionary containing initial conditions
                Required keys depend on specific model implementation
        """
        self.state = initial_state.copy()
        self.time = 0.0
    
    @abstractmethod
    def step(self, controls: Dict[str, float], dt: float) -> Dict[str, np.ndarray]:
        """
        Advance the simulation by one time step.
        
        Args:
            controls: Dictionary of control inputs (model-specific)
            dt: Time step in seconds
        
        Returns:
            Updated state dictionary
        """
        pass
    
    @abstractmethod
    def get_render_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current position and orientation for rendering.
        
        Returns:
            position: (3,) array [x, y, z] in meters (NED)
            orientation_rpy: (3,) array [roll, pitch, yaw] in radians
        """
        pass
    
    def reset(self, initial_state: Dict[str, np.ndarray]):
        """Reset the model to initial conditions."""
        self.state = initial_state.copy()
        self.time = 0.0
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get the current state dictionary."""
        return self.state.copy()
    
    def get_time(self) -> float:
        """Get the current simulation time."""
        return self.time
