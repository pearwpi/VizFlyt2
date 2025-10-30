"""
Simple point-mass dynamics.

Just position, velocity, orientation, angular velocity. That's it.
"""

from typing import Dict, Tuple, Literal
import numpy as np
from .base import DynamicsModel


class PointMassDynamics(DynamicsModel):
    """
    Simple point-mass dynamics.
    
    State:
        position: [x, y, z] in NED (m)
        velocity: [vx, vy, vz] in NED (m/s)
        orientation_rpy: [roll, pitch, yaw] (rad)
        angular_velocity: [wx, wy, wz] (rad/s)
    
    Control Modes:
        'velocity': Set velocity directly (kinematic, no physics)
        'acceleration': Set acceleration (optional gravity)
    """
    
    def __init__(
        self,
        initial_state: Dict[str, np.ndarray],
        control_mode: Literal['velocity', 'acceleration'] = 'acceleration',
        gravity: bool = False,
    ):
        """
        Args:
            initial_state: Dict with 'position', 'velocity', 'orientation_rpy'
            control_mode: 'velocity' or 'acceleration'
            gravity: Add gravity (9.81 m/s² down) in acceleration mode
        """
        if 'angular_velocity' not in initial_state:
            initial_state['angular_velocity'] = np.zeros(3)
        
        super().__init__(initial_state)
        self.control_mode = control_mode
        self.gravity = gravity
    
    def step(self, controls: Dict[str, np.ndarray], dt: float) -> Dict[str, np.ndarray]:
        """
        Step simulation forward.
        
        Args:
            controls:
                velocity mode: {'velocity': [vx,vy,vz], 'angular_velocity': [wx,wy,wz]}
                acceleration mode: {'acceleration': [ax,ay,az], 'angular_velocity': [wx,wy,wz]}
            dt: Time step (s)
        """
        if self.control_mode == 'velocity':
            # Just set velocity directly
            self.state['velocity'] = np.array(controls.get('velocity', self.state['velocity']))
            self.state['angular_velocity'] = np.array(controls.get('angular_velocity', self.state['angular_velocity']))
            
        else:  # acceleration
            accel = np.array(controls.get('acceleration', np.zeros(3)))
            if self.gravity:
                accel += np.array([0, 0, 9.81])  # Add gravity
            
            self.state['velocity'] += accel * dt
            self.state['angular_velocity'] = np.array(controls.get('angular_velocity', self.state['angular_velocity']))
        
        # Integrate
        self.state['position'] += self.state['velocity'] * dt
        self.state['orientation_rpy'] += self.state['angular_velocity'] * dt
        
        # Wrap angles
        self.state['orientation_rpy'] = np.arctan2(
            np.sin(self.state['orientation_rpy']), 
            np.cos(self.state['orientation_rpy'])
        )
        
        self.time += dt
        return self.state.copy()
    
    def get_render_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and orientation for rendering."""
        return self.state['position'].copy(), self.state['orientation_rpy'].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.state['velocity'].copy()
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity."""
        return self.state['angular_velocity'].copy()
    
    def get_speed(self) -> float:
        """Get speed magnitude."""
        return np.linalg.norm(self.state['velocity'])
    
    def get_altitude_agl(self) -> float:
        """Get altitude (negative of z in NED)."""
        return -self.state['position'][2]
