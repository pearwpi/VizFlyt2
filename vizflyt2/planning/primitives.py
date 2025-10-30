"""
Simple trajectory primitives.

Pre-built trajectory generators for common patterns.
"""

import numpy as np
from typing import Dict, List, Tuple


def line_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    dt: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Straight line trajectory.
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        duration: Duration (s)
        dt: Time step (s)
    
    Returns:
        Dict with 'time', 'position', 'velocity'
    """
    num_steps = int(duration / dt)
    time = np.linspace(0, duration, num_steps)
    
    # Linear interpolation
    positions = np.outer(1 - time/duration, start) + np.outer(time/duration, end)
    
    # Constant velocity
    velocity = (end - start) / duration
    velocities = np.tile(velocity, (num_steps, 1))
    
    return {
        'time': time,
        'position': positions,
        'velocity': velocities
    }


def circle_trajectory(
    center: np.ndarray,
    radius: float,
    duration: float,
    dt: float = 0.01,
    angular_velocity: float = None
) -> Dict[str, np.ndarray]:
    """
    Circular trajectory in horizontal plane.
    
    Args:
        center: Circle center [x, y, z]
        radius: Circle radius (m)
        duration: Duration (s)
        dt: Time step (s)
        angular_velocity: Angular velocity (rad/s), defaults to one full circle
    
    Returns:
        Dict with 'time', 'position', 'velocity'
    """
    num_steps = int(duration / dt)
    time = np.linspace(0, duration, num_steps)
    
    if angular_velocity is None:
        angular_velocity = 2 * np.pi / duration
    
    theta = angular_velocity * time
    
    positions = np.zeros((num_steps, 3))
    positions[:, 0] = center[0] + radius * np.cos(theta)
    positions[:, 1] = center[1] + radius * np.sin(theta)
    positions[:, 2] = center[2]
    
    velocities = np.zeros((num_steps, 3))
    velocities[:, 0] = -radius * angular_velocity * np.sin(theta)
    velocities[:, 1] = radius * angular_velocity * np.cos(theta)
    velocities[:, 2] = 0
    
    return {
        'time': time,
        'position': positions,
        'velocity': velocities
    }


def figure8_trajectory(
    center: np.ndarray,
    size: float,
    duration: float,
    dt: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Figure-8 trajectory in horizontal plane.
    
    Args:
        center: Center point [x, y, z]
        size: Size of figure-8 (m)
        duration: Duration for one complete figure-8 (s)
        dt: Time step (s)
    
    Returns:
        Dict with 'time', 'position', 'velocity'
    """
    num_steps = int(duration / dt)
    time = np.linspace(0, duration, num_steps)
    
    omega = 2 * np.pi / duration
    
    positions = np.zeros((num_steps, 3))
    positions[:, 0] = center[0] + size * np.sin(omega * time)
    positions[:, 1] = center[1] + size * np.sin(2 * omega * time) / 2
    positions[:, 2] = center[2]
    
    velocities = np.zeros((num_steps, 3))
    velocities[:, 0] = size * omega * np.cos(omega * time)
    velocities[:, 1] = size * omega * np.cos(2 * omega * time)
    velocities[:, 2] = 0
    
    return {
        'time': time,
        'position': positions,
        'velocity': velocities
    }


def spiral_trajectory(
    center: np.ndarray,
    radius_start: float,
    radius_end: float,
    height_change: float,
    duration: float,
    dt: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Spiral trajectory (changing radius and altitude).
    
    Args:
        center: Spiral center [x, y, z]
        radius_start: Starting radius (m)
        radius_end: Ending radius (m)
        height_change: Change in altitude (m, positive is up)
        duration: Duration (s)
        dt: Time step (s)
    
    Returns:
        Dict with 'time', 'position', 'velocity'
    """
    num_steps = int(duration / dt)
    time = np.linspace(0, duration, num_steps)
    
    omega = 2 * np.pi / duration
    theta = omega * time
    
    # Linearly change radius and height
    radius = radius_start + (radius_end - radius_start) * time / duration
    height = center[2] - height_change * time / duration
    
    positions = np.zeros((num_steps, 3))
    positions[:, 0] = center[0] + radius * np.cos(theta)
    positions[:, 1] = center[1] + radius * np.sin(theta)
    positions[:, 2] = height
    
    # Compute velocities numerically
    velocities = np.zeros((num_steps, 3))
    velocities[:-1] = np.diff(positions, axis=0) / dt
    velocities[-1] = velocities[-2]  # Repeat last
    
    return {
        'time': time,
        'position': positions,
        'velocity': velocities
    }


def waypoint_trajectory(
    waypoints: List[np.ndarray],
    speeds: List[float] = None,
    dt: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Trajectory through waypoints with linear segments.
    
    Args:
        waypoints: List of waypoint positions [x, y, z]
        speeds: Speed for each segment (m/s), defaults to 10 m/s
        dt: Time step (s)
    
    Returns:
        Dict with 'time', 'position', 'velocity'
    """
    waypoints = [np.array(w) for w in waypoints]
    
    if speeds is None:
        speeds = [10.0] * (len(waypoints) - 1)
    
    # Compute segment durations
    segments = []
    for i in range(len(waypoints) - 1):
        distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
        duration = distance / speeds[i]
        seg = line_trajectory(waypoints[i], waypoints[i+1], duration, dt)
        segments.append(seg)
    
    # Concatenate segments
    time = np.concatenate([seg['time'] + (0 if i == 0 else segments[i-1]['time'][-1] + dt) 
                          for i, seg in enumerate(segments)])
    position = np.concatenate([seg['position'] for seg in segments])
    velocity = np.concatenate([seg['velocity'] for seg in segments])
    
    return {
        'time': time,
        'position': position,
        'velocity': velocity
    }
