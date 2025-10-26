"""
Utility functions for dynamics simulations.
"""

import numpy as np
from typing import Tuple, List, Dict


def ned_to_enu(position_ned: np.ndarray) -> np.ndarray:
    """
    Convert NED coordinates to ENU (East-North-Up).
    
    Args:
        position_ned: [x, y, z] in NED frame
    
    Returns:
        position_enu: [x, y, z] in ENU frame
    """
    return np.array([position_ned[1], position_ned[0], -position_ned[2]])


def enu_to_ned(position_enu: np.ndarray) -> np.ndarray:
    """
    Convert ENU coordinates to NED (North-East-Down).
    
    Args:
        position_enu: [x, y, z] in ENU frame
    
    Returns:
        position_ned: [x, y, z] in NED frame
    """
    return np.array([position_enu[1], position_enu[0], -position_enu[2]])


def compute_airspeed_components(
    velocity_ned: np.ndarray,
    orientation_rpy: np.ndarray,
    wind_ned: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Compute airspeed components in body frame.
    
    Args:
        velocity_ned: Velocity in NED frame (m/s)
        orientation_rpy: [roll, pitch, yaw] in radians
        wind_ned: Wind velocity in NED frame (m/s)
    
    Returns:
        u: Forward airspeed (m/s)
        v: Right airspeed (m/s)
        w: Down airspeed (m/s)
    """
    if wind_ned is None:
        wind_ned = np.zeros(3)
    
    # Relative velocity
    v_rel = velocity_ned - wind_ned
    
    # Rotation matrix NED to body
    roll, pitch, yaw = orientation_rpy
    
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_ned_to_body = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
        [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]
    ])
    
    # Transform to body frame
    v_body = R_ned_to_body @ v_rel
    
    return v_body[0], v_body[1], v_body[2]


def compute_flight_path_angle(velocity_ned: np.ndarray) -> Tuple[float, float]:
    """
    Compute flight path angle and heading from velocity.
    
    Args:
        velocity_ned: Velocity in NED frame (m/s)
    
    Returns:
        gamma: Flight path angle (rad) - positive is climbing
        chi: Course angle (rad) - heading over ground
    """
    vn, ve, vd = velocity_ned
    v_horizontal = np.sqrt(vn**2 + ve**2)
    
    if v_horizontal > 0.1:
        gamma = np.arctan2(-vd, v_horizontal)
        chi = np.arctan2(ve, vn)
    else:
        gamma = 0.0
        chi = 0.0
    
    return gamma, chi


def create_trajectory_waypoints(
    waypoints_ned: List[np.ndarray],
    num_points: int = 100
) -> np.ndarray:
    """
    Create smooth trajectory through waypoints using linear interpolation.
    
    Args:
        waypoints_ned: List of waypoint positions in NED
        num_points: Number of interpolated points
    
    Returns:
        trajectory: (num_points, 3) array of positions
    """
    waypoints = np.array(waypoints_ned)
    
    # Compute cumulative distance along path
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        distances[i] = distances[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
    
    # Interpolate
    total_distance = distances[-1]
    interp_distances = np.linspace(0, total_distance, num_points)
    
    trajectory = np.zeros((num_points, 3))
    for dim in range(3):
        trajectory[:, dim] = np.interp(interp_distances, distances, waypoints[:, dim])
    
    return trajectory


def compute_turn_rate(velocity: float, bank_angle: float, g: float = 9.81) -> float:
    """
    Compute turn rate for coordinated turn.
    
    Args:
        velocity: Airspeed (m/s)
        bank_angle: Bank angle (rad)
        g: Gravitational acceleration (m/s^2)
    
    Returns:
        turn_rate: Turn rate (rad/s)
    """
    if velocity < 0.1:
        return 0.0
    
    return g * np.tan(bank_angle) / velocity


def compute_turn_radius(velocity: float, bank_angle: float, g: float = 9.81) -> float:
    """
    Compute turn radius for coordinated turn.
    
    Args:
        velocity: Airspeed (m/s)
        bank_angle: Bank angle (rad)
        g: Gravitational acceleration (m/s^2)
    
    Returns:
        radius: Turn radius (m)
    """
    if abs(bank_angle) < 0.01:
        return np.inf
    
    return velocity**2 / (g * np.tan(bank_angle))


def save_trajectory_csv(trajectory: Dict[str, np.ndarray], filename: str):
    """
    Save trajectory data to CSV file.
    
    Args:
        trajectory: Dictionary with 'time', 'position', 'velocity', etc.
        filename: Output CSV filename
    """
    import csv
    
    time = trajectory['time']
    pos = trajectory['position']
    vel = trajectory['velocity']
    rpy = trajectory['orientation_rpy']
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'time', 'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'roll', 'pitch', 'yaw'
        ])
        
        # Data
        for i in range(len(time)):
            writer.writerow([
                time[i],
                pos[i, 0], pos[i, 1], pos[i, 2],
                vel[i, 0], vel[i, 1], vel[i, 2],
                rpy[i, 0], rpy[i, 1], rpy[i, 2]
            ])
    
    print(f"Trajectory saved to {filename}")


def load_trajectory_csv(filename: str) -> Dict[str, np.ndarray]:
    """
    Load trajectory data from CSV file.
    
    Args:
        filename: CSV filename
    
    Returns:
        trajectory: Dictionary with trajectory data
    """
    import csv
    
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            data.append([float(x) for x in row])
    
    data = np.array(data)
    
    return {
        'time': data[:, 0],
        'position': data[:, 1:4],
        'velocity': data[:, 4:7],
        'orientation_rpy': data[:, 7:10]
    }
