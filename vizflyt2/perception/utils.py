import math
import numpy as np

def quaternion_to_euler(x_rot, y_rot, z_rot, w):
    """Convert quaternion to roll, pitch, yaw (Euler angles)"""
    t0 = +2.0 * (w * x_rot + y_rot * z_rot)
    t1 = +1.0 - 2.0 * (x_rot * x_rot + y_rot * y_rot)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y_rot - z_rot * x_rot)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z_rot + x_rot * y_rot)
    t4 = +1.0 - 2.0 * (y_rot * y_rot + z_rot * z_rot)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler Angles to Quaternion
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def quaternion_multiply(quaternion1, quaternion0):
    """
    Multiplies 2 quaternions
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    make rotation matrix from rol, pitch, yaw
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    return R