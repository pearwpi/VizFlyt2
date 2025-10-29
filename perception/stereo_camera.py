from .splat_render import SplatRenderer
from .modules import BaseModule, VisionModule, ComposedBase
import numpy as np
import os
import cv2
from typing import Tuple, Optional, Dict


class StereoComposedBase(ComposedBase):
    """
    Custom composition for StereoCamera + VisionModule.
    
    Runs the vision module independently on left and right views,
    returning outputs with _left and _right suffixes.
    """
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Render stereo views, then apply vision processing to each view independently.
        """
        # Render stereo from base module
        base_results = self.base.render(position, orientation_rpy, **kwargs)
        
        # Extract left and right views
        # Build monocular inputs for each view
        left_inputs = {}
        right_inputs = {}
        
        for key, value in base_results.items():
            if key.endswith('_left'):
                # Strip _left suffix to get monocular key name (e.g., 'rgb_left' -> 'rgb')
                mono_key = key[:-5]
                left_inputs[mono_key] = value
            elif key.endswith('_right'):
                # Strip _right suffix
                mono_key = key[:-6]
                right_inputs[mono_key] = value
        
        # Apply vision module to left view
        left_vision_results = self.vision.render(position, orientation_rpy, **left_inputs)
        
        # Apply vision module to right view
        right_vision_results = self.vision.render(position, orientation_rpy, **right_inputs)
        
        # Combine results with _left and _right suffixes
        combined = base_results.copy()
        
        for key, value in left_vision_results.items():
            combined[f"{key}_left"] = value
        
        for key, value in right_vision_results.items():
            combined[f"{key}_right"] = value
        
        return combined


class StereoCamera(BaseModule):
    """
    A stereo camera system that uses SplatRenderer to render left and right views.
    
    The stereo baseline is oriented along the Y-axis (right) in the camera body frame,
    with the left camera at -baseline/2 and right camera at +baseline/2.
    
    Inherits from BaseModule since it renders images from the Gaussian Splat scene.
    """
    
    def __init__(self, 
                 config_path: str, 
                 json_path: str, 
                 baseline: float = 0.05,
                 aspect_ratio: float = 16/9):
        """
        Initialize the stereo camera system.
        
        Args:
            config_path: Path to the nerfstudio config file
            json_path: Path to camera settings JSON
            baseline: Distance between left and right cameras in meters (default: 0.05m)
            aspect_ratio: Image aspect ratio (default: 16/9)
        """
        self.renderer = SplatRenderer(config_path, json_path, aspect_ratio=aspect_ratio)
        self.baseline = baseline
    
    def compose(self, other: VisionModule, name: Optional[str] = None) -> BaseModule:
        """
        Compose this StereoCamera with a VisionModule.
        
        Uses StereoComposedBase to run the vision module independently on
        left and right views.
        """
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose BaseModule with another BaseModule")
        
        if not isinstance(other, VisionModule):
            raise TypeError(f"BaseModule can only compose with VisionModule, got {type(other)}")
        
        return StereoComposedBase(self, other, name)
        
    def _compute_stereo_offsets(self, rpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the left and right camera position offsets for stereo setup.
        The baseline is perpendicular to the forward direction (along the Y-axis in body frame).
        
        Args:
            rpy: Roll, pitch, yaw angles in radians (3,)
        
        Returns:
            left_offset, right_offset: 3D offsets in NED frame
        """
        roll, pitch, yaw = rpy
        
        # Rotation matrix from body frame to NED frame
        # Standard aerospace rotation sequence: Yaw -> Pitch -> Roll (ZYX)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        # Full rotation matrix (NED = R * Body)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        
        # In the camera body frame, the stereo baseline is along the Y-axis (right)
        # Left camera: -baseline/2 along Y, Right camera: +baseline/2 along Y
        half_baseline = self.baseline / 2.0
        
        # Transform to NED frame
        left_offset_body = np.array([0, -half_baseline, 0])
        right_offset_body = np.array([0, half_baseline, 0])
        
        left_offset = R @ left_offset_body
        right_offset = R @ right_offset_body
        
        return left_offset, right_offset
    
    def render(self, 
               position: np.ndarray, 
               orientation_rpy: np.ndarray,
               **kwargs) -> Dict[str, np.ndarray]:
        """
        Render both left and right stereo views from the given pose.
        
        This is the BaseModule interface method that renders stereo pairs.
        
        Args:
            position: (3,) [x, y, z] in meters (in NED frame)
            orientation_rpy: (3,) [roll, pitch, yaw] in radians (in NED frame)
            **kwargs: Additional parameters (unused but included for interface consistency)
        
        Returns:
            Dictionary containing:
                - 'rgb_left': Left camera RGB image (H, W, 3) uint8
                - 'rgb_right': Right camera RGB image (H, W, 3) uint8
                - 'depth_left': Left camera depth colormap (H, W, 3) uint8
                - 'depth_right': Right camera depth colormap (H, W, 3) uint8
                - 'depth_raw_left': Left camera raw depth (H, W) float32
                - 'depth_raw_right': Right camera raw depth (H, W) float32
        """
        # Compute stereo camera positions
        left_offset, right_offset = self._compute_stereo_offsets(orientation_rpy)
        left_pos = position + left_offset
        right_pos = position + right_offset
        
        # Render left camera
        left_results = self.renderer.render(
            left_pos.astype(float), 
            orientation_rpy.astype(float)
        )
        
        # Render right camera
        right_results = self.renderer.render(
            right_pos.astype(float), 
            orientation_rpy.astype(float)
        )
        
        return {
            'rgb_left': left_results['rgb'],
            'rgb_right': right_results['rgb'],
            'depth_left': left_results['depth'],
            'depth_right': right_results['depth'],
            'depth_raw_left': left_results['depth_raw'],
            'depth_raw_right': right_results['depth_raw']
        }
    
    def render_stereo(self, 
                     position: np.ndarray, 
                     orientation_rpy: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Render both left and right stereo views from the given pose.
        
        Convenience wrapper for render() method for backward compatibility.
        
        Args:
            position: (3,) [x, y, z] in meters (in NED frame)
            orientation_rpy: (3,) [roll, pitch, yaw] in radians (in NED frame)
        
        Returns:
            Dictionary containing stereo image pairs and depth maps
        """
        return self.render(position, orientation_rpy)
    
    def render_mono(self, 
                   position: np.ndarray, 
                   orientation_rpy: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Render a single (mono) view from the given pose.
        
        Args:
            position: (3,) [x, y, z] in meters (in NED frame)
            orientation_rpy: (3,) [roll, pitch, yaw] in radians (in NED frame)
        
        Returns:
            Dictionary containing:
                - 'rgb': RGB image (H, W, 3) uint8
                - 'depth': Depth colormap (H, W, 3) uint8
                - 'depth_raw': Raw depth values (H, W) float32
        """
        return self.renderer.render(
            position.astype(float), 
            orientation_rpy.astype(float)
        )
    
    def set_baseline(self, baseline: float):
        """
        Update the stereo baseline distance.
        
        Args:
            baseline: New baseline distance in meters
        """
        self.baseline = baseline
    
    def get_baseline(self) -> float:
        """
        Get the current stereo baseline distance.
        
        Returns:
            Baseline distance in meters
        """
        return self.baseline
    
    def get_image_dimensions(self) -> Tuple[int, int]:
        """
        Get the image dimensions (height, width).
        
        Returns:
            (height, width) tuple
        """
        return (self.renderer.image_height, self.renderer.image_width)
