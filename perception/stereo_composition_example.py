"""
Example demonstrating stereo composition with vision modules.

The StereoCamera class now runs vision modules independently on each view,
producing _left and _right outputs for all vision module results.
"""

import numpy as np
import os
from stereo_camera import StereoCamera
from modules import VisionModule
from typing import Dict


class SnowModule(VisionModule):
    """Example vision module that adds snow effect to RGB images."""
    
    def __init__(self, intensity: float = 0.3):
        self.intensity = intensity
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Add snow effect to the RGB image."""
        rgb = kwargs.get('rgb')
        if rgb is None:
            raise ValueError("SnowModule requires 'rgb' input")
        
        # Simple snow simulation: add random white pixels
        snow_mask = np.random.random(rgb.shape[:2]) < self.intensity
        rgb_snow = rgb.copy()
        rgb_snow[snow_mask] = [255, 255, 255]
        
        return {'rgb': rgb_snow}


class EdgeDetectionModule(VisionModule):
    """Example vision module that detects edges in images."""
    
    def __init__(self, threshold1: float = 100, threshold2: float = 200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Detect edges using Canny edge detection."""
        import cv2
        
        rgb = kwargs.get('rgb')
        if rgb is None:
            raise ValueError("EdgeDetectionModule requires 'rgb' input")
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        
        # Convert back to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return {'edges': edges_rgb}


def example_base_stereo(config_path: str, json_path: str):
    """
    Example: Base StereoCamera rendering without any vision processing.
    
    This serves as a baseline to compare with composition examples.
    Output will contain:
        - rgb_left, rgb_right (raw stereo pair)
        - depth_left, depth_right
        - depth_raw_left, depth_raw_right
    """
    # Create stereo camera
    stereo = StereoCamera(
        config_path=config_path,
        json_path=json_path,
        baseline=0.065  # 6.5cm baseline
    )
    
    # Render without any composition
    position = np.array([0.0, 0.0, -0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    
    results = stereo.render(position, orientation)
    
    print("Base Stereo outputs:", results.keys())
    
    # Save concatenated image (left | right)
    import cv2
    os.makedirs("outputs", exist_ok=True)
    concatenated = np.hstack([results['rgb_left'], results['rgb_right']])
    cv2.imwrite("outputs/stereo_base.png", concatenated)
    print("Saved: outputs/stereo_base.png")


def example_stereo_with_snow(config_path: str, json_path: str):
    """
    Example: StereoCamera + SnowModule
    
    The snow effect is applied independently to left and right views.
    Output will contain:
        - rgb_left, rgb_right (with snow)
        - depth_left, depth_right
        - depth_raw_left, depth_raw_right
    """
    # Create stereo camera
    stereo = StereoCamera(
        config_path=config_path,
        json_path=json_path,
        baseline=0.065  # 6.5cm baseline
    )
    
    # Create snow effect
    snow = SnowModule(intensity=0.2)
    
    # Compose using + operator
    stereo_with_snow = stereo + snow
    
    # Render
    position = np.array([0.0, 0.0, -0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    
    results = stereo_with_snow.render(position, orientation)
    
    # Results will have:
    # - rgb_left: left view with snow
    # - rgb_right: right view with snow
    # - depth_left, depth_right
    # - depth_raw_left, depth_raw_right
    
    print("Stereo + Snow outputs:", results.keys())
    
    # Save concatenated image (left | right)
    import cv2
    os.makedirs("outputs", exist_ok=True)
    concatenated = np.hstack([results['rgb_left'], results['rgb_right']])
    cv2.imwrite("outputs/stereo_snow.png", concatenated)
    print("Saved: outputs/stereo_snow.png")


def example_stereo_with_edges(config_path: str, json_path: str):
    """
    Example: StereoCamera + EdgeDetectionModule
    
    Edge detection is applied independently to left and right views.
    Output will contain:
        - rgb_left, rgb_right (original)
        - edges_left, edges_right (edge detection results)
        - depth_left, depth_right
        - depth_raw_left, depth_raw_right
    """
    # Create stereo camera
    stereo = StereoCamera(
        config_path=config_path,
        json_path=json_path,
        baseline=0.065  # 6.5cm baseline
    )
    
    # Create edge detection module
    edges = EdgeDetectionModule(threshold1=50, threshold2=150)
    
    # Compose using + operator
    stereo_with_edges = stereo + edges
    
    # Render at origin
    position = np.array([0.0, 0.0, 0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    
    results = stereo_with_edges.render(position, orientation)
    
    print("Stereo + Edges outputs:", results.keys())
    
    # Save concatenated image (2x2 grid: rgb on top, edges on bottom)
    import cv2
    os.makedirs("outputs", exist_ok=True)
    top_row = np.hstack([results['rgb_left'], results['rgb_right']])
    bottom_row = np.hstack([results['edges_left'], results['edges_right']])
    concatenated = np.vstack([top_row, bottom_row])
    
    cv2.imwrite("outputs/stereo_edges.png", concatenated)
    print("Saved: outputs/stereo_edges.png (2x2 grid: RGB top, Edges bottom)")


def example_stereo_with_chained_vision(config_path: str, json_path: str):
    """
    Example: StereoCamera + SnowModule + EdgeDetectionModule
    
    Both vision modules are applied independently to each view:
    1. Left view: base render -> snow -> edge detection
    2. Right view: base render -> snow -> edge detection
    
    Output will contain _left and _right versions of all outputs.
    """
    # Create stereo camera
    stereo = StereoCamera(
        config_path=config_path,
        json_path=json_path
    )
    
    # Create vision pipeline: snow then edges
    snow = SnowModule(intensity=0.15)
    edges = EdgeDetectionModule(threshold1=50, threshold2=150)
    
    # Compose: stereo + (snow + edges)
    # This creates a stereo system where each view gets snow then edge detection
    vision_pipeline = snow + edges
    stereo_with_vision = stereo + vision_pipeline
    
    # Render at origin
    position = np.array([0.0, 0.0, 0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    
    results = stereo_with_vision.render(position, orientation)
    
    # Results will have:
    # - rgb_left: left view with snow
    # - rgb_right: right view with snow
    # - edges_left: edges detected on left view
    # - edges_right: edges detected on right view
    # - depth_left, depth_right
    # - depth_raw_left, depth_raw_right
    
    print("Stereo + Snow + Edges outputs:", results.keys())
    
    # Save concatenated image (2x2 grid: rgb_left, rgb_right on top; edges_left, edges_right on bottom)
    import cv2
    os.makedirs("outputs", exist_ok=True)
    rgb_left_bgr = results['rgb_left']
    rgb_right_bgr = results['rgb_right']
    edges_left_bgr = results['edges_left']
    edges_right_bgr = results['edges_right']

    top_row = np.hstack([rgb_left_bgr, rgb_right_bgr])
    bottom_row = np.hstack([edges_left_bgr, edges_right_bgr])
    concatenated = np.vstack([top_row, bottom_row])
    
    cv2.imwrite("outputs/stereo_chained.png", concatenated)
    print("Saved: outputs/stereo_chained.png (2x2 grid: RGB top, Edges bottom)")


def how_it_works():
    """
    How StereoCamera composition works:
    
    1. StereoCamera.compose() creates a StereoComposedBase instance
    
    2. StereoComposedBase.render() does:
       a. Call stereo.render() -> get rgb_left, rgb_right, depth_left, depth_right, etc.
       b. Split into two monocular dictionaries:
          - left_inputs = {'rgb': rgb_left, 'depth': depth_left, ...}
          - right_inputs = {'rgb': rgb_right, 'depth': depth_right, ...}
       c. Apply vision module to left_inputs -> left_vision_results
       d. Apply vision module to right_inputs -> right_vision_results
       e. Combine with _left and _right suffixes
    
    This means VisionModules always operate on monocular images (with 'rgb', 'depth' keys)
    but stereo composition automatically applies them to both views.
    """
    pass


if __name__ == "__main__":
    # Default paths - update these to match your setup
    config_path = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
    json_path = "../splats/cam_settings.json"
    
    print("Stereo Composition Examples")
    print("=" * 60)
    print()
    
    print("Example 0: Base Stereo (no composition)")
    print("-" * 60)
    example_base_stereo(config_path, json_path)
    print()
    
    print("Example 1: Stereo + Snow")
    print("-" * 60)
    example_stereo_with_snow(config_path, json_path)
    print()
    
    print("Example 2: Stereo + Edges")
    print("-" * 60)
    example_stereo_with_edges(config_path, json_path)
    print()

    print("Example 3: Stereo + Snow + Edges")
    print("-" * 60)
    example_stereo_with_chained_vision(config_path, json_path)
    print()
    
    print("How It Works:")
    print("-" * 60)
    print(how_it_works.__doc__)
