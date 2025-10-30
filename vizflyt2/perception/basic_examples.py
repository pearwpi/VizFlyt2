"""
Example usage of VizFlyt2 Perception Modules

This file demonstrates various ways to use the perception modules directly
for different rendering and perception tasks.
"""

import numpy as np
import os
import cv2
from splat_render import SplatRenderer
from stereo_camera import StereoCamera
from modules import VisionModule


# ========== Example 1: Basic Mono Rendering ==========

def example_mono_rendering(config_path: str, json_path: str):
    """Basic mono camera rendering."""
    print("=== Example 1: Mono Rendering ===")
    
    # Create mono renderer
    renderer = SplatRenderer(
        config_path=config_path,
        json_path=json_path
    )
    
    # Define a pose at origin
    position = np.array([0.0, 0.0, 0.0])  # NED coordinates
    orientation_rpy = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
    
    # Render
    results = renderer.render(position, orientation_rpy)
    
    # Access results
    rgb_image = results['rgb']
    depth_map = results['depth']
    
    print(f"RGB shape: {rgb_image.shape}")
    print(f"Depth shape: {depth_map.shape}")
    
    # Save image
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/mono_render.png", rgb_image)
    print("Saved: outputs/mono_render.png")
    print()


# ========== Example 2: Stereo Rendering ==========

def example_stereo_rendering(config_path: str, json_path: str):
    """Stereo camera rendering with configurable baseline."""
    print("=== Example 2: Stereo Rendering ===")
    
    # Create stereo renderer with 6.5cm baseline
    stereo = StereoCamera(
        config_path=config_path,
        json_path=json_path,
        baseline=0.065  # 6.5cm
    )
    
    # Render stereo pair at origin
    position = np.array([0.0, 0.0, 0.0])
    orientation_rpy = np.array([0.0, 0.0, 0.0])
    
    results = stereo.render(position, orientation_rpy)
    
    # Access stereo results
    rgb_left = results['rgb_left']
    rgb_right = results['rgb_right']
    depth_left = results['depth_left']
    depth_right = results['depth_right']
    
    print(f"Left RGB shape: {rgb_left.shape}")
    print(f"Right RGB shape: {rgb_right.shape}")
    print(f"Stereo baseline: {stereo.get_baseline()}m")
    
    # Save concatenated stereo pair
    os.makedirs("outputs", exist_ok=True)
    concatenated = np.hstack([rgb_left, rgb_right])
    cv2.imwrite("outputs/stereo_render.png", concatenated)
    print("Saved: outputs/stereo_render.png")
    print()


# ========== Example 3: Module Composition ==========

def example_module_composition(config_path: str, json_path: str):
    """Compose renderer with vision module."""
    print("=== Example 3: Module Composition ===")
    
    # Create renderer
    renderer = SplatRenderer(
        config_path=config_path,
        json_path=json_path
    )
    
    # Create a simple vision module
    class SimpleBlur(VisionModule):
        def render(self, position, orientation_rpy, **kwargs):
            rgb = kwargs.get('rgb')
            if rgb is not None:
                blurred = cv2.GaussianBlur(rgb, (15, 15), 0)
                return {'rgb': blurred}
            return {}
    
    # Compose using + operator
    blur = SimpleBlur()
    composed = renderer + blur
    
    # Render
    position = np.array([0.0, 0.0, 0.0])
    orientation_rpy = np.array([0.0, 0.0, 0.0])
    results = composed.render(position, orientation_rpy)
    
    print(f"Composed outputs: {list(results.keys())}")
    print()

    # Save concatenated stereo pair
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/mono_blur.png", results['rgb'])
    print("Saved: outputs/mono_blur.png")
    print()


# ========== Main ==========

if __name__ == "__main__":
    # Default paths - update these to match your setup
    config_path = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
    json_path = "../splats/cam_settings.json"
    
    print("VizFlyt2 Perception Module Examples\n")
    
    # Run examples
    example_mono_rendering(config_path, json_path)
    example_stereo_rendering(config_path, json_path)
    example_module_composition(config_path, json_path)
    
    print("All examples completed!")

