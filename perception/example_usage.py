#!/usr/bin/env python3
"""
VizFlyt2 Perception - Simple Usage Example

Render a few images around the origin using mono and stereo cameras.
"""

import numpy as np
import cv2
import os
from splat_render import SplatRenderer
from stereo_camera import StereoCamera


def main():
    # Configuration - UPDATE THESE PATHS
    config_path = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
    json_path = "../splats/cam_settings.json"
    
    print("=" * 60)
    print("VizFlyt2 - Simple Rendering Example")
    print("=" * 60)
    
    # ========== MONO RENDERING ==========
    print("\n1. Mono Camera - Rendering 5 images at origin...")
    
    renderer = SplatRenderer(config_path, json_path)
    
    # Create simple poses at origin
    os.makedirs("outputs", exist_ok=True)
    
    for i in range(5):
        # Position: at origin
        position = np.array([0.0, 0.0, 0.0])  # NED coordinates
        orientation_rpy = np.array([0.0, 0.0, 0.0])  # no rotation
        
        # Render
        results = renderer.render(position, orientation_rpy)
        
        # Save RGB image
        cv2.imwrite(
            f"outputs/mono_frame_{i:03d}.png",
            results['rgb']
        )
        
        print(f"   Frame {i}: pos={position}, depth_range=[{results['depth'].min():.2f}, {results['depth'].max():.2f}]m")
    
    print(f"   ✓ Saved to: outputs/")
    
    # ========== STEREO RENDERING ==========
    print("\n2. Stereo Camera - Rendering 5 stereo pairs at origin...")
    
    stereo = StereoCamera(config_path, json_path, baseline=0.065)
    
    for i in range(5):
        position = np.array([0.0, 0.0, 0.0])
        orientation_rpy = np.array([0.0, 0.0, 0.0])
        
        # Render stereo
        results = stereo.render(position, orientation_rpy)
        
        # Save concatenated stereo pair
        concatenated = np.hstack([results['rgb_left'], results['rgb_right']])
        cv2.imwrite(
            f"outputs/stereo_frame_{i:03d}.png",
            concatenated
        )
        
        print(f"   Frame {i}: pos={position}")
    
    print(f"   ✓ Saved to: outputs/")
    
    print("\n" + "=" * 60)
    print("Done! Check the outputs/ folder for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
