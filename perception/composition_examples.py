#!/usr/bin/env python3
"""
Module Composition Examples

Demonstrates how to compose perception modules to create complex pipelines.

## Module Composition Concepts

The composition system allows you to chain modules where outputs from one module
are automatically passed as inputs to the next module. This enables powerful
combinations like:

1. **Noise + Flow**: Add noise to images, then compute optical flow on noisy images
2. **Stereo + Events**: Render stereo cameras, then simulate event cameras on both
3. **Custom Pipelines**: Create multi-stage processing chains

## How It Works

1. Each module has a `render(position, orientation_rpy, **kwargs)` method
2. The first module receives base inputs (position, orientation, initial data)
3. Each subsequent module receives outputs from previous modules via **kwargs
4. Output keys are prefixed with module names to avoid conflicts

## Module Requirements

For composition to work, modules should:
- Have a `render()` method
- Accept **kwargs to receive outputs from previous modules
- Return a dictionary of outputs

Example:
    class MyModule:
        def render(self, position, orientation_rpy, rgb=None, **kwargs):
            # Process rgb if available
            result = process(rgb) if rgb is not None else None
            return {'my_output': result}
"""

import numpy as np
import cv2
from splat_render import SplatRenderer
from stereo_camera import StereoCamera
from modules import VisionModule


# ========== Example Modules ==========

class NoiseModule(VisionModule):
    """Adds Gaussian noise to RGB images."""
    
    def __init__(self, noise_sigma=10):
        self.noise_sigma = noise_sigma
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        """Add noise to RGB image."""
        if rgb is None:
            return {}
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_sigma, rgb.shape)
        noisy_rgb = np.clip(rgb.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return {
            'rgb': noisy_rgb,  # Pass noisy RGB to next module
            'noise': noise
        }


class FlowModule(VisionModule):
    """Computes optical flow between consecutive frames."""
    
    def __init__(self):
        self.prev_rgb = None
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        """Compute optical flow."""
        if rgb is None:
            return {}
        
        if self.prev_rgb is None:
            # First frame, no flow
            self.prev_rgb = rgb.copy()
            return {'flow': np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.float32)}
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(self.prev_rgb, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self.prev_rgb = rgb.copy()
        
        return {
            'flow': flow,
            'flow_magnitude': np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        }
        
        


class EventCameraModule(VisionModule):
    """Simulates event camera from intensity changes."""
    
    def __init__(self, threshold=20, stereo=False):
        self.threshold = threshold
        self.stereo = stereo
        self.prev_left = None
        self.prev_right = None
    
    def render(self, position, orientation_rpy, 
               rgb_left=None, rgb_right=None, rgb=None, **kwargs):
        """Generate events from intensity changes."""
        results = {}
        
        # Handle mono or stereo
        if self.stereo and rgb_left is not None and rgb_right is not None:
            # Process left camera
            gray_left = cv2.cvtColor(rgb_left, cv2.COLOR_RGB2GRAY)
            if self.prev_left is not None:
                diff_left = gray_left.astype(float) - self.prev_left.astype(float)
                events_left = (np.abs(diff_left) > self.threshold).astype(np.uint8) * 255
                results['events_left'] = events_left
                results['polarity_left'] = np.sign(diff_left)
            self.prev_left = gray_left.copy()
            
            # Process right camera
            gray_right = cv2.cvtColor(rgb_right, cv2.COLOR_RGB2GRAY)
            if self.prev_right is not None:
                diff_right = gray_right.astype(float) - self.prev_right.astype(float)
                events_right = (np.abs(diff_right) > self.threshold).astype(np.uint8) * 255
                results['events_right'] = events_right
                results['polarity_right'] = np.sign(diff_right)
            self.prev_right = gray_right.copy()
            
        elif rgb is not None:
            # Process mono
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if self.prev_left is not None:
                diff = gray.astype(float) - self.prev_left.astype(float)
                events = (np.abs(diff) > self.threshold).astype(np.uint8) * 255
                results['events'] = events
                results['polarity'] = np.sign(diff)
            self.prev_left = gray.copy()
        
        return results


# ========== Composition Examples ==========

def example_noise_flow(config_path: str, json_path: str):
    """Example: Compose noise + flow modules."""
    print("=" * 60)
    print("Example 1: Noise + Flow Composition")
    print("=" * 60)
    
    # Create renderer
    renderer = SplatRenderer(config_path, json_path)
    
    # Create modules
    noise_module = NoiseModule(noise_sigma=15)
    flow_module = FlowModule()
    
    # Compose modules using + operator
    composed = renderer + noise_module + flow_module
    
    print(f"\nComposed module: {composed}")
    
    # Render a few frames
    print("\nRendering 5 frames with noisy optical flow...")
    import os
    os.makedirs("outputs", exist_ok=True)
    
    for i in range(5):
        position = np.array([0.0, 0.0, 0.0])  # At origin
        orientation_rpy = np.array([0.0, 0.0, 0.0])
        
        # Render with composition
        results = composed.render(position, orientation_rpy)
        
        # Save RGB (now noisy)
        cv2.imwrite(
            f"outputs/noisy_flow_frame_{i:03d}.png",
            results['rgb']
        )
        
        print(f"   Frame {i}: Rendered")
    
    print(f"\n✓ Saved to: outputs/")
    print()


def example_stereo_events(config_path: str, json_path: str):
    """Example: Compose stereo + events modules."""
    print("=" * 60)
    print("Example 2: Stereo + Events Composition")
    print("=" * 60)
    
    # Create stereo renderer
    stereo = StereoCamera(config_path, json_path, baseline=0.065)
    
    # Create event camera module
    event_module = EventCameraModule(threshold=15, stereo=True)
    
    # Compose using + operator
    composed = stereo + event_module
    
    print(f"\nComposed module: {composed}")
    
    # Render frames
    print("\nRendering 5 frames with stereo event cameras...")
    import os
    os.makedirs("outputs", exist_ok=True)
    
    for i in range(5):
        position = np.array([0.0, 0.0, 0.0])  # At origin
        orientation_rpy = np.array([0.0, 0.0, 0.0])
        
        results = composed.render(position, orientation_rpy)
        
        # Save concatenated stereo pair
        if 'rgb_left' in results and 'rgb_right' in results:
            concatenated = np.hstack([results['rgb_left'], results['rgb_right']])
            cv2.imwrite(
                f"outputs/stereo_events_frame_{i:03d}.png",
                concatenated
            )
        
        print(f"   Frame {i}: Rendered")
    
    print(f"\n✓ Saved to: outputs/")
    print()


def example_custom_pipeline(config_path: str, json_path: str):
    """Example: Create a custom multi-stage pipeline."""
    print("=" * 60)
    print("Example 3: Custom Multi-Stage Pipeline")
    print("=" * 60)
    
    renderer = SplatRenderer(config_path, json_path)
    
    # Create a 3-stage pipeline: noise -> events -> flow
    noise_module = NoiseModule(noise_sigma=10)
    event_module = EventCameraModule(threshold=12, stereo=False)
    flow_module = FlowModule()
    
    # Compose all three using + operator
    composed = renderer + noise_module + event_module + flow_module
    
    print(f"\nComposed module: {composed}")
    print(f"Pipeline: renderer -> noise -> events -> flow")
    
    print("\nRendering 3 frames with complex pipeline...")
    
    for i in range(3):
        position = np.array([0.0, 0.0, 0.0])  # At origin
        orientation_rpy = np.array([0.0, 0.0, 0.0])
        
        results = composed.render(position, orientation_rpy)
        
        print(f"   Frame {i}: {len(results.keys())} outputs from pipeline")
    
    print("\n✓ Pipeline executed successfully!")
    print()


def main():
    # Default paths - update these to match your setup
    config_path = "../splats/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml"
    json_path = "../splats/cam_settings.json"
    
    print("\n" + "=" * 60)
    print("VizFlyt2 - Module Composition Examples")
    print("=" * 60)
    print()
    
    print("Select an example:")
    print("1. Noise + Flow (optical flow on noisy images)")
    print("2. Stereo + Events (stereo event cameras)")
    print("3. Custom Pipeline (noise -> events -> flow)")
    print("4. Run all examples")
    print()
    
    choice = input("Enter choice (1-4, or 'q' to quit): ").strip()
    
    if choice == '1':
        example_noise_flow(config_path, json_path)
    elif choice == '2':
        example_stereo_events(config_path, json_path)
    elif choice == '3':
        example_custom_pipeline(config_path, json_path)
    elif choice == '4':
        example_noise_flow(config_path, json_path)
        example_stereo_events(config_path, json_path)
        example_custom_pipeline(config_path, json_path)
    elif choice.lower() == 'q':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
