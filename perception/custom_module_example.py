"""
Example: Creating Custom Modules

This shows TWO ways to create custom VisionModules:
1. Factory Decorators (quick & easy for simple functions)
2. Subclassing (full control for stateful modules)
"""

from modules import BaseModule, VisionModule, vision_module_factory, rgb_vision_module_factory
import numpy as np
import cv2


# ========== Method 1: Factory Decorators (Simple & Quick) ==========

@rgb_vision_module_factory
def simple_edge_detector(rgb_image):
    """
    Simple edge detection using decorator.
    Automatically extracts 'rgb' from kwargs.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return {
        'edges': edges,
        'rgb': rgb_image  # Pass through original
    }


@vision_module_factory
def depth_and_rgb_processor(position, orientation_rpy, **kwargs):
    """
    Process both depth and RGB using generic decorator.
    Has access to all kwargs.
    """
    rgb = kwargs.get('rgb')
    depth = kwargs.get('depth')
    
    if rgb is None or depth is None:
        return {}
    
    # Combine RGB with depth overlay
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Blend
    blended = cv2.addWeighted(rgb, 0.7, depth_colored, 0.3, 0)
    
    return {
        'blended': blended,
        'rgb': rgb,
        'depth': depth
    }


@rgb_vision_module_factory
def brightness_adjust(rgb_image):
    """Simple brightness adjustment."""
    return {'rgb': cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=30)}


# ========== Method 2: Subclassing (Full Control) ==========

class EdgeDetectionModule(VisionModule):
    """
    Example VisionModule that detects edges in RGB images.
    """
    
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        """Detect edges using Canny edge detection."""
        if rgb is None:
            return {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        return {
            'edges': edges,
            'rgb': rgb  # Pass through original RGB
        }


class DepthColorizer(VisionModule):
    """
    Example VisionModule that colorizes depth maps.
    """
    
    def __init__(self, colormap=cv2.COLORMAP_JET):
        self.colormap = colormap
    
    def render(self, position, orientation_rpy, depth=None, **kwargs):
        """Apply colormap to depth."""
        if depth is None:
            return {}
        
        # Normalize depth to 0-255 range
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_norm, self.colormap)
        
        return {
            'depth_colored': depth_colored,
            'depth': depth  # Pass through original depth
        }


class TemporalFilterModule(VisionModule):
    """
    Example VisionModule that applies temporal filtering to reduce noise.
    """
    
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # Blending factor
        self.prev_rgb = None
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        """Apply exponential moving average to RGB."""
        if rgb is None:
            return {}
        
        if self.prev_rgb is not None:
            # Blend current frame with previous
            filtered_rgb = (self.alpha * rgb + (1 - self.alpha) * self.prev_rgb).astype(np.uint8)
        else:
            filtered_rgb = rgb
        
        self.prev_rgb = filtered_rgb.copy()
        
        return {
            'rgb': filtered_rgb,
            'rgb_original': rgb  # Keep original too
        }


# ========== Example: Custom BaseModule (Advanced) ==========

class SyntheticRenderer(BaseModule):
    """
    Example BaseModule that generates synthetic RGB and depth.
    (For demonstration - in practice you'd use SplatRenderer)
    """
    
    def __init__(self, resolution=(1080, 1920)):
        self.resolution = resolution
    
    def render(self, position, orientation_rpy, **kwargs):
        """Generate synthetic RGB and depth images."""
        h, w = self.resolution
        
        # Generate synthetic RGB (gradient based on position)
        x_pos = int(position[0] * 50) % 255
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = x_pos  # Red channel based on X position
        rgb[:, :, 1] = 128    # Fixed green
        rgb[:, :, 2] = 200    # Fixed blue
        
        # Generate synthetic depth (distance field)
        depth = np.ones((h, w), dtype=np.float32) * np.linalg.norm(position)
        
        return {
            'rgb': rgb,
            'depth': depth
        }


# ========== Usage Example ==========

def main():
    """Demonstrate custom module usage."""
    from splat_render import SplatRenderer
    
    print("=" * 60)
    print("Custom Module Examples")
    print("=" * 60)
    
    # Example 1: Decorator-based modules (quick & simple)
    print("\n1. Factory Decorators (Simple Functions)")
    print("-" * 60)
    
    # Example usage:
    # renderer = SplatRenderer("config.yml", "cam_settings.json")
    # composed = renderer + simple_edge_detector + brightness_adjust
    # results = composed.render(position, orientation_rpy)
    # edges = results['edges']
    
    print("@rgb_vision_module_factory creates VisionModule from function")
    print("- Automatically extracts 'rgb' from kwargs")
    print("- Perfect for stateless processing")
    print("- Example: simple_edge_detector, brightness_adjust")
    print("\n@vision_module_factory for generic processing")
    print("- Access to all kwargs (rgb, depth, etc.)")
    print("- Example: depth_and_rgb_processor")
    
    # Example 2: Class-based modules (full control)
    print("\n2. Subclassing (Full Control)")
    print("-" * 60)
    
    print("EdgeDetectionModule inherits from VisionModule")
    print("- Has __init__ parameters (low_threshold, high_threshold)")
    print("- Stateful if needed")
    print("- Example: EdgeDetectionModule(low_threshold=50, high_threshold=150)")
    
    # Example 3: Both work with composition!
    print("\n3. Module Composition (Both Methods Work!)")
    print("-" * 60)
    
    print("Decorators and classes compose seamlessly:")
    print("  renderer + simple_edge_detector + DepthColorizer()")
    print("  renderer + brightness_adjust + EdgeDetectionModule()")
    print("  renderer + depth_and_rgb_processor")
    
    # Example 4: Type checking
    print("\n4. Module Type Checking")
    print("-" * 60)
    
    edge_mod = EdgeDetectionModule()
    depth_mod = DepthColorizer()
    synthetic_mod = SyntheticRenderer()
    
    print(f"EdgeDetectionModule is VisionModule: {isinstance(edge_mod, VisionModule)}")
    print(f"simple_edge_detector is VisionModule: {isinstance(simple_edge_detector, VisionModule)}")
    print(f"DepthColorizer is VisionModule: {isinstance(depth_mod, VisionModule)}")
    print(f"SyntheticRenderer is BaseModule: {isinstance(synthetic_mod, BaseModule)}")
    
    # Example 5: When to use which method
    print("\n5. Which Method to Use?")
    print("-" * 60)
    print("Use Decorators when:")
    print("  ✓ Simple, stateless processing")
    print("  ✓ No __init__ parameters needed")
    print("  ✓ Quick prototyping")
    print("  ✓ One-off transformations")
    print("\nUse Subclassing when:")
    print("  ✓ Need initialization parameters")
    print("  ✓ Stateful processing (e.g., temporal filtering)")
    print("  ✓ Complex logic with helper methods")
    print("  ✓ Reusable, configurable modules")
    
    print("\n" + "=" * 60)
    print("Both methods enable:")
    print("  ✓ Easy composition with + operator")
    print("  ✓ Type safety and validation")
    print("  ✓ Clear code organization")
    print("  ✓ Flexible pipeline creation")
    print("=" * 60)


if __name__ == "__main__":
    main()
