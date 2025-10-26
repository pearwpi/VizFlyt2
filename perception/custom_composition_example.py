#!/usr/bin/env python3
"""
Example showing how to customize composition behavior by subclassing ComposedBase.

This demonstrates the simplicity of the new composition system.
"""

import numpy as np
from modules import BaseModule, VisionModule, ComposedBase
from typing import Dict, Optional
import time


class TimedComposition(ComposedBase):
    """
    Custom composition that adds timing information.
    
    Simply override render() to add custom behavior!
    """
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Add timing to the composition."""
        start_time = time.time()
        
        # Call parent render (does the actual composition)
        results = super().render(position, orientation_rpy, **kwargs)
        
        # Add timing info
        results['_timing_ms'] = (time.time() - start_time) * 1000
        
        return results


class PreprocessingComposition(ComposedBase):
    """
    Custom composition that preprocesses data before vision module.
    """
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Preprocess RGB before applying vision module."""
        # Render from base
        base_results = self.base.render(position, orientation_rpy, **kwargs)
        
        # Custom preprocessing: normalize RGB
        if 'rgb' in base_results:
            base_results['rgb_normalized'] = base_results['rgb'].astype(float) / 255.0
        
        # Apply vision processing with preprocessed data
        vision_results = self.vision.render(position, orientation_rpy, **base_results)
        
        # Combine
        combined = base_results.copy()
        combined.update(vision_results)
        
        return combined


class ConditionalComposition(ComposedBase):
    """
    Custom composition that conditionally applies vision processing.
    """
    
    def __init__(self, base: BaseModule, vision: VisionModule, 
                 name: Optional[str] = None, quality_threshold: float = 0.5):
        super().__init__(base, vision, name)
        self.quality_threshold = quality_threshold
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Only apply vision if quality is good enough."""
        # Render from base
        base_results = self.base.render(position, orientation_rpy, **kwargs)
        
        # Check quality (assume base provides this)
        quality = base_results.get('quality', np.random.random())
        
        if quality >= self.quality_threshold:
            # Good quality - apply vision processing
            vision_results = self.vision.render(position, orientation_rpy, **base_results)
            combined = base_results.copy()
            combined.update(vision_results)
            combined['vision_applied'] = True
            return combined
        else:
            # Poor quality - skip vision processing
            base_results['vision_applied'] = False
            base_results['skip_reason'] = f"Quality {quality:.2f} < {self.quality_threshold}"
            return base_results


class CustomRenderer(BaseModule):
    """
    Custom renderer that uses custom composition classes.
    """
    
    def __init__(self, name="CustomRenderer"):
        self.name = name
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Dummy render with quality metric."""
        return {
            'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
            'depth': np.zeros((480, 640, 3), dtype=np.uint8),
            'depth_raw': np.zeros((480, 640), dtype=np.float32),
            'quality': np.random.random()
        }
    
    def compose(self, other: VisionModule, name: Optional[str] = None, 
                composition_class=None, **kwargs) -> BaseModule:
        """
        Override compose to allow choosing composition class.
        
        This lets users pick which composition behavior they want!
        """
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose BaseModule with another BaseModule")
        
        if not isinstance(other, VisionModule):
            raise TypeError(f"BaseModule can only compose with VisionModule")
        
        # Use custom composition class if provided
        if composition_class is None:
            from modules import ComposedBase
            composition_class = ComposedBase
        
        return composition_class(self, other, name, **kwargs)


# Example usage
if __name__ == "__main__":
    from composition_demo import SnowModule
    
    print("=== Simple Custom Composition Examples ===\n")
    
    renderer = CustomRenderer()
    snow = SnowModule(intensity=0.3)
    position = np.array([0.0, 0.0, 0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    
    # Example 1: Timed composition
    print("1. Timed Composition:")
    timed = renderer.compose(snow, composition_class=TimedComposition)
    results = timed.render(position, orientation)
    print(f"   Timing: {results['_timing_ms']:.2f} ms")
    print()
    
    # Example 2: Preprocessing composition
    print("2. Preprocessing Composition:")
    preprocessed = renderer.compose(snow, composition_class=PreprocessingComposition)
    results = preprocessed.render(position, orientation)
    print(f"   Has normalized RGB: {'rgb_normalized' in results}")
    print()
    
    # Example 3: Conditional composition
    print("3. Conditional Composition (quality threshold = 0.5):")
    for i in range(5):
        conditional = renderer.compose(snow, composition_class=ConditionalComposition,
                                      quality_threshold=0.5)
        results = conditional.render(position, orientation)
        print(f"   Attempt {i+1}: Quality={results['quality']:.2f}, "
              f"Applied={results['vision_applied']}")
    print()
    
    print("✓ Super simple! Just subclass ComposedBase and override render()!")

