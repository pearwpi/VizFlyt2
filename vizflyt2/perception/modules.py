"""
VizFlyt2 Perception Module Base Classes

Defines the base module hierarchy for the perception system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseModule(ABC):
    """
    Base class for rendering modules that generate RGB and depth from Gaussian Splat.
    
    These modules render directly from the splat scene and don't require previous
    module outputs. Examples: SplatRenderer, StereoCamera.
    
    Composition rules:
    - BaseModule + VisionModule -> BaseModule (rendering with post-processing)
    - BaseModule + BaseModule -> ERROR (cannot compose two renderers)
    """
    
    @abstractmethod
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Render RGB and depth images from the scene.
        
        Args:
            position: (3,) array [x, y, z] in meters (NED frame)
            orientation_rpy: (3,) array [roll, pitch, yaw] in radians
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with at minimum 'rgb' and 'depth' keys
        """
        pass
    
    def compose(self, other: 'VisionModule', name: Optional[str] = None) -> 'BaseModule':
        """
        Compose this BaseModule with a VisionModule.
        
        Returns a new BaseModule that renders then applies vision processing.
        """
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose BaseModule with another BaseModule")
        
        if not isinstance(other, VisionModule):
            raise TypeError(f"BaseModule can only compose with VisionModule, got {type(other)}")
        
        return ComposedBase(self, other, name)
    
    def __add__(self, other: 'VisionModule') -> 'BaseModule':
        """Syntactic sugar: renderer + vision_module"""
        return self.compose(other)


class VisionModule(ABC):
    """
    Base class for vision processing modules that operate on RGB/depth images.
    
    These modules process existing image data rather than rendering from the splat.
    Examples: EventCamera, OpticalFlow, Snow effects, Segmentation.
    
    Composition rules:
    - VisionModule + VisionModule -> VisionModule (chained processing)
    - VisionModule + BaseModule -> ERROR (vision modules don't render)
    """
    
    @abstractmethod
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Process RGB/depth inputs to generate derived outputs.
        
        Args:
            position: (3,) array [x, y, z] in meters (NED frame)
            orientation_rpy: (3,) array [roll, pitch, yaw] in radians
            **kwargs: Input data from previous modules (rgb, depth, etc.)
        
        Returns:
            Dictionary with processing outputs
        """
        pass
    
    def compose(self, other: 'VisionModule', name: Optional[str] = None) -> 'VisionModule':
        """
        Compose this VisionModule with another VisionModule.
        
        Creates a pipeline where this module's outputs feed into the next.
        """
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose VisionModule with BaseModule")
        
        if not isinstance(other, VisionModule):
            raise TypeError(f"VisionModule can only compose with VisionModule, got {type(other)}")
        
        return ComposedVision(self, other, name)
    
    def __add__(self, other: 'VisionModule') -> 'VisionModule':
        """Syntactic sugar: vision_module1 + vision_module2"""
        return self.compose(other)
    
def vision_module_factory(func) -> VisionModule:
    """
    Factory function to create vision modules by name.
    """

    class CustomVisionModule(VisionModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize custom module-specific parameters here

        def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
            # Call the provided function to perform vision processing
            return func(position, orientation_rpy, **kwargs)

    return CustomVisionModule()

def rgb_vision_module_factory(func) -> VisionModule:
    """
    Factory function to create RGB-specific vision modules by name.
    Assumes input contains 'rgb' key.
    """

    class CustomRGBVisionModule(VisionModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize custom module-specific parameters here

        def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
            rgb_image = kwargs.get('rgb', None)
            if rgb_image is None:
                raise ValueError("Input to RGB vision module must contain 'rgb' key.")
            # Call the provided function to perform vision processing
            return func(rgb_image)

    return CustomRGBVisionModule()


class ComposedBase(BaseModule):
    """
    A BaseModule composed with a VisionModule.
    
    Simple delegation pattern - just stores modules and calls them in sequence.
    Subclasses can override render() to customize the composition behavior.
    """
    
    def __init__(self, base: BaseModule, vision: VisionModule, name: Optional[str] = None):
        self.base = base
        self.vision = vision
        self.name = name or f"{base.__class__.__name__}+{vision.__class__.__name__}"
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Default composition: render from base, then apply vision processing.
        
        Override this method in subclasses to customize behavior (add timing,
        preprocessing, conditional logic, etc.)
        """
        # Render from base module
        base_results = self.base.render(position, orientation_rpy, **kwargs)
        
        # Apply vision processing
        vision_results = self.vision.render(position, orientation_rpy, **base_results)
        
        # Combine (vision overrides base for same keys)
        combined = base_results.copy()
        combined.update(vision_results)
        
        return combined
    
    def __repr__(self) -> str:
        return f"BaseModule({self.name})"


class ComposedVision(VisionModule):
    """
    A VisionModule composed with another VisionModule.
    
    Simple delegation pattern - just stores modules and calls them in sequence.
    Subclasses can override render() to customize the composition behavior.
    """
    
    def __init__(self, first: VisionModule, second: VisionModule, name: Optional[str] = None):
        self.first = first
        self.second = second
        self.name = name or f"{first.__class__.__name__}+{second.__class__.__name__}"
    
    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Default composition: apply first module, then second with first's outputs.
        
        Override this method in subclasses to customize behavior.
        """
        # Apply first vision module
        first_results = self.first.render(position, orientation_rpy, **kwargs)
        
        # Apply second with combined inputs
        combined_inputs = kwargs.copy()
        combined_inputs.update(first_results)
        second_results = self.second.render(position, orientation_rpy, **combined_inputs)
        
        # Combine results
        combined = first_results.copy()
        combined.update(second_results)
        
        return combined
    
    def __repr__(self) -> str:
        return f"VisionModule({self.name})"