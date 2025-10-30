"""
VizFlyt2 Perception Module

Provides rendering capabilities using Gaussian Splatting and modular vision pipeline.
"""

from .modules import BaseModule, VisionModule, ComposedBase

# Try to import SplatRenderer and StereoCamera (requires nerfstudio)
try:
    from .splat_render import SplatRenderer
    from .stereo_camera import StereoCamera
    __all__ = [
        'BaseModule',
        'VisionModule', 
        'ComposedBase',
        'SplatRenderer',
        'StereoCamera',
    ]
except ImportError:
    # Nerfstudio not available, only export base modules
    __all__ = [
        'BaseModule',
        'VisionModule',
        'ComposedBase',
    ]

__version__ = "0.1.0"
