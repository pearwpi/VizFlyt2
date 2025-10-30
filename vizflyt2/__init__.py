"""
VizFlyt2 - Flexible Perception, Dynamics, and Planning System

A system for rendering photorealistic synthetic sensor data using Gaussian Splatting,
with integrated dynamics simulation and planning capabilities.

Usage:
    import vizflyt2
    
    # Access submodules
    from vizflyt2 import dynamics, planning, perception
    
    # Or import directly from submodules
    from planning import PotentialFieldPlanner
    from dynamics import PointMassDynamics
"""

__version__ = "0.1.0"

# Import submodules - these will be available as vizflyt2.dynamics, etc.
from . import dynamics
from . import planning

try:
    from . import perception
    __all__ = ['perception', 'dynamics', 'planning', '__version__']
except ImportError:
    __all__ = ['dynamics', 'planning', '__version__']
    perception = None  # Mark as unavailable

# For convenience, expose commonly used classes at top level
# These can be accessed as vizflyt2.PointMassDynamics, etc.
from .dynamics import PointMassDynamics
from .planning import PotentialFieldPlanner, TrajectoryPlanner

if perception is not None:
    try:
        from .perception import SplatRenderer, StereoCamera
    except ImportError:
        pass  # Some perception components not available
