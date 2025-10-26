"""
Planning module for trajectory generation and reactive obstacle avoidance.

All planners implement compute_action(**kwargs) → Dict[str, np.ndarray]
This unified interface supports:
- Reactive planners: depth_image → velocity commands
- RL agents: observation → action
- Trajectory followers: time → reference velocity

Base Classes:
    - BasePlanner: Abstract base for all planners
    - ReactiveVisualPlanner: Base for vision-based reactive planners
    - TrajectoryPlanner (base): Base for trajectory-following planners

Planners:
    - TrajectoryPlanner: Generate smooth trajectories using primitives
    - PotentialFieldPlanner: Reactive obstacle avoidance from depth images

Primitives:
    - line, circle, figure-8, spiral, waypoint sequences
"""

from .base import BasePlanner, ReactiveVisualPlanner
from .base import TrajectoryPlanner as BaseTrajectoryPlanner

from .primitives import (
    line_trajectory,
    circle_trajectory,
    figure8_trajectory,
    spiral_trajectory,
    waypoint_trajectory
)

from .trajectory import TrajectoryPlanner

from .planner import (
    PotentialFieldPlanner,
    calculate_velocity,
    thresholding,
    calculate_free_direction_cc,
    calculate_free_direction_cc_boundary
)

__all__ = [
    # Base classes
    'BasePlanner',
    'ReactiveVisualPlanner',
    'BaseTrajectoryPlanner',
    # Trajectory planning
    'TrajectoryPlanner',
    'line_trajectory',
    'circle_trajectory',
    'figure8_trajectory',
    'spiral_trajectory',
    'waypoint_trajectory',
    # Vision-based planning
    'PotentialFieldPlanner',
    'calculate_velocity',
    'thresholding',
    'calculate_free_direction_cc',
    'calculate_free_direction_cc_boundary'
]

