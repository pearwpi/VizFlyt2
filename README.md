# VizFlyt2 (WIP)

A modular Python framework for photorealistic synthetic sensor simulation and autonomous flight. VizFlyt2 integrates Gaussian Splatting-based rendering with simple point-mass dynamics and intelligent trajectory planning for vision-based navigation research.

This is currently a WIP project, some features are currently unreleased to the public or still under development. Features to be added soon include:
- Realistic drone dynamics
- State estimators and controllers to simulate input delays
- ESDF generation for collision checks
- Metric-based splat generation
- Integration with Gymnasium for RL training
- Integration with other common simulators (IsaacSim, Gazebo, MuJoco)


## Overview

VizFlyt2 provides three core components that work seamlessly together:

- **Perception**: Photorealistic RGB and depth rendering using Gaussian Splatting (via Nerfstudio), with support for mono/stereo cameras and composable vision modules
- **Dynamics**: Lightweight point-mass model (~105 LOC) with velocity and acceleration control modes
- **Planning**: Trajectory primitives (line, circle, figure-8, spiral, waypoints) and reactive obstacle avoidance using potential fields

### Key Features

- **Modular composition** using `+` operator to chain perception modules
- **Unified planner interface** enabling easy switching between strategies (including RL agents)
- **NED coordinate system** for aerospace applications
- **Extensible architecture** for custom modules and planners

## Installation

```bash
git clone https://github.com/pearwpi/VizFlyt2.git
cd VizFlyt2
pip install -e .

# Optional: Install nerfstudio for perception
pip install nerfstudio
```

**Requirements**: Python 3.8+, CUDA GPU (11.8 and 12.6 tested, on a 30-series GPU. Compatibility mostly lies within the nerfstudio package)

## Quick Start

### Integrated Loop (Perception → Planning → Dynamics)

```python
from vizflyt2.perception.splat_render import SplatRenderer
from vizflyt2.planning import PotentialFieldPlanner
from vizflyt2.dynamics import PointMassDynamics
import numpy as np

# Initialize components
renderer = SplatRenderer("config.yml", "cam_settings.json")
planner = PotentialFieldPlanner(step_size=2.0)
dynamics = PointMassDynamics(
    initial_state={'position': np.array([0., 0., -50.]),
                   'velocity': np.array([0., 0., 0.]),
                   'orientation_rpy': np.array([0., 0., 0.])},
    control_mode='velocity'
)

# Simulation loop
for step in range(1000):
    pos, orient = dynamics.get_render_params()
    depth = renderer.render(pos, orient)['depth']
    velocity = planner.compute_action(depth_image=depth)['velocity']
    dynamics.step({'velocity': velocity}, dt=0.1)
    planner.step()
```

### Perception: Mono and Stereo Cameras

```python
from vizflyt2.perception.splat_render import SplatRenderer
from vizflyt2.perception.stereo_camera import StereoCamera

# Mono camera
renderer = SplatRenderer("config.yml", "cam_settings.json")
results = renderer.render(position, orientation_rpy)
rgb, depth = results['rgb'], results['depth']

# Stereo camera (6.5cm baseline)
stereo = StereoCamera("config.yml", "cam_settings.json", baseline=0.065)
results = stereo.render(position, orientation_rpy)
left_rgb, right_rgb = results['rgb_left'], results['rgb_right']
```

### Composable Vision Modules

```python
from vizflyt2.perception.modules import rgb_vision_module_factory
import cv2

@rgb_vision_module_factory
def edge_detector(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return {'edges': cv2.Canny(gray, 50, 150)}

# Compose with + operator
pipeline = renderer + edge_detector
results = pipeline.render(position, orientation_rpy)
# Access: results['rgb'], results['depth'], results['edges']
```

### Dynamics

```python
from vizflyt2.dynamics import PointMassDynamics

dynamics = PointMassDynamics(
    initial_state={'position': np.array([0., 0., -50.]),
                   'velocity': np.array([10., 0., 0.]),
                   'orientation_rpy': np.array([0., 0., 0.])},
    control_mode='velocity'  # or 'acceleration'
)

for i in range(1000):
    dynamics.step({'velocity': np.array([10., 5., 0.])}, dt=0.01)
```

### Planning

```python
from vizflyt2.planning import TrajectoryPlanner, PotentialFieldPlanner

# Trajectory planning
traj_planner = TrajectoryPlanner(dt=0.01)
traj_planner.plan_circle(center=np.array([0., 0., -50.]), radius=20., duration=10.)
while not traj_planner.is_complete():
    velocity = traj_planner.compute_action()['velocity']
    dynamics.step({'velocity': velocity}, dt=0.01)
    traj_planner.step()

# Reactive obstacle avoidance
reactive_planner = PotentialFieldPlanner(step_size=2.0)
velocity = reactive_planner.compute_action(depth_image=depth)['velocity']
```

## Project Structure

```
vizflyt2/
├── perception/          # Gaussian Splatting rendering and vision modules
│   ├── splat_render.py  # Mono camera renderer
│   ├── stereo_camera.py # Stereo camera renderer
│   ├── modules.py       # Module base classes and decorators
│   └── utils.py
├── dynamics/            # Point-mass dynamics model
│   ├── point_mass.py    # Simple integrator (~105 lines)
│   ├── base.py
│   └── utils.py
└── planning/            # Trajectory and reactive planning
    ├── trajectory.py    # Trajectory primitives
    ├── planner.py       # Potential field obstacle avoidance
    ├── primitives.py    # Low-level trajectory generators
    └── base.py

Additional resources:
├── QUICK_REFERENCE.md   # Code snippets and API reference
├── COMPOSITION_GUIDE.md # Detailed module composition guide
└── */EXAMPLES.md        # Example scripts documentation
```

## Examples

The repository includes several example scripts demonstrating key features:

- `perception/basic_examples.py` - Mono/stereo rendering basics
- `perception/custom_module_example.py` - Creating custom vision modules
- `planning/example_integration.py` - Complete perception-planning-dynamics loop
- See `perception/EXAMPLES.md` for a comprehensive guide

## Configuration

### Camera Settings (JSON, nerfstudio format)
```json
{
  "camera": {
    "c2w_matrix": [[...], [...], [...], [...]],
    "fov_radians": 1.3089969389957472,
    "render_resolution": 1080
  }
}
```

### Coordinate System (NED)
- **X**: North (forward), **Y**: East (right), **Z**: Down
- **Orientation**: Roll-Pitch-Yaw (radians), aerospace convention

## Documentation

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API quick reference
- [COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md) - Module composition patterns
- [vizflyt2/perception/EXAMPLES.md](vizflyt2/perception/EXAMPLES.md) - Example walkthrough
- [vizflyt2/dynamics/README.md](vizflyt2/dynamics/README.md) - Dynamics details
- [vizflyt2/planning/README.md](vizflyt2/planning/README.md) - Planning algorithms

## Performance

Typical rendering times (RTX 3090ti): ~200Hz

## License

## Acknowledgments

Built on [Nerfstudio](https://docs.nerf.studio/) for Gaussian Splatting rendering. Developed by Colin Balfour and the WPI PeAR Lab.
See [https://github.com/pearwpi/VizFlyt](https://github.com/pearwpi/VizFlyt) for the original implementation of the [VizFlyt paper](https://arxiv.org/abs/2503.22876v1)
```bibtex
@inproceedings{vizflyt2025,
  author    = {Kushagra Srivastava*, Rutwik Kulkarni*, Manoj Velmurugan*, Nitin J. Sanket},
  title     = {VizFlyt: An Open-Source Perception-Centric Hardware-In-The-Loop Framework for Aerial Robotics},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  note      = {Accepted for publication},
  url       = {https://github.com/pearwpi/VizFlyt}
}
```
