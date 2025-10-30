# Planning Module

Unified interface for trajectory planning and reactive control. All planners output control actions via `compute_action()`.

## Quick Start

### Reactive Planning (Depth → Velocity)

```python
from planning import PotentialFieldPlanner

planner = PotentialFieldPlanner(step_size=0.5)

for step in range(1000):
    depth = get_depth_image()
    action = planner.compute_action(depth_image=depth)
    
    dynamics.step({'velocity': action['velocity']}, dt=0.1)
    planner.step()
```

### Trajectory Following

```python
from planning import TrajectoryPlanner
import numpy as np

planner = TrajectoryPlanner(dt=0.01)
planner.plan_circle(np.array([0., 0., -50.]), radius=20., duration=10.)

planner.reset()
while not planner.is_complete():
    action = planner.compute_action()
    
    dynamics.step({'velocity': action['velocity']}, dt=0.01)
    planner.step()
    planner.current_index += 1
```

## Architecture

All planners inherit from `BasePlanner` and implement `compute_action(**kwargs)`:

```
BasePlanner
├── ReactiveVisualPlanner → PotentialFieldPlanner
└── TrajectoryPlanner (base) → TrajectoryPlanner
```

**Unified Interface:**
```python
action = planner.compute_action(**inputs)
velocity = action['velocity']  # Always present
position = action.get('position')  # Optional reference
info = action.get('info')  # Optional debug data
```

## Planners

### PotentialFieldPlanner

Reactive obstacle avoidance using depth images.

```python
planner = PotentialFieldPlanner(
    step_size=0.5,          # Forward speed (m/s)
    safety_radius=1.0,      # Avoidance scaling
    threshold=60,           # Obstacle detection threshold
    verbose=False           # Print debug info
)

action = planner.compute_action(depth_image=depth)
```

- **Input**: Depth image (obstacles = high values)
- **Output**: `{'velocity': [forward, lateral, vertical], 'info': {...}}`
- **Method**: Potential fields (attractive + repulsive forces)

### TrajectoryPlanner

Pre-computed smooth trajectories using primitives.

```python
planner = TrajectoryPlanner(dt=0.01)

# Available primitives
planner.plan_line(start, end, duration)
planner.plan_circle(center, radius, duration)
planner.plan_figure8(center, size, duration)
planner.plan_spiral(center, r_start, r_end, height_change, duration)
planner.plan_waypoints(waypoints, speeds=None)
```

- **Input**: None (uses internal trajectory)
- **Output**: `{'velocity': [...], 'position': [...], 'time': t}`

## Examples

### Basic Usage

```python
from planning import PotentialFieldPlanner, TrajectoryPlanner
import numpy as np

# Reactive planner
reactive = PotentialFieldPlanner(step_size=0.5, verbose=False)
depth = np.zeros((480, 640), dtype=np.uint8)
action = reactive.compute_action(depth_image=depth)
print(action['velocity'])

# Trajectory planner
traj = TrajectoryPlanner()
traj.plan_circle(np.array([0., 0., -50.]), 20., 10.)
action = traj.compute_action()
print(action['velocity'], action['position'])
```

### Runtime Switching

```python
# Start with trajectory
planner = TrajectoryPlanner()
planner.plan_line(start, end, 10.)

for step in range(1000):
    depth = get_depth_image()
    
    # Switch to reactive if obstacle detected
    if has_obstacle(depth) and isinstance(planner, TrajectoryPlanner):
        planner = PotentialFieldPlanner(step_size=0.5)
    
    # Unified interface works for both
    if isinstance(planner, PotentialFieldPlanner):
        action = planner.compute_action(depth_image=depth)
    else:
        action = planner.compute_action()
        planner.current_index += 1
    
    execute_action(action['velocity'])
    planner.step()
```

### Custom Planner (RL Agent)

```python
from planning import BasePlanner

class RLPlanner(BasePlanner):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def compute_action(self, observation=None, **kwargs):
        action = self.model.predict(observation)
        return {'velocity': action[:3]}

# Use like any other planner
rl_planner = RLPlanner(model)
action = rl_planner.compute_action(observation=obs)
```

## Integration with Dynamics

```python
from planning import PotentialFieldPlanner
from dynamics import PointMassDynamics
import numpy as np

# Setup
planner = PotentialFieldPlanner(step_size=0.5)
dynamics = PointMassDynamics(
    initial_state={
        'position': np.array([0., 0., -50.]),
        'velocity': np.array([0., 0., 0.]),
        'orientation_rpy': np.array([0., 0., 0.])
    },
    control_mode='velocity'
)

# Run loop
for step in range(1000):
    depth = get_depth_image()
    action = planner.compute_action(depth_image=depth)
    
    dynamics.step({'velocity': action['velocity']}, dt=0.1)
    planner.step()
```

See `example_integration.py` for complete perception + planning + dynamics pipeline.

## API Reference

### BasePlanner

```python
action = planner.compute_action(**kwargs)  # Get control action
planner.reset()                            # Reset planner
planner.step()                             # Increment counter
steps = planner.step_count                 # Get step count
```

### PotentialFieldPlanner

```python
PotentialFieldPlanner(
    step_size=0.5,              # Forward velocity
    z_step_size=0.2,            # Vertical scaling
    safety_radius=1.0,          # Motion scaling
    threshold=60,               # Obstacle threshold
    band_size=50,               # Boundary band width
    output_dir=None,            # Save visualizations
    verbose=False               # Print debug info
)

action = planner.compute_action(depth_image=depth)
# Returns: {'velocity': np.ndarray, 'info': dict}
```

### TrajectoryPlanner

```python
TrajectoryPlanner(dt=0.01)

# Planning
planner.plan_line(start, end, duration)
planner.plan_circle(center, radius, duration)
planner.plan_figure8(center, size, duration)
planner.plan_spiral(center, r_start, r_end, height_change, duration)
planner.plan_waypoints(waypoints, speeds=None)

# Execution
action = planner.compute_action()
# Returns: {'velocity': np.ndarray, 'position': np.ndarray, 'time': float}

# Helpers
planner.is_complete()          # Check if done
planner.get_trajectory()       # Get full trajectory
```

## Run Examples

```bash
cd planning

# Unified interface demo
python example_unified_interface.py

# Vision-based planning
python example_vision_planning.py

# Trajectory planning
python example_planning.py

# Full integration (perception + dynamics + planning)
python example_integration.py
```

## Philosophy

**Planners output actions, not just trajectories.** This supports:
- Reactive planning (depth → velocity)
- Trajectory following (time → velocity)
- RL agents (observation → action)
- Hybrid strategies

All planners share the same interface for easy composition and runtime switching.
