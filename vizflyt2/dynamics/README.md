# Dynamics Module

Physics models for quadrotor simulation - from simple to differentiable.

## Available Models

### 1. **PointMassDynamics** (Simple & Fast)
Super simple point-mass dynamics for trajectory generation.

- Just basic physics - position, velocity, orientation, angular velocity
- Two control modes:
  - **Velocity mode**: Set velocity directly (kinematic, no physics)
  - **Acceleration mode**: Set acceleration (optionally enable gravity)
- No mass, no inertia, no forces, no torques
- NumPy-based, fast for simulation

### 2. **DifferentiableQuadrotorDynamics** (Gradient-Based Learning)
PyTorch-based differentiable dynamics for optimization and learning.

- **Fully differentiable** - gradients flow through all operations
- **Realistic controller modeling** - delay + EMA filtering
- **Physics effects** - linear + quadratic drag, gravity
- **Attitude computation** - from thrust direction + heading
- Use for: RL training, trajectory optimization, inverse dynamics

See [`DIFFERENTIABLE_README.md`](DIFFERENTIABLE_README.md) for details.

## Quick Examples

### Velocity Mode

Direct control - useful for scripted trajectories:

```python
import numpy as np
from dynamics import PointMassDynamics

dynamics = PointMassDynamics(
    initial_state={
        'position': np.array([0., 0., -50.]),
        'velocity': np.array([10., 0., 0.]),
        'orientation_rpy': np.array([0., 0., 0.])
    },
    control_mode='velocity'
)

dt = 0.01
for i in range(1000):
    # Set velocity directly
    dynamics.step({
        'velocity': np.array([10., 5., 0.]),
        'angular_velocity': np.array([0., 0., 0.1])  # optional
    }, dt)
    
    position, orientation = dynamics.get_render_params()
```

### Acceleration Mode

Physics-based but simple:

```python
dynamics = PointMassDynamics(
    initial_state={
        'position': np.array([0., 0., -50.]),
        'velocity': np.array([10., 0., 0.]),
        'orientation_rpy': np.array([0., 0., 0.])
    },
    control_mode='acceleration',
    gravity=True  # Add 9.81 m/s² downward
)

for i in range(1000):
    state = dynamics.get_state()
    altitude = -state['position'][2]
    
    # Simple controller
    az = -9.81 + 2.0 * (50.0 - altitude)  # Cancel gravity + correction
    
    dynamics.step({
        'acceleration': np.array([0., 0., az]),
        'angular_velocity': np.array([0., 0., 0.])  # optional
    }, dt)
```

## State

```python
state = {
    'position': np.array([x, y, z]),           # NED frame (m)
    'velocity': np.array([vx, vy, vz]),        # NED frame (m/s)
    'orientation_rpy': np.array([r, p, y]),    # radians
    'angular_velocity': np.array([wx, wy, wz]) # rad/s
}
```

## Controls

**Velocity Mode:**
```python
controls = {
    'velocity': np.array([vx, vy, vz]),        # Required
    'angular_velocity': np.array([wx, wy, wz]) # Optional (default: current)
}
```

**Acceleration Mode:**
```python
controls = {
    'acceleration': np.array([ax, ay, az]),    # Required
    'angular_velocity': np.array([wx, wy, wz]) # Optional (default: current)
}
```

## Integration with Rendering

```python
from dynamics import PointMassDynamics
from perception.splat_render import SplatRenderer

dynamics = PointMassDynamics(initial_state, control_mode='velocity')
renderer = SplatRenderer("config.yml", "cam.json")

for i in range(num_frames):
    # Circular trajectory
    t = dynamics.get_time()
    vx = -10 * np.sin(0.5 * t)
    vy = 10 * np.cos(0.5 * t)
    
    dynamics.step({'velocity': np.array([vx, vy, 0.])}, dt)
    
    # Render
    position, orientation = dynamics.get_render_params()
    result = renderer.render(position, orientation)
    save_frame(result['rgb'])
```

## Example

```bash
cd dynamics
python example_simple.py
```

Runs both modes and generates plots.

## API

```python
# Create
dynamics = PointMassDynamics(initial_state, control_mode='velocity'|'acceleration', gravity=False)

# Step
state = dynamics.step(controls, dt)

# Query
position, orientation = dynamics.get_render_params()
velocity = dynamics.get_velocity()
speed = dynamics.get_speed()
altitude = dynamics.get_altitude_agl()
```

## Coordinate System

**NED (North-East-Down):**
- X: North
- Y: East  
- Z: Down (altitude is -Z)

**Orientation (Roll-Pitch-Yaw):**
- Roll: Rotation about X
- Pitch: Rotation about Y
- Yaw: Rotation about Z

## That's It

105 lines of code. Simple as it gets.
