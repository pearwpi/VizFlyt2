# Differentiable Quadrotor Dynamics

PyTorch-based differentiable quadrotor dynamics for gradient-based learning and optimization.

## Features

- **Fully Differentiable**: All operations use PyTorch tensors, enabling gradient flow for:
  - Trajectory optimization
  - Inverse dynamics
  - Learning-based control
  - System identification

- **Realistic Controller Modeling**:
  - Fixed time delay (models sensor-to-actuator lag)
  - EMA filtering (models inner-loop control bandwidth)
  - Configurable response characteristics

- **Physics Effects**:
  - Linear + quadratic drag
  - Gravity
  - Semi-implicit Euler integration

- **Attitude Computation**:
  - Roll, pitch, yaw derived from thrust vector and desired heading
  - Smooth handling of singularities

- **Dual Interface**:
  - NumPy interface (compatible with VizFlyt2)
  - PyTorch interface (for gradient-based methods)

## Quick Start

### Basic Usage (NumPy Interface)

```python
from vizflyt2.dynamics import DifferentiableQuadrotorDynamics
import numpy as np

# Initialize
dynamics = DifferentiableQuadrotorDynamics(
    initial_state={
        'position': np.array([0., 0., -50.]),
        'velocity': np.array([2., 0., 0.]),
    },
    dt=0.1,
    ctrl_delay_sec=0.06,  # 60ms delay
    ctrl_alpha=0.25,      # EMA smoothing
    drag_quad=0.02,       # Quadratic drag
)

# Simulate
for step in range(100):
    controls = {
        'acceleration': np.array([0., 0., 1.]),  # m/s^2
        'frame': 'world'  # or 'body'
    }
    state = dynamics.step(controls, dt=0.1)
    print(f"Position: {state['position']}")
```

### Gradient-Based Optimization (PyTorch Interface)

```python
import torch

# Initialize learnable action sequence
n_steps = 50
actions = torch.zeros(n_steps, 3, requires_grad=True)
optimizer = torch.optim.Adam([actions], lr=0.5)

target = torch.tensor([20., 10., -45.])

# Optimize
for iteration in range(100):
    optimizer.zero_grad()
    
    # Reset and simulate
    dynamics.physics.reset(p0, v0, att0)
    for t in range(n_steps):
        result = dynamics.physics.step(actions[t])
        final_pos = result['p_next']
    
    # Compute loss and backprop
    loss = torch.norm(final_pos - target)
    loss.backward()
    optimizer.step()
```

### Integration with VizFlyt2 Planning

```python
from vizflyt2.planning import PotentialFieldPlanner
from vizflyt2.dynamics import DifferentiableQuadrotorDynamics

# Works seamlessly with existing planners
dynamics = DifferentiableQuadrotorDynamics()
planner = PotentialFieldPlanner()

for step in range(1000):
    # Planning
    depth = get_depth_image()
    action = planner.compute_action(depth_image=depth)
    
    # Dynamics
    controls = {'acceleration': action['velocity'], 'frame': 'world'}
    state = dynamics.step(controls, dt=0.1)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 1/15 | Simulation timestep (seconds) |
| `mass` | 1.0 | Quadrotor mass (kg) |
| `gravity` | (0, 0, -9.81) | Gravity vector (m/s²) |
| `drag_lin` | 0.0 | Linear drag coefficient |
| `drag_quad` | 0.02 | Quadratic drag coefficient |
| `ctrl_delay_sec` | 0.06 | Controller delay (seconds) |
| `ctrl_alpha` | 0.25 | EMA filter coefficient (0=no smooth, 1=no memory) |
| `action_is_normalized` | False | If True, actions are [-1,1] scaled by `a_max` |
| `a_max` | (12, 12, 12) | Maximum acceleration per axis (m/s²) |

## Control Inputs

The `step()` method accepts a dictionary with:

- **`acceleration`** (required): (3,) thrust acceleration [ax, ay, az] in m/s²
- **`frame`** (optional): `'world'` or `'body'` coordinate frame (default: `'world'`)
- **`heading`** (optional): (2,) or (3,) desired XY heading for yaw alignment

## State

The dynamics maintains:

- **`position`**: (3,) [x, y, z] in NED frame (meters)
- **`velocity`**: (3,) [vx, vy, vz] in world frame (m/s)
- **`orientation_rpy`**: (3,) [roll, pitch, yaw] in radians
- **`acceleration`**: (3,) current thrust acceleration (m/s²)

## Advanced Usage

### Access Torch State Directly

```python
# Get state as torch tensors (with gradients)
torch_state = dynamics.get_torch_state()
position = torch_state['position']  # torch.Tensor with grad

# Set state from torch tensors
new_state = {
    'position': torch.tensor([1., 2., -50.]),
    'velocity': torch.tensor([0., 0., 0.]),
}
dynamics.set_torch_state(new_state)
```

### Custom Device/Dtype

```python
import torch

# Use GPU
dynamics = DifferentiableQuadrotorDynamics(
    device=torch.device('cuda'),
    dtype=torch.float64,  # Double precision
)
```

## Examples

Run the complete example:

```bash
cd vizflyt2/dynamics
python example_differentiable.py
```

This demonstrates:
1. Basic simulation with NumPy interface
2. Gradient-based trajectory optimization
3. Body-frame control
4. Visualization of results

## Comparison with PointMassDynamics

| Feature | PointMassDynamics | DifferentiableQuadrotorDynamics |
|---------|-------------------|--------------------------------|
| Framework | NumPy | PyTorch |
| Gradients | ❌ No | ✅ Yes |
| Controller delay | ❌ No | ✅ Yes (configurable) |
| EMA filtering | ❌ No | ✅ Yes |
| Drag | Linear only | Linear + quadratic |
| Attitude | Simple | From thrust + heading |
| GPU support | ❌ No | ✅ Yes |
| Speed | Fast | Moderate (GPU helps) |

## Use Cases

- **Learning-based control**: Train RL policies or neural controllers
- **Trajectory optimization**: Find optimal trajectories with gradient descent
- **System identification**: Learn dynamics parameters from data
- **Inverse problems**: Compute actions to achieve desired states
- **Sensitivity analysis**: Understand how actions affect future states

## Notes

- For pure simulation (no gradients needed), `PointMassDynamics` may be faster
- Controller delay + EMA provides more realistic response than instantaneous
- Body-frame control is useful for vehicle-centric policies
- The physics core is reusable outside VizFlyt2 (see `DifferentiablePointMass` class)
