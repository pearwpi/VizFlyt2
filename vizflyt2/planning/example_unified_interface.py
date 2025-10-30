"""
Example: Unified Planning Interface

All planners in the module implement compute_action(**kwargs) → Dict[str, np.ndarray]
This allows flexible, composable planning strategies.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from planning import TrajectoryPlanner, PotentialFieldPlanner, BasePlanner

print("=" * 70)
print("UNIFIED PLANNING INTERFACE")
print("=" * 70)

# Example 1: Trajectory-Following Planner
print("\n1. Trajectory-Following Planner")
print("-" * 70)

traj_planner = TrajectoryPlanner(dt=0.01)
traj_planner.plan_figure8(center=np.array([0., 0., -50.]), size=15., duration=10.0)

# Use compute_action() - unified interface
traj_planner.reset()
for i in range(5):
    action = traj_planner.compute_action()
    velocity = action['velocity']
    position_ref = action['position']
    time = action['time']
    
    print(f"  Step {i}: t={time:.2f}s, vel=[{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
    
    # Send to dynamics
    # dynamics.set_control(velocity)
    # dynamics.step()
    
    traj_planner.step()
    traj_planner.current_index += 1

print(f"  ✓ Total steps: {traj_planner.step_count}")

# Example 2: Reactive Visual Planner
print("\n2. Reactive Visual Planner")
print("-" * 70)

visual_planner = PotentialFieldPlanner(step_size=0.5, verbose=False)

# Simulate different obstacle scenarios
scenarios = [
    ("Clear path", np.zeros((480, 640), dtype=np.uint8)),
    ("Central obstacle", lambda: np.where(
        (np.mgrid[0:480, 0:640][0] > 200) & (np.mgrid[0:480, 0:640][0] < 280) &
        (np.mgrid[0:480, 0:640][1] > 280) & (np.mgrid[0:480, 0:640][1] < 360),
        150, 0
    ).astype(np.uint8)),
    ("Left obstacle", lambda: np.where(
        (np.mgrid[0:480, 0:640][1] < 200), 150, 0
    ).astype(np.uint8))
]

for name, depth in scenarios:
    if callable(depth):
        depth = depth()
    
    action = visual_planner.compute_action(depth_image=depth)
    velocity = action['velocity']
    
    print(f"  {name}: vel=[{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
    visual_planner.step()

print(f"  ✓ Total steps: {visual_planner.step_count}")

# Example 3: Generic Planner Function (works with any planner)
print("\n3. Generic Planner Function")
print("-" * 70)

def execute_planner(planner: BasePlanner, num_steps: int, **kwargs):
    """
    Execute any planner for num_steps using unified interface.
    
    Args:
        planner: Any planner inheriting from BasePlanner
        num_steps: Number of steps to execute
        **kwargs: Planner-specific inputs
    """
    planner.reset()
    actions = []
    
    for i in range(num_steps):
        action = planner.compute_action(**kwargs)
        actions.append(action)
        planner.step()
        
        # Advance trajectory planner if applicable
        if hasattr(planner, 'current_index'):
            planner.current_index += 1
            if hasattr(planner, 'is_complete') and planner.is_complete():
                break
    
    return actions

# Works with TrajectoryPlanner
traj_planner2 = TrajectoryPlanner()
traj_planner2.plan_circle(np.array([0., 0., -50.]), radius=15., duration=5.)
actions_traj = execute_planner(traj_planner2, num_steps=10)
print(f"  ✓ Executed TrajectoryPlanner: {len(actions_traj)} actions")

# Works with PotentialFieldPlanner
depth_img = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
visual_planner2 = PotentialFieldPlanner(step_size=0.8, verbose=False)
actions_visual = execute_planner(visual_planner2, num_steps=10, depth_image=depth_img)
print(f"  ✓ Executed PotentialFieldPlanner: {len(actions_visual)} actions")

# Example 4: Switching Between Planners
print("\n4. Hybrid Planning Strategy")
print("-" * 70)

# Start with trajectory following
current_planner = TrajectoryPlanner()
current_planner.plan_line(
    start=np.array([0., 0., -50.]),
    end=np.array([100., 0., -50.]),
    duration=10.
)

# Simulate hybrid behavior
for step in range(10):
    # Check if we need to switch to reactive planning
    # (e.g., if we detect obstacles)
    obstacle_detected = step > 5  # Simulated detection
    
    if obstacle_detected and not isinstance(current_planner, PotentialFieldPlanner):
        print("  ! Obstacle detected - switching to reactive planner")
        current_planner = PotentialFieldPlanner(step_size=0.5, verbose=False)
    
    # Compute action based on current planner
    if isinstance(current_planner, PotentialFieldPlanner):
        depth = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
        action = current_planner.compute_action(depth_image=depth)
    else:
        action = current_planner.compute_action()
        current_planner.current_index += 1
    
    velocity = action['velocity']
    print(f"  Step {step}: planner={current_planner.__class__.__name__}, "
          f"vel=[{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
    
    current_planner.step()

print(f"  ✓ Hybrid planning executed successfully")

# Example 5: Future Extension - Custom Planner
print("\n5. Custom Planner Example (Template)")
print("-" * 70)

from planning import BasePlanner

class RLAgentPlanner(BasePlanner):
    """
    Template for RL agent planner.
    
    Maps: observation → action
    """
    
    def __init__(self, model=None):
        super().__init__()
        self.model = model  # Load your trained RL model here
    
    def compute_action(self, observation=None, **kwargs):
        """
        Compute action from observation using RL policy.
        
        Args:
            observation: State/perception observation
            **kwargs: Additional context
        
        Returns:
            Dictionary with 'velocity' and optionally 'angular_velocity'
        """
        # Placeholder: would call self.model.predict(observation)
        # For now, return random action
        velocity = np.random.randn(3) * 0.5
        velocity[0] = abs(velocity[0])  # Keep forward positive
        
        return {
            'velocity': velocity,
            'info': {'model_confidence': 0.95}
        }

# Create custom planner
rl_planner = RLAgentPlanner()
observation = np.random.randn(128)  # Mock observation vector
action = rl_planner.compute_action(observation=observation)
print(f"  ✓ Custom RLAgentPlanner: vel={action['velocity']}")
print(f"  ✓ Is BasePlanner: {isinstance(rl_planner, BasePlanner)}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("  • All planners implement compute_action(**kwargs)")
print("  • Supports reactive, trajectory-following, and RL planners")
print("  • Easy to switch between planners at runtime")
print("  • Extensible for custom planning strategies")
print("=" * 70)
