"""
Differentiable Quadrotor Dynamics

Wraps DifferentiablePointMass for use in VizFlyt2, providing:
- PyTorch-based physics with gradient flow
- Delay + EMA controller modeling
- Quadratic/linear drag
- Attitude computation from thrust vector
- Compatible with VizFlyt2 DynamicsModel interface
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from .base import DynamicsModel


class DifferentiableQuadrotorDynamics(DynamicsModel):
    """
    Differentiable quadrotor dynamics using PyTorch for gradient-based learning.
    
    Features:
    - Point-mass approximation with realistic controller response
    - Fixed delay + EMA filtering (inner-loop dynamics)
    - Quadratic + linear drag
    - Attitude derived from thrust direction and heading
    - All operations are differentiable (torch tensors)
    
    Control Inputs:
    - 'acceleration': (3,) thrust acceleration command [ax, ay, az] m/s^2
    - 'heading': (optional) (2,) desired XY heading [hx, hy] for yaw
    - 'frame': (optional) 'world' or 'body' frame for acceleration input
    
    State (in user's coordinate system - NED or Blender):
    - position: (3,) [x, y, z] (meters)
    - velocity: (3,) [vx, vy, vz] in world frame (m/s)
    - orientation_rpy: (3,) [roll, pitch, yaw] in radians
    - acceleration: (3,) current thrust acceleration (m/s^2)
    
    Coordinate Systems:
    - NED: X=forward/North, Y=right/East, Z=down, gravity=(0,0,+9.81)
    - Blender: X=right, Y=forward, Z=up, gravity=(0,0,-9.81)
    Internally uses Blender, converts at API boundary.
    """
    
    def __init__(
        self,
        initial_state: Optional[Dict[str, np.ndarray]] = None,
        dt: float = 1.0/15.0,
        mass: float = 1.0,
        gravity: Optional[Tuple[float, float, float]] = None,
        drag_lin: float = 0.0,
        drag_quad: float = 0.02,
        ctrl_delay_sec: float = 0.06,
        ctrl_alpha: float = 0.5,
        action_is_normalized: bool = False,
        a_max: Tuple[float, float, float] = (12., 12., 12.),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        coordinate_system: str = 'NED',
    ):
        """
        Initialize differentiable quadrotor dynamics.
        
        Args:
            initial_state: Initial conditions (position, velocity, orientation_rpy)
            dt: Simulation timestep (seconds)
            mass: Quadrotor mass (kg)
            gravity: Gravity vector [gx, gy, gz] (m/s^2) in the coordinate_system frame.
                     If None, defaults to (0,0,9.81) for NED or (0,0,-9.81) for Blender.
            drag_lin: Linear drag coefficient
            drag_quad: Quadratic drag coefficient
            ctrl_delay_sec: Controller delay (seconds)
            ctrl_alpha: EMA smoothing factor (0=no smoothing, 1=no memory)
            action_is_normalized: If True, actions are in [-1, 1] and scaled by a_max
            a_max: Maximum acceleration per axis (m/s^2)
            device: PyTorch device (defaults to CPU)
            dtype: PyTorch dtype (defaults to float32)
            coordinate_system: 'NED' or 'Blender' coordinate frame
                NED: X=forward/North, Y=right/East, Z=down, gravity=(0,0,+9.81)
                Blender: X=right, Y=forward, Z=up, gravity=(0,0,-9.81)
        """
        # Set up default initial state if not provided
        if initial_state is None:
            initial_state = {
                'position': np.array([0., 0., 0.], dtype=np.float32),
                'velocity': np.array([0., 0., 0.], dtype=np.float32),
                'orientation_rpy': np.array([0., 0., 0.], dtype=np.float32),
            }
        
        super().__init__(initial_state)
        
        # Store configuration
        self.dt = dt
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.action_is_normalized = action_is_normalized
        self.coordinate_system = coordinate_system.upper()
        
        # Validate coordinate system
        if self.coordinate_system not in ['NED', 'BLENDER']:
            raise ValueError(f"coordinate_system must be 'NED' or 'Blender', got '{coordinate_system}'")
        
        # Set default gravity based on coordinate system if not provided
        if gravity is None:
            if self.coordinate_system == 'NED':
                gravity = (0., 0., 9.81)  # NED: Z is down, positive gravity
            else:
                gravity = (0., 0., -9.81)  # Blender: Z is up, negative gravity
        
        # Convert gravity to Blender frame if needed
        # NED gravity is (0, 0, +9.81), Blender gravity is (0, 0, -9.81)
        if self.coordinate_system == 'NED':
            gravity_blender = self._ned_to_blender(np.array(gravity))
            gravity = tuple(gravity_blender)
        else:
            # Already in Blender format, use as-is
            pass
        
        # Initialize the differentiable physics core (always in Blender internally)
        self.physics = DifferentiablePointMass(
            dt=dt,
            mass=mass,
            gravity=gravity,
            drag_lin=drag_lin,
            drag_quad=drag_quad,
            ctrl_delay_sec=ctrl_delay_sec,
            ctrl_alpha=ctrl_alpha,
            action_is_normalized=action_is_normalized,
            a_max=a_max,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Initialize physics state from initial_state (convert NED to Blender if needed)
        pos_init = initial_state['position']
        vel_init = initial_state.get('velocity', [0., 0., 0.])
        att_init = initial_state.get('orientation_rpy', [0., 0., 0.])
        
        if self.coordinate_system == 'NED':
            pos_init = self._ned_to_blender(np.array(pos_init))
            vel_init = self._ned_to_blender(np.array(vel_init))
            att_init = np.array([att_init[1], att_init[0], att_init[2]])  # swap roll/pitch
        
        p0 = torch.tensor(pos_init, device=self.device, dtype=self.dtype)
        v0 = torch.tensor(vel_init, device=self.device, dtype=self.dtype)
        att0 = torch.tensor(att_init, device=self.device, dtype=self.dtype)
        self.physics.reset(p0, v0, att0)
        
        # Keep numpy copy of state for VizFlyt2 interface
        self._update_state_from_physics()
    
    def _blender_to_ned(self, vec: np.ndarray) -> np.ndarray:
        """Convert Blender (X=right, Y=forward, Z=up) to NED (X=forward, Y=right, Z=down)."""
        # Transformation: (Bx, By, Bz) → (By, Bx, -Bz)
        return np.array([vec[1], vec[0], -vec[2]])
    
    def _ned_to_blender(self, vec: np.ndarray) -> np.ndarray:
        """Convert NED (X=forward, Y=right, Z=down) to Blender (X=right, Y=forward, Z=up)."""
        # Inverse transformation: (Nx, Ny, Nz) → (Ny, Nx, -Nz)
        return np.array([vec[1], vec[0], -vec[2]])
    
    def _blender_to_ned_torch(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert Blender to NED for torch tensors."""
        return torch.stack([vec[1], vec[0], -vec[2]])
    
    def _ned_to_blender_torch(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert NED to Blender for torch tensors."""
        return torch.stack([vec[1], vec[0], -vec[2]])
    
    def _update_state_from_physics(self):
        """Sync numpy state dict from torch physics state."""
        pos = self.physics.p.detach().cpu().numpy()
        vel = self.physics.v.detach().cpu().numpy()
        att = self.physics.att_euler.detach().cpu().numpy()
        acc = self.physics.a_ema.detach().cpu().numpy()
        
        # Convert from Blender (internal) to NED (external) if needed
        if self.coordinate_system == 'NED':
            self.state['position'] = self._blender_to_ned(pos)
            self.state['velocity'] = self._blender_to_ned(vel)
            self.state['acceleration'] = self._blender_to_ned(acc)
            # Attitude: In Blender roll=X, pitch=Y, yaw=Z
            # In NED: roll=X, pitch=Y, yaw=Z but axes swapped
            # Roll is about body-X, pitch about body-Y, yaw about body-Z
            # After coordinate swap: roll and pitch swap, yaw stays same
            self.state['orientation_rpy'] = np.array([att[1], att[0], att[2]])  # swap roll/pitch
        else:
            self.state['position'] = pos
            self.state['velocity'] = vel
            self.state['orientation_rpy'] = att
            self.state['acceleration'] = acc
    
    def step(self, controls: Dict[str, Union[np.ndarray, torch.Tensor]], dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Advance simulation by one timestep.
        
        Args:
            controls: Dictionary with:
                - 'acceleration': (3,) thrust acceleration command [ax, ay, az] m/s^2
                - 'heading': (optional) (2,) or (3,) XY heading [hx, hy] or [hx, hy, hz]
                - 'frame': (optional) 'world' or 'body' (default: 'world')
            dt: Timestep (if None, uses self.dt)
        
        Returns:
            Updated state dictionary (numpy arrays)
        """
        if dt is not None and abs(dt - self.dt) > 1e-6:
            raise ValueError(f"dt={dt} does not match physics dt={self.dt}. Create new instance for different dt.")
        
        # Extract controls
        accel = controls.get('acceleration', controls.get('velocity', np.zeros(3)))
        heading = controls.get('heading', None)
        frame = controls.get('frame', 'world')
        
        # Convert NED to Blender if needed (user provides NED, physics expects Blender)
        if self.coordinate_system == 'NED':
            if isinstance(accel, np.ndarray):
                accel = self._ned_to_blender(accel)
            if heading is not None and isinstance(heading, np.ndarray):
                heading = self._ned_to_blender(heading)
        
        # Convert to torch tensors if needed
        if isinstance(accel, np.ndarray):
            accel = torch.tensor(accel, device=self.device, dtype=self.dtype)
        if heading is not None and isinstance(heading, np.ndarray):
            heading = torch.tensor(heading, device=self.device, dtype=self.dtype)
            # Ensure heading is 3D (add zero z-component if 2D)
            if heading.shape[0] == 2:
                heading = torch.cat([heading, torch.zeros(1, device=self.device, dtype=self.dtype)])
        
        # For NED, also need to convert torch heading
        if self.coordinate_system == 'NED' and heading is not None and not isinstance(controls.get('heading'), np.ndarray):
            heading = self._ned_to_blender_torch(heading)
        
        # Step physics
        result = self.physics.step(
            action_cmd=accel,
            heading_xy=heading,
            add_gravity=True,
            frame=frame
        )
        
        # Update time
        self.time += self.dt
        
        # Sync state back to numpy
        self._update_state_from_physics()
        
        return self.state.copy()
    
    def get_render_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current position and orientation for rendering.
        
        Returns:
            position: (3,) [x, y, z] in NED (meters)
            orientation_rpy: (3,) [roll, pitch, yaw] (radians)
        """
        return self.state['position'].copy(), self.state['orientation_rpy'].copy()
    
    def reset(self, initial_state: Optional[Dict[str, np.ndarray]] = None):
        """
        Reset dynamics to initial state.
        
        Args:
            initial_state: New initial state (if None, uses constructor defaults)
        """
        if initial_state is not None:
            super().reset(initial_state)
        
        # Get state (in user's coordinate system)
        pos = self.state['position']
        vel = self.state.get('velocity', np.array([0., 0., 0.]))
        att = self.state.get('orientation_rpy', np.array([0., 0., 0.]))
        
        # Convert to Blender (internal) if user provided NED
        if self.coordinate_system == 'NED':
            pos = self._ned_to_blender(pos)
            vel = self._ned_to_blender(vel)
            att = np.array([att[1], att[0], att[2]])  # swap roll/pitch back
        
        p0 = torch.tensor(pos, device=self.device, dtype=self.dtype)
        v0 = torch.tensor(vel, device=self.device, dtype=self.dtype)
        att0 = torch.tensor(att, device=self.device, dtype=self.dtype)
        
        self.physics.reset(p0, v0, att0)
        self.time = 0.0
        self._update_state_from_physics()
    
    def get_torch_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current state as torch tensors (useful for gradient-based methods).
        
        Returns:
            Dictionary with torch tensors (references to internal state)
        """
        return {
            'position': self.physics.p,
            'velocity': self.physics.v,
            'orientation_rpy': self.physics.att_euler,
            'acceleration': self.physics.a_ema,
        }
    
    def set_torch_state(self, state: Dict[str, torch.Tensor]):
        """
        Set state directly from torch tensors (for gradient-based optimization).
        
        Args:
            state: Dictionary with torch tensors
        """
        if 'position' in state:
            self.physics.p = state['position'].to(device=self.device, dtype=self.dtype)
        if 'velocity' in state:
            self.physics.v = state['velocity'].to(device=self.device, dtype=self.dtype)
        if 'orientation_rpy' in state:
            self.physics.att_euler = state['orientation_rpy'].to(device=self.device, dtype=self.dtype)
        
        self._update_state_from_physics()


# Import the differentiable physics core
class DifferentiablePointMass:
    """
    Differentiable point-mass quadrotor surrogate with:
      - desired thrust acceleration a_cmd (world frame)
      - fixed delay + EMA (inner-loop response)
      - quadratic/linear drag
      - semi-implicit Euler integration
      - attitude from thrust direction + yaw heading
    All internal state is torch.Tensor so gradients can flow through actions.
    """

    def __init__(
        self,
        dt=1.0/15.0,
        mass=1.0,
        gravity=(0., 0., -9.81),
        drag_lin=0.0,
        drag_quad=0.02,
        ctrl_delay_sec=0.06,
        ctrl_alpha=0.25,
        action_is_normalized=True,
        a_max=(12., 12., 12.),
        device=None,
        dtype=torch.float32,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Scalars / vectors
        self.dt   = torch.as_tensor(dt,   device=self.device, dtype=self.dtype)
        self.mass = torch.as_tensor(mass, device=self.device, dtype=self.dtype)
        self.g    = torch.tensor(gravity, device=self.device, dtype=self.dtype)

        self.drag_lin  = torch.as_tensor(drag_lin,  device=self.device, dtype=self.dtype)
        self.drag_quad = torch.as_tensor(drag_quad, device=self.device, dtype=self.dtype)

        self.ctrl_alpha = torch.as_tensor(ctrl_alpha, device=self.device, dtype=self.dtype)
        self.delay_steps = max(1, int(round(float(ctrl_delay_sec) / float(dt))))

        self.action_is_normalized = action_is_normalized
        self.a_max = torch.tensor(a_max, device=self.device, dtype=self.dtype)

        # State (torch tensors)
        self.p = torch.zeros(3, device=self.device, dtype=self.dtype)   # position
        self.v = torch.zeros(3, device=self.device, dtype=self.dtype)   # velocity
        self.att_euler = torch.zeros(3, device=self.device, dtype=self.dtype)  # roll, pitch, yaw

        # Controller memory
        self.a_delay = torch.zeros(self.delay_steps, 3, device=self.device, dtype=self.dtype)  # FIFO buffer
        self.a_ema   = torch.zeros(3, device=self.device, dtype=self.dtype)  # EMA state

    # ---------- utilities (all torch) ----------

    def decode_action(self, action):
        """
        Map action to world-frame thrust acceleration (m/s^2).
        If normalized, clamp to [-1,1] then scale by a_max (tanh is preferable in the policy).
        """
        a = action
        if self.action_is_normalized:
            a = torch.clamp(a, -1.0, 1.0) * self.a_max
        else:
            a = torch.clamp(a, -self.a_max, self.a_max)
        return a

    def apply_delay_ema(self, a_cmd):
        """
        Fixed L-step delay + EMA smoothing:
          a_delayed = a_delay[0]
          a_ema <- (1 - alpha) * a_ema + alpha * a_delayed
        Delay implemented via differentiable shift (no in-place detach).
        """
        # Delayed value is current head
        a_delayed = self.a_delay[0]

        # Shift buffer left and append a_cmd at the end
        self.a_delay = torch.cat([self.a_delay[1:], a_cmd.unsqueeze(0)], dim=0)

        # EMA update
        self.a_ema = (1.0 - self.ctrl_alpha) * self.a_ema + self.ctrl_alpha * a_delayed
        return self.a_ema

    def drag_accel(self, v):
        """
        a_drag = -(k1*v + k2*|v|*v)  (per-mass form)
        """
        speed = torch.norm(v) + 1e-9
        return -(self.drag_lin * v + self.drag_quad * speed * v)

    @staticmethod
    def _normalize(x, eps=1e-9):
        return x / (torch.norm(x) + eps)

    def attitude_from_thrust_and_heading(self, thrust_world, heading_xy):
        """
        Compute Euler XYZ (roll, pitch, yaw) from:
          - body z-axis aligned with thrust direction
          - yaw aligned with heading in XY
        Fully torch; small eps added for numerical robustness.
        """
        world_up = torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype)

        # Body z-axis from thrust (keep upright fallback via tiny blend)
        t = thrust_world
        # Smooth-ish fallback: add a tiny up component to avoid zero/degenerate cases
        bz = self._normalize(t + 1e-8 * world_up)

        # Desired heading on XY
        h = heading_xy
        h = self._normalize(h)

        # Project heading onto plane orthogonal to bz to get body y-axis
        by = h - torch.dot(h, bz) * bz
        by = self._normalize(by + 1e-8 * torch.tensor([0., 1., 0.], device=self.device, dtype=self.dtype))

        # Right-handed body x-axis
        bx = torch.cross(by, bz, dim=0)
        bx = self._normalize(bx)

        # Rotation matrix R = [bx by bz] (world<-body), columns are body axes
        R = torch.stack([bx, by, bz], dim=1)

        # Euler XYZ (Blender: roll=X, pitch=Y, yaw=Z)
        # pitch = asin(-R[2,0]); roll = atan2(R[2,1], R[2,2]); yaw = atan2(R[1,0], R[0,0])
        pitch = torch.asin(torch.clamp(-R[2, 0], -1.0 + 1e-7, 1.0 - 1e-7))
        roll  = torch.atan2(R[2, 1], R[2, 2])
        yaw   = torch.atan2(R[1, 0], R[0, 0])
        return torch.stack([roll, pitch, yaw], dim=0)

    # ---------- core API ----------

    def reset(self, p0, v0=None, att0=None):
        """
        Initialize (or re-initialize) the physics state.
        p0, v0, att0 are torch tensors on the same device/dtype (or will be converted).
        """
        self.p = torch.as_tensor(p0, device=self.device, dtype=self.dtype).clone()
        self.v = torch.zeros(3, device=self.device, dtype=self.dtype) if v0 is None \
                 else torch.as_tensor(v0, device=self.device, dtype=self.dtype).clone()
        self.att_euler = torch.zeros(3, device=self.device, dtype=self.dtype) if att0 is None \
                         else torch.as_tensor(att0, device=self.device, dtype=self.dtype).clone()

        self.a_delay = torch.zeros(self.delay_steps, 3, device=self.device, dtype=self.dtype)
        self.a_ema   = torch.zeros(3, device=self.device, dtype=self.dtype)

    def step(self, action_cmd, heading_xy=None, add_gravity=True, frame='body'):
        """
        One differentiable physics step.
        Args:
          action_cmd: torch(3,) desired thrust acceleration in world frame (normalized or physical)
          heading_xy: torch(3,) desired XY heading for yaw alignment (default +Y)
          add_gravity: if True, include gravity in a_total; if False, treat action as net accel beyond gravity
          frame: 'body' or 'world' - coordinate frame for action_cmd
        Returns:
          dict with p_next, v_next, a_thrust, attitude (all torch), and delta_pos (torch)
        """
        if heading_xy is None:
            heading_xy = torch.tensor([0., 1., 0.], device=self.device, dtype=self.dtype)
            
        if frame == 'body':
            # Convert body-frame action to world-frame using current attitude
            roll, pitch, yaw = self.att_euler
            cy = torch.cos(yaw);   sy = torch.sin(yaw)
            cp = torch.cos(pitch); sp = torch.sin(pitch)
            cr = torch.cos(roll);  sr = torch.sin(roll)
            # Create rotation matrix using torch.stack to preserve gradients
            R_bw = torch.stack([
                torch.stack([cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy]),
                torch.stack([cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy]),
                torch.stack([-sp,     sr * cp,                cr * cp])
            ])  # body->world rotation matrix [3,3]
            action_cmd = R_bw @ action_cmd  # now in world frame

        print('action_cmd:', action_cmd)
        action_cmd = action_cmd - self.g
        print('action_cmd after gravity compensation:', action_cmd)
        
        # Scale / clamp action
        a_cmd = self.decode_action(action_cmd)

        # Inner-loop delay + EMA filtering
        a_thrust = a_cmd#self.apply_delay_ema(a_cmd)

        # Total acceleration
        a_drag = self.drag_accel(self.v)
        a_total = a_thrust + a_drag + (self.g if add_gravity else torch.zeros_like(self.g))
        print('add_gravity:', add_gravity, self.g)
        print('a_thrust:', a_thrust)
        print('a_drag:', a_drag)
        print('a_total:', a_total)

        # Semi-implicit Euler
        v_next = self.v + a_total * self.dt
        p_next = self.p + v_next * self.dt

        # Attitude from thrust + heading
        attitude = self.attitude_from_thrust_and_heading(a_thrust, heading_xy)

        # Compute delta for convenience
        delta_pos = p_next - self.p

        # Update internal state (keep graph)
        self.v = v_next
        self.p = p_next
        self.att_euler = attitude

        return {
            "p_next": p_next,
            "v_next": v_next,
            "a_thrust": a_thrust,
            "attitude": attitude,      # roll, pitch, yaw
            "delta_pos": delta_pos
        }
