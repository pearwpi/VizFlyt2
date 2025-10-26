# Module Composition Guide

## Overview

The VizFlyt2 Perception API supports **module composition**, allowing you to chain multiple perception modules together. Outputs from one module automatically flow as inputs to the next module.

## Module Hierarchy

VizFlyt2 uses a two-tier module system defined in `perception/modules.py`:

### BaseModule
Renders RGB and depth directly from the Gaussian Splat scene:
- **MonoRenderer**: Single camera view
- **StereoRenderer**: Stereo camera views

BaseModules are **first-stage** modules that generate fundamental RGB/depth data from the 3D scene.

### VisionModule  
Processes RGB/depth images to produce derived outputs:
- **EventCamera**: DVS event simulation
- **OpticalFlow**: Motion estimation
- **SnowModule**: Weather effect simulation (snow)
- **Custom modules**: Segmentation, detection, etc.

VisionModules are **second-stage** modules that process existing image data rather than rendering from the splat.

## Why Use Composition?

Instead of manually connecting modules and passing data between them, composition lets you:

- ✅ Create complex pipelines declaratively
- ✅ Automatically handle data flow between modules
- ✅ Reuse existing modules in new combinations
- ✅ Avoid namespace conflicts with automatic prefixing

## Basic Concept

```
Module A → Module B → Module C
   ↓          ↓          ↓
output_a → output_b → output_c
   └─────→   (uses output_a as input)
              └─────→ (uses output_a and output_b as inputs)
```

## Quick Examples

### Example 1: Snow + Flow

Add snow effect to images, then compute optical flow on the snowy images:

```python
# Create modules
snow_module = SnowModule(intensity=0.4)
flow_module = FlowModule()

# Compose them
composed = api.compose_modules(
    ('snow', snow_module),
    ('flow', flow_module),
    name='snowy_flow'
)

# Register and use
api.register_custom_module('snowy_flow', composed)
results = api.render(position, orientation_rpy)

# Access outputs (with automatic prefixing)
snowy_rgb = results['snowy_flow_snow_rgb']
flow = results['snowy_flow_flow_flow']
```

### Example 2: Stereo + Events

Render stereo cameras, then simulate event cameras on both views:

```python
# Create event camera module
event_module = EventCameraModule(threshold=15, stereo=True)

# Use convenience method
composed = api.create_stereo_events_module(event_module)

# Register and use
api.register_custom_module('stereo_events', composed)
results = api.render(position, orientation_rpy)

# Access stereo event outputs
events_left = results['stereo_events_events_events_left']
events_right = results['stereo_events_events_events_right']
```

### Example 3: Multi-Stage Pipeline

Create a 3-stage pipeline:

```python
composed = api.compose_modules(
    ('snow', SnowModule()),
    ('events', EventCameraModule()),
    ('flow', FlowModule()),
    name='complex_pipeline'
)
```

This creates: `snow → events → flow`

## How to Write Composable Modules

### VisionModule Template

You can create vision modules in two ways: by **subclassing** or using **factory decorators**.

#### Option 1: Factory Decorators (Simple Functions)

For simple vision processing, use the decorator factories:

```python
from modules import vision_module_factory, rgb_vision_module_factory

# Generic vision module (access all kwargs)
@vision_module_factory
def my_processor(position, orientation_rpy, **kwargs):
    """Process any inputs from previous modules."""
    rgb = kwargs.get('rgb')
    depth = kwargs.get('depth')
    if rgb is None:
        return {}
    
    result = process(rgb, depth)
    return {'output': result}

# RGB-specific vision module (simplified)
@rgb_vision_module_factory
def my_rgb_processor(rgb_image):
    """Process only RGB input (automatically extracted)."""
    # rgb_image is automatically extracted from kwargs['rgb']
    # Returns None if 'rgb' not available
    result = process(rgb_image)
    return {'output': result}

# Use with composition
pipeline = renderer + my_processor
# or
pipeline = renderer + my_rgb_processor
```

**When to use decorators:**
- Simple, stateless processing functions
- Quick prototyping
- One-off image transformations
- When you don't need `__init__` parameters

**When to use subclassing:**
- Need initialization parameters or state
- Complex logic requiring multiple methods
- Reusable modules with configuration

#### Option 2: Subclassing (Full Control)

For modules with state or complex logic, inherit from `VisionModule`:

```python
from modules import VisionModule
import numpy as np

class MyVisionModule(VisionModule):
    """
    Example vision module that processes RGB images.
    """
    
    def __init__(self, param1=10):
        self.param1 = param1
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        """
        Process RGB input to generate output.
        
        Args:
            position: Camera position (always passed)
            orientation_rpy: Camera orientation (always passed)
            rgb: RGB image from previous module (may be None)
            **kwargs: Other outputs from previous modules
        
        Returns:
            Dictionary of outputs to pass to next module
        """
        # Check for required inputs
        if rgb is None:
            return {}
        
        # Process data
        result = self.process(rgb)
        
        # Return outputs
        return {
            'my_output': result,
            'rgb': rgb  # Pass through for next module if needed
        }
```

### BaseModule Template

If you need a custom renderer (rare), inherit from `BaseModule`:

```python
from modules import BaseModule
import numpy as np

class MyCustomRenderer(BaseModule):
    """
    Custom renderer that generates RGB and depth from a source.
    """
    
    def __init__(self, config):
        self.config = config
        # Initialize rendering backend
    
    def render(self, position, orientation_rpy, **kwargs):
        """
        Render RGB and depth from position/orientation.
        
        Returns:
            Dictionary with 'rgb' and 'depth' keys at minimum
        """
        # Render from your backend
        rgb = self.render_rgb(position, orientation_rpy)
        depth = self.render_depth(position, orientation_rpy)
        
        return {'rgb': rgb, 'depth': depth}
```

### Key Guidelines

1. **Inherit from the right base class**:
   - `BaseModule` for renderers that create RGB/depth from scratch
   - `VisionModule` for processors that use RGB/depth as input

2. **Accept `**kwargs`**: This receives outputs from previous modules
3. **Return dictionary**: All outputs should be in a dict with descriptive keys
4. **Handle None gracefully**: Input data might not always be available
5. **Pass through relevant data**: If next modules need certain inputs, include them in your output

### Example: Snow Module (VisionModule)

```python
from modules import VisionModule
import numpy as np

class SnowModule(VisionModule):
    """Adds snow effect to RGB images."""
    
    def __init__(self, intensity=0.3, flake_size=2):
        """
        Args:
            intensity: Snow intensity (0.0 to 1.0), controls density
            flake_size: Size of snowflakes in pixels
        """
        self.intensity = intensity
        self.flake_size = flake_size
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        if rgb is None:
            return {}
        
        h, w = rgb.shape[:2]
        snowy_rgb = rgb.copy().astype(float)
        
        # Create snow mask (random snowflakes)
        num_flakes = int(h * w * self.intensity * 0.001)
        snow_mask = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_flakes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            brightness = np.random.uniform(0.7, 1.0)
            
            # Draw snowflake
            y_min = max(0, y - self.flake_size // 2)
            y_max = min(h, y + self.flake_size // 2 + 1)
            x_min = max(0, x - self.flake_size // 2)
            x_max = min(w, x + self.flake_size // 2 + 1)
            snow_mask[y_min:y_max, x_min:x_max] = brightness
        
        # Apply snow (white overlay)
        for c in range(3):
            snowy_rgb[:, :, c] = snowy_rgb[:, :, c] * (1 - snow_mask * 0.8) + snow_mask * 255 * 0.8
        
        snowy_rgb = np.clip(snowy_rgb, 0, 255).astype(np.uint8)
        
        # Return snowy RGB for next module + snow mask
        return {
            'rgb': snowy_rgb,      # Override rgb for next module
            'snow_mask': snow_mask  # Additional output
        }
```

### Example: Event Camera Module (VisionModule)

```python
from modules import VisionModule
import cv2
import numpy as np

class EventCameraModule(VisionModule):
    """Simulates event camera from RGB intensity changes."""
    
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.prev_frame = None
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        if rgb is None:
            return {}
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        if self.prev_frame is not None:
            # Compute events
            diff = gray.astype(float) - self.prev_frame.astype(float)
            events = (np.abs(diff) > self.threshold).astype(np.uint8)
            
            self.prev_frame = gray
            return {'events': events, 'polarity': np.sign(diff)}
        
        self.prev_frame = gray
        return {}
```

## Data Flow Example

Here's how data flows through a composed pipeline:

```python
# Create pipeline: Base Renderer → Snow → Flow
api = create_mono_api(config, json_path)
composed = api.compose_modules(
    ('snow', SnowModule()),
    ('flow', FlowModule())
)
api.register_custom_module('snowy_flow', composed)

# Render
results = api.render(position, orientation_rpy)

# Data flow:
# 1. Base renderer outputs: {'rgb': <image>, 'depth': <depth>}
# 2. Snow module receives rgb, outputs: {'rgb': <snowy>, 'snow_mask': <mask>}
# 3. Flow module receives rgb (snowy), outputs: {'flow': <flow>}
# 4. Final results with prefixing:
#    - Base: {'rgb': <original>, 'depth': <depth>}
#    - Composed: {'snowy_flow_snow_rgb': <snowy>,
#                 'snowy_flow_snow_snow_mask': <mask>,
#                 'snowy_flow_flow_flow': <flow>}
```

## Output Key Naming

Composed module outputs are prefixed to avoid conflicts:

```
{composition_name}_{module_name}_{output_key}
```

Examples:
- `snowy_flow_snow_rgb` = composition "snowy_flow" → module "snow" → output "rgb"
- `stereo_events_events_events_left` = composition "stereo_events" → module "events" → output "events_left"

## Convenience Methods

The API provides convenience methods for common compositions:

```python
# Stereo + Events
composed = api.create_stereo_events_module(event_module)

# Snow + Flow
composed = api.create_snow_flow_module(snow_module, flow_module)
```

## Complete Example

See `perception/composition_demo.py` for complete working examples including:

1. **Snow + Flow**: Optical flow on snowy images
2. **Stereo + Events**: Stereo event cameras
3. **Multi-stage**: Three-module pipeline

Run:
```bash
cd perception
python composition_demo.py
```

## Tips

- Start with simple 2-module compositions
- Test each module individually before composing
- Use descriptive names for your modules
- Check output keys using `results.keys()` to debug data flow
- Modules can be reused in different compositions
