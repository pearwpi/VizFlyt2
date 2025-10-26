# Module Composition Architecture

## Overview

The perception system uses a type-safe composition architecture where modules can be combined using the `+` operator or `.compose()` method. Composition is defined within the module classes themselves, ensuring proper type constraints.

## Module Hierarchy

```
BaseModule (ABC)
├── SplatRenderer - Renders from Gaussian Splat
├── StereoCamera - Renders stereo pairs
└── ComposedBaseModule - BaseModule + VisionModule composition

VisionModule (ABC)
├── SnowModule - Adds snow weather effect
├── FlowModule - Computes optical flow
├── EventCameraModule - Simulates event camera
└── ComposedVisionModule - VisionModule + VisionModule composition
```

## Composition Rules

### ✅ Allowed Compositions

1. **BaseModule + VisionModule → BaseModule**
   ```python
   renderer = api.get_mono_renderer()  # BaseModule
   snow = SnowModule()  # VisionModule
   snowy_renderer = renderer + snow  # Returns BaseModule
   results = snowy_renderer.render(position, orientation)
   ```

2. **VisionModule + VisionModule → VisionModule**
   ```python
   snow = SnowModule()  # VisionModule
   flow = FlowModule()  # VisionModule
   snowy_flow = snow + flow  # Returns VisionModule
   
   # Can then be added to a BaseModule
   renderer = api.get_mono_renderer()
   pipeline = renderer + snowy_flow  # Returns BaseModule
   ```

### ❌ Forbidden Compositions

**BaseModule + BaseModule → ERROR**
```python
mono = api.get_mono_renderer()  # BaseModule
stereo = api.get_stereo_renderer()  # BaseModule
invalid = mono + stereo  # TypeError: Cannot compose BaseModule with another BaseModule
```

**Rationale**: You can't chain two renderers together. Each pipeline must have exactly one renderer at the start.

## Implementation Details

### BaseModule Class

```python
class BaseModule(ABC):
    @abstractmethod
    def render(self, position, orientation_rpy, **kwargs) -> Dict[str, np.ndarray]:
        """Render RGB and depth from Gaussian Splat."""
        pass
    
    def compose(self, other: VisionModule, name: str = None) -> BaseModule:
        """Compose with a VisionModule."""
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose BaseModule with another BaseModule")
        return ComposedBaseModule(self, other, name=name)
    
    def __add__(self, other: VisionModule) -> BaseModule:
        """Syntactic sugar: renderer + vision_module."""
        return self.compose(other)
```

### VisionModule Class

```python
class VisionModule(ABC):
    @abstractmethod
    def render(self, position, orientation_rpy, **kwargs) -> Dict[str, np.ndarray]:
        """Process RGB/depth to generate derived outputs."""
        pass
    
    def compose(self, other: VisionModule, name: str = None) -> VisionModule:
        """Compose with another VisionModule."""
        if isinstance(other, BaseModule):
            raise TypeError("Cannot compose VisionModule with BaseModule")
        return ComposedVisionModule(self, other, name=name)
    
    def __add__(self, other: VisionModule) -> VisionModule:
        """Syntactic sugar: vision + vision."""
        return self.compose(other)
```

## Usage Examples

### Example 1: Simple Composition

```python
from API import PerceptionAPI
from composition_demo import SnowModule

api = PerceptionAPI("config.yml", "camera.json")
mono = api.get_mono_renderer()
snow = SnowModule(intensity=0.4)

# Compose
snowy_mono = mono + snow

# Render
results = snowy_mono.render([0, 0, 0], [0, 0, 0])
# Results: {'rgb': snowy_rgb, 'depth': depth, 'depth_raw': depth_raw, 'snow_mask': mask}
```

### Example 2: Chained Processing

```python
from composition_demo import SnowModule, FlowModule, EventCameraModule

# Chain multiple vision modules
snow = SnowModule(intensity=0.3)
flow = FlowModule()
events = EventCameraModule(threshold=20)

vision_chain = snow + flow + events  # VisionModule

# Add to renderer
mono = api.get_mono_renderer()
full_pipeline = mono + vision_chain  # BaseModule

# Render
results = full_pipeline.render([0, 0, 0], [0, 0, 0])
# Results include: rgb, depth, snow_mask, flow, flow_magnitude, events, polarity
```

### Example 3: Stereo + Events

```python
stereo = api.get_stereo_renderer(baseline=0.065)

class StereoEventModule(VisionModule):
    def render(self, position, orientation_rpy, rgb_left=None, rgb_right=None, **kwargs):
        # Process stereo images to generate stereo events
        ...
        return {'events_left': ..., 'events_right': ...}

events = StereoEventModule()
stereo_events = stereo + events  # BaseModule

results = stereo_events.render([0, 0, 0], [0, 0, 0])
# Results: rgb_left, rgb_right, depth_left, depth_right, events_left, events_right
```

## Architecture Benefits

1. **Type Safety**: Composition rules enforced at the class level prevent invalid module combinations
2. **Clean Syntax**: Use `+` operator or `.compose()` method for intuitive composition
3. **Automatic Data Flow**: Outputs from one module automatically flow to the next
4. **Extensibility**: Easy to create new VisionModules that integrate seamlessly
5. **No External Composition Logic**: All composition handled within module classes

## File Organization

- `modules.py` - BaseModule, VisionModule, and composed module classes
- `splat_render.py` - SplatRenderer (BaseModule)
- `stereo_camera.py` - StereoCamera (BaseModule)
- `API.py` - PerceptionAPI factory for creating renderers
- `composition_demo.py` - Example VisionModules and usage patterns

## Creating Custom Modules

### Custom VisionModule

```python
from modules import VisionModule
import numpy as np

class MyVisionModule(VisionModule):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def render(self, position, orientation_rpy, rgb=None, depth=None, **kwargs):
        """Process inputs and return outputs."""
        if rgb is None:
            return {}
        
        # Your processing here
        result = process(rgb, self.param1, self.param2)
        
        return {
            'my_output': result,
            'rgb': rgb  # Pass through if needed by next module
        }
```

### Custom BaseModule (Advanced)

Only create custom BaseModules if you have a new rendering method (e.g., different splat backend, synthetic data generator).

```python
from modules import BaseModule
import numpy as np

class MyRenderer(BaseModule):
    def render(self, position, orientation_rpy, **kwargs):
        """Render from scene."""
        # Your rendering logic
        rgb = render_rgb(position, orientation_rpy)
        depth = render_depth(position, orientation_rpy)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'depth_raw': depth  # Raw depth values
        }
```

## Migration from Old API

### Old Way (Removed)
```python
# OLD: Composition via API methods
composed = api.compose_modules(snow_mod, flow_mod)
api.register_custom_module('snowy_flow', composed)
results = api.render(position, orientation, modalities=['snowy_flow'])
```

### New Way
```python
# NEW: Composition via module methods
mono = api.get_mono_renderer()
snowy_flow = SnowModule() + FlowModule()
pipeline = mono + snowy_flow
results = pipeline.render(position, orientation)
```
