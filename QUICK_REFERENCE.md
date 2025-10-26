# VizFlyt2 Quick Reference Guide

## Module Architecture

```python
from modules import BaseModule, VisionModule

# BaseModule: Renders RGB/depth from splat (first stage)
# - MonoRenderer, StereoRenderer

# VisionModule: Processes RGB/depth (second stage)  
# - EventCamera, OpticalFlow, Noise, etc.
```

## Installation
```bash
pip install numpy opencv-python torch transforms3d scipy nerfstudio
```

## Basic Usage

### Mono Camera
```python
from perception.API import create_mono_api
import numpy as np

api = create_mono_api("config.yml", "cam_settings.json")
position = np.array([0.0, 0.0, 0.0])
orientation_rpy = np.array([0.0, 0.0, 0.0])
results = api.render(position, orientation_rpy)
# results['rgb'], results['depth']
```

### Stereo Camera
```python
from perception.API import create_stereo_api

api = create_stereo_api("config.yml", "cam_settings.json", baseline=0.065)
results = api.render(position, orientation_rpy)
# results['rgb_left'], results['rgb_right'], results['depth_left'], results['depth_right']
```

### Batch Rendering
```python
positions = np.array([[0,0,0], [1,0,0], [2,0,0]])
orientations = np.array([[0,0,0], [0,0,0], [0,0,0]])
results_list = api.render_batch(positions, orientations)
```

## Common Operations

### Change Stereo Baseline
```python
api.set_stereo_baseline(0.10)  # 10cm baseline
```

### Switch Modes
```python
api.enable_stereo(baseline=0.065)
api.disable_stereo()
```

### Get Info
```python
api.get_image_dimensions()  # (height, width)
api.get_enabled_modalities()  # ['mono'] or ['stereo']
api.get_config_info()  # Full config dictionary
```

## Custom Module

### Option 1: Factory Decorators (Quick & Easy)
```python
from modules import rgb_vision_module_factory

@rgb_vision_module_factory
def my_processor(rgb_image):
    """Automatically extracts rgb from kwargs."""
    result = process(rgb_image)
    return {'output': result}

# Use in composition
pipeline = renderer + my_processor
```

### Option 2: Subclass (Full Control)
```python
from modules import VisionModule

class MyVisionModule(VisionModule):
    def __init__(self, param=10):
        self.param = param
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        if rgb is None:
            return {}
        result = process(rgb)
        return {'output': result}

# Use in composition
pipeline = renderer + MyVisionModule(param=20)
```

## Module Composition
```python
# Compose snow + flow modules using + operator
from composition_demo import SnowModule, FlowModule

api = PerceptionAPI("config.yml", "camera.json")
mono = api.get_mono_renderer()

snow = SnowModule(intensity=0.3)
flow = FlowModule()

# Chain vision modules
snowy_flow = snow + flow  # VisionModule

# Add to renderer
pipeline = mono + snowy_flow  # BaseModule
results = pipeline.render(position, orientation_rpy)
```

## Hooks
```python
def pre_hook(position, orientation_rpy):
    print(f"Rendering {position}")

def post_hook(results):
    print(f"Rendered {results.keys()}")

api.add_pre_render_hook(pre_hook)
api.add_post_render_hook(post_hook)
```

## Coordinate System (NED)
- Position: [x_north, y_east, z_down] in meters
- Orientation: [roll, pitch, yaw] in radians
  - Roll: rotation about forward axis
  - Pitch: rotation about right axis  
  - Yaw: rotation about down axis

## File Formats

### Trajectory File
```
# x y z roll pitch yaw
0.0 0.0 0.0 0.0 0.0 0.0
1.0 0.0 0.0 0.0 0.0 0.0
```

### Camera Settings JSON
```json
{
  "camera": {
    "c2w_matrix": [[...], [...], [...], [...]],
    "fov_radians": 1.3089969389957472,
    "render_resolution": 1080
  }
}
```

## Examples
See `perception/example_usage.py` for basic usage.
See `perception/composition_examples.py` for:
1. Noise + Flow composition
2. Stereo + Events composition
3. Multi-stage pipelines

Run: `python perception/example_usage.py`
Run: `python perception/composition_examples.py`
