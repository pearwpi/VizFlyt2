# VizFlyt2 Quick Reference# VizFlyt2 Quick Reference Guide



## Installation## Module Architecture

```bash

pip install numpy opencv-python torch transforms3d scipy nerfstudio```python

```from modules import BaseModule, VisionModule



## Basic Rendering# BaseModule: Renders RGB/depth from splat (first stage)

# - MonoRenderer, StereoRenderer

### Mono Camera

```python# VisionModule: Processes RGB/depth (second stage)  

from perception.splat_render import SplatRenderer# - EventCamera, OpticalFlow, Noise, etc.

import numpy as np```



renderer = SplatRenderer("config.yml", "cam_settings.json")## Installation

position = np.array([0.0, 0.0, 0.0])  # NED: [x_north, y_east, z_down]```bash

orientation_rpy = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] radianspip install numpy opencv-python torch transforms3d scipy nerfstudio

```

results = renderer.render(position, orientation_rpy)

# results['rgb']   - (H, W, 3) uint8 BGR## Basic Usage

# results['depth'] - (H, W) float32

```### Mono Camera

```python

### Stereo Camerafrom perception.API import create_mono_api

```pythonimport numpy as np

from perception.stereo_camera import StereoCamera

api = create_mono_api("config.yml", "cam_settings.json")

stereo = StereoCamera("config.yml", "cam_settings.json", baseline=0.065)position = np.array([0.0, 0.0, 0.0])

results = stereo.render(position, orientation_rpy)orientation_rpy = np.array([0.0, 0.0, 0.0])

results = api.render(position, orientation_rpy)

# results['rgb_left'], results['rgb_right']# results['rgb'], results['depth']

# results['depth_left'], results['depth_right']```

```

### Stereo Camera

## Module Composition```python

from perception.API import create_stereo_api

### Using + Operator

```pythonapi = create_stereo_api("config.yml", "cam_settings.json", baseline=0.065)

# Compose modulesresults = api.render(position, orientation_rpy)

pipeline = renderer + vision_module1 + vision_module2# results['rgb_left'], results['rgb_right'], results['depth_left'], results['depth_right']

```

# Render through pipeline

results = pipeline.render(position, orientation_rpy)### Batch Rendering

``````python

positions = np.array([[0,0,0], [1,0,0], [2,0,0]])

### Decorator Factories (Quick)orientations = np.array([[0,0,0], [0,0,0], [0,0,0]])

```pythonresults_list = api.render_batch(positions, orientations)

from perception.modules import rgb_vision_module_factory```

import cv2

## Common Operations

@rgb_vision_module_factory

def edge_detector(rgb_image):### Change Stereo Baseline

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)```python

    edges = cv2.Canny(gray, 50, 150)api.set_stereo_baseline(0.10)  # 10cm baseline

    return {'edges': edges}```



# Use it### Switch Modes

pipeline = renderer + edge_detector```python

```api.enable_stereo(baseline=0.065)

api.disable_stereo()

### Subclassing (Full Control)```

```python

from perception.modules import VisionModule### Get Info

```python

class MyModule(VisionModule):api.get_image_dimensions()  # (height, width)

    def __init__(self, threshold=10):api.get_enabled_modalities()  # ['mono'] or ['stereo']

        self.threshold = thresholdapi.get_config_info()  # Full config dictionary

    ```

    def render(self, position, orientation_rpy, rgb=None, **kwargs):

        if rgb is None:## Custom Module

            return {}

        result = process(rgb, self.threshold)### Option 1: Factory Decorators (Quick & Easy)

        return {'output': result}```python

from modules import rgb_vision_module_factory

# Use it

pipeline = renderer + MyModule(threshold=20)@rgb_vision_module_factory

```def my_processor(rgb_image):

    """Automatically extracts rgb from kwargs."""

## Common Patterns    result = process(rgb_image)

    return {'output': result}

### Chain Processing

```python# Use in composition

# noise → blur → edgespipeline = renderer + my_processor

pipeline = renderer + noise_module + blur_module + edge_module```

```

### Option 2: Subclass (Full Control)

### Stereo Composition```python

```pythonfrom modules import VisionModule

# Vision modules run independently on left/right

stereo_pipeline = stereo + snow_module + edge_detectorclass MyVisionModule(VisionModule):

    def __init__(self, param=10):

results = stereo_pipeline.render(position, orientation_rpy)        self.param = param

# results['rgb_left'], results['rgb_right']    

# results['edges_left'], results['edges_right']  # From edge_detector    def render(self, position, orientation_rpy, rgb=None, **kwargs):

```        if rgb is None:

            return {}

### Save Results        result = process(rgb)

```python        return {'output': result}

import cv2

import os# Use in composition

pipeline = renderer + MyVisionModule(param=20)

os.makedirs("outputs", exist_ok=True)```

cv2.imwrite("outputs/frame.png", results['rgb'])

```## Module Composition

```python

## Module Types# Compose snow + flow modules using + operator

from composition_demo import SnowModule, FlowModule

### BaseModule (Renderers)

- `SplatRenderer` - Mono cameraapi = PerceptionAPI("config.yml", "camera.json")

- `StereoCamera` - Stereo pairmono = api.get_mono_renderer()



### VisionModule (Processors)snow = SnowModule(intensity=0.3)

Create with:flow = FlowModule()

1. **`@rgb_vision_module_factory`** - Simple RGB processing

2. **`@vision_module_factory`** - Generic processing (access all kwargs)# Chain vision modules

3. **Subclass `VisionModule`** - Full control with parameters/statesnowy_flow = snow + flow  # VisionModule



## Composition Rules# Add to renderer

pipeline = mono + snowy_flow  # BaseModule

✅ **Allowed:**results = pipeline.render(position, orientation_rpy)

- `BaseModule + VisionModule` → ComposedBase```

- `VisionModule + VisionModule` → ComposedVision  

- `renderer + vision1 + vision2` → Chained pipeline## Hooks

```python

❌ **Not Allowed:**def pre_hook(position, orientation_rpy):

- `BaseModule + BaseModule` → Error (can't chain renderers)    print(f"Rendering {position}")



## Data Flowdef post_hook(results):

    print(f"Rendered {results.keys()}")

```

renderer.render()     vision1.render()      vision2.render()api.add_pre_render_hook(pre_hook)

      ↓                     ↓                      ↓api.add_post_render_hook(post_hook)

{'rgb': ...,         {'rgb': processed,     {'edges': ...,```

 'depth': ...}  →     'noise': ...}    →     'rgb': ...}

      └──────────────────────────────────────────┘## Coordinate System (NED)

                  All merged into final results- Position: [x_north, y_east, z_down] in meters

```- Orientation: [roll, pitch, yaw] in radians

  - Roll: rotation about forward axis

## Coordinate System (NED)  - Pitch: rotation about right axis  

  - Yaw: rotation about down axis

**Position:** `[x, y, z]` in meters

- x: North (forward)## File Formats

- y: East (right)

- z: Down### Trajectory File

```

**Orientation:** `[roll, pitch, yaw]` in radians# x y z roll pitch yaw

- Roll: Right-wing down is positive0.0 0.0 0.0 0.0 0.0 0.0

- Pitch: Nose up is positive1.0 0.0 0.0 0.0 0.0 0.0

- Yaw: Nose right is positive```



## File Formats### Camera Settings JSON

```json

### Camera Settings JSON{

```json  "camera": {

{    "c2w_matrix": [[...], [...], [...], [...]],

  "camera": {    "fov_radians": 1.3089969389957472,

    "c2w_matrix": [[...], [...], [...], [...]],    "render_resolution": 1080

    "fov_radians": 1.3089969389957472,  }

    "render_resolution": 1080}

  }```

}

```## Examples

See `perception/example_usage.py` for basic usage.

### Trajectory FileSee `perception/composition_examples.py` for:

```1. Noise + Flow composition

# x y z roll pitch yaw2. Stereo + Events composition

0.0 0.0 0.0 0.0 0.0 0.03. Multi-stage pipelines

1.0 0.0 0.0 0.0 0.0 0.0

2.0 0.0 0.0 0.0 0.0 0.0Run: `python perception/example_usage.py`

```Run: `python perception/composition_examples.py`


## Example Snippets

### Render Trajectory
```python
import numpy as np

# Load trajectory
trajectory = np.loadtxt("trajectory.txt")
positions = trajectory[:, :3]
orientations = trajectory[:, 3:6]

# Render each frame
for i, (pos, ori) in enumerate(zip(positions, orientations)):
    results = renderer.render(pos, ori)
    cv2.imwrite(f"outputs/frame_{i:04d}.png", results['rgb'])
```

### Stereo Side-by-Side
```python
stereo = StereoCamera("config.yml", "cam.json", baseline=0.065)
results = stereo.render(position, orientation_rpy)

# Concatenate left | right
stereo_pair = np.hstack([results['rgb_left'], results['rgb_right']])
cv2.imwrite("outputs/stereo.png", stereo_pair)
```

### Custom RGB Processing
```python
@rgb_vision_module_factory
def brightness_boost(rgb_image):
    boosted = cv2.convertScaleAbs(rgb_image, alpha=1.3, beta=20)
    return {'rgb': boosted}

pipeline = renderer + brightness_boost
```

### Custom Multi-Input Processing
```python
@vision_module_factory
def depth_overlay(position, orientation_rpy, **kwargs):
    rgb = kwargs.get('rgb')
    depth = kwargs.get('depth')
    
    if rgb is None or depth is None:
        return {}
    
    # Colorize depth
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Blend with RGB
    blended = cv2.addWeighted(rgb, 0.7, depth_color, 0.3, 0)
    return {'blended': blended}

pipeline = renderer + depth_overlay
```

## Troubleshooting

**RGB is None in my vision module:**
- Check previous module returns `'rgb'` in its output dict
- For decorators, ensure `rgb` is in kwargs

**Outputs have `_left`/`_right` suffixes:**
- You're using `StereoCamera` composition
- Vision modules run independently on each view

**TypeError: Cannot compose BaseModule with BaseModule:**
- Only one renderer per pipeline
- Use `renderer + vision_modules`, not `renderer1 + renderer2`

## See Also

- **[README.md](README.md)** - Main documentation
- **[COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md)** - Detailed composition guide
- **[perception/EXAMPLES.md](perception/EXAMPLES.md)** - Example scripts
- **Examples:** `basic_examples.py`, `custom_module_example.py`, `composition_examples.py`
