# Module Composition Guide

## Overview

Composition lets you chain perception modules using the `+` operator. Outputs from one module automatically flow as inputs to the next.

## Module Types

### BaseModule (Rendering)
- **SplatRenderer**: Mono camera
- **StereoCamera**: Stereo pair

### VisionModule (Processing)
- **Custom modules**: Created via decorators or subclassing
- Process RGB/depth to produce derived outputs

## Quick Example

```python
from perception.splat_render import SplatRenderer
from perception.modules import rgb_vision_module_factory
import cv2

renderer = SplatRenderer("config.yml", "cam.json")

@rgb_vision_module_factory
def edge_detector(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return {'edges': edges}

pipeline = renderer + edge_detector
results = pipeline.render(position, orientation_rpy)
```

## Creating Custom Modules

### Method 1: Decorators (Quick)

```python
from perception.modules import rgb_vision_module_factory, vision_module_factory

# RGB-only
@rgb_vision_module_factory
def my_processor(rgb_image):
    result = process(rgb_image)
    return {'output': result}

# Generic (access all inputs)
@vision_module_factory
def my_processor(position, orientation_rpy, **kwargs):
    rgb = kwargs.get('rgb')
    depth = kwargs.get('depth')
    if rgb is None:
        return {}
    result = process(rgb, depth)
    return {'output': result}
```

### Method 2: Subclassing (Full Control)

```python
from perception.modules import VisionModule

class EdgeDetector(VisionModule):
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        if rgb is None:
            return {}
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        return {'edges': edges, 'rgb': rgb}

edge_detector = EdgeDetector(low_threshold=100)
pipeline = renderer + edge_detector
```

## Stereo Composition

Vision modules run **independently** on left and right views:

```python
from perception.stereo_camera import StereoCamera

stereo = StereoCamera("config.yml", "cam.json", baseline=0.065)

@rgb_vision_module_factory
def add_snow(rgb_image):
    mask = np.random.random(rgb_image.shape[:2]) < 0.2
    rgb_snow = rgb_image.copy()
    rgb_snow[mask] = [255, 255, 255]
    return {'rgb': rgb_snow}

pipeline = stereo + add_snow
results = pipeline.render(position, orientation_rpy)
# results['rgb_left'], results['rgb_right'] both have snow
```

## Composition Rules

✅ **Allowed:**
- `BaseModule + VisionModule`
- `VisionModule + VisionModule`
- `renderer + vision1 + vision2`

❌ **Not Allowed:**
- `BaseModule + BaseModule` (can't chain renderers)

## Data Flow

```
renderer          vision1           vision2
   ↓                 ↓                 ↓
{'rgb': ...}  →  {'rgb': ...,   →  {'edges': ...,
                  'noise': ...}      'rgb': ...}
      └──────────merged─────────merged──┘
```

## Best Practices

1. **Pass Through RGB When Needed**
```python
return {'my_output': result, 'rgb': rgb}
```

2. **Check for Required Inputs**
```python
if rgb is None:
    return {}
```

3. **Use Descriptive Output Keys**
```python
return {'edges': edges}  # Good
return {'output': result}  # Too generic
```

## Examples

See [`perception/EXAMPLES.md`](perception/EXAMPLES.md) for runnable examples:
- `basic_examples.py` - Basic usage
- `custom_module_example.py` - Decorators vs subclassing
- `composition_examples.py` - Advanced pipelines
- `stereo_composition_example.py` - Stereo-specific

## See Also
- [README.md](README.md) - Main documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference
- [perception/EXAMPLES.md](perception/EXAMPLES.md) - Example scripts
