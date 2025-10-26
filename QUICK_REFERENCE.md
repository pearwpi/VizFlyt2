# VizFlyt2 Quick Reference

## Installation
```bash
pip install numpy opencv-python torch transforms3d scipy nerfstudio
```

## Basic Rendering

### Mono Camera
```python
from perception.splat_render import SplatRenderer
import numpy as np

renderer = SplatRenderer("config.yml", "cam_settings.json")
position = np.array([0.0, 0.0, 0.0])
orientation_rpy = np.array([0.0, 0.0, 0.0])

results = renderer.render(position, orientation_rpy)
# results['rgb']   - (H, W, 3) uint8 BGR
# results['depth'] - (H, W) float32
```

### Stereo Camera
```python
from perception.stereo_camera import StereoCamera

stereo = StereoCamera("config.yml", "cam.json", baseline=0.065)
results = stereo.render(position, orientation_rpy)
# results['rgb_left'], results['rgb_right']
# results['depth_left'], results['depth_right']
```

## Module Composition

### Using + Operator
```python
pipeline = renderer + vision_module1 + vision_module2
results = pipeline.render(position, orientation_rpy)
```

### Decorator Factory
```python
from perception.modules import rgb_vision_module_factory
import cv2

@rgb_vision_module_factory
def edge_detector(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return {'edges': edges}

pipeline = renderer + edge_detector
```

### Subclassing
```python
from perception.modules import VisionModule

class MyModule(VisionModule):
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        if rgb is None:
            return {}
        result = process(rgb, self.threshold)
        return {'output': result}

pipeline = renderer + MyModule(threshold=20)
```

## Common Patterns

### Chain Processing
```python
pipeline = renderer + noise + blur + edges
```

### Stereo Composition
```python
stereo_pipeline = stereo + snow + edges
results = stereo_pipeline.render(position, orientation_rpy)
# results['edges_left'], results['edges_right']
```

## Coordinate System (NED)
- Position: `[x_north, y_east, z_down]` meters
- Orientation: `[roll, pitch, yaw]` radians

## See Also
- **[README.md](README.md)** - Main documentation
- **[COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md)** - Detailed guide
- **[perception/EXAMPLES.md](perception/EXAMPLES.md)** - Example scripts
