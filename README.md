# VizFlyt2

A flexible perception system for rendering photorealistic synthetic sensor data using Gaussian Splatting. VizFlyt2 provides mono and stereo camera rendering with support for custom perception modalities.

## 🌟 Features

- **High-Fidelity Rendering**: Photorealistic RGB and depth rendering using Gaussian Splatting (via Nerfstudio)
- **Stereo Camera Support**: Configurable stereo baseline for binocular vision
- **Flexible API**: Easy-to-use Python API with extensible architecture
- **Multiple Modalities**: Support for mono, stereo, and custom perception modules
- **Batch Processing**: Efficient rendering of entire trajectories
- **NED Frame Support**: Uses North-East-Down coordinate system for aerospace applications

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Module Composition](#module-composition)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Coordinate Systems](#coordinate-systems)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Nerfstudio

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pearwpi/VizFlyt2.git
cd VizFlyt2
```

2. Install dependencies:
```bash
pip install numpy opencv-python torch transforms3d scipy
# Install nerfstudio following their official guide
pip install nerfstudio
```

3. Prepare your Gaussian Splatting model:
   - Train a splatfacto model using Nerfstudio
   - Note the path to your `config.yml` file
   - Prepare camera settings JSON file

## 🎯 Quick Start

### Mono Camera Rendering

```python
from perception.API import create_mono_api
import numpy as np

# Initialize the API
api = create_mono_api(
    config_path="path/to/config.yml",
    json_path="cam_settings.json"
)

# Define pose (NED coordinates)
position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
orientation_rpy = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw in radians

# Render
results = api.render(position, orientation_rpy)

# Access outputs
rgb_image = results['rgb']      # (H, W, 3) uint8
depth_map = results['depth']    # (H, W) float32
```

### Stereo Camera Rendering

```python
from perception.API import create_stereo_api
import numpy as np

# Initialize stereo API with 6.5cm baseline
api = create_stereo_api(
    config_path="path/to/config.yml",
    json_path="cam_settings.json",
    baseline=0.065  # meters
)

# Render stereo pair
results = api.render(position, orientation_rpy)

# Access outputs
rgb_left = results['rgb_left']       # (H, W, 3) uint8
rgb_right = results['rgb_right']     # (H, W, 3) uint8
depth_left = results['depth_left']   # (H, W) float32
depth_right = results['depth_right'] # (H, W) float32
```

### Rendering from Trajectory

```python
# Load trajectory
positions = np.loadtxt('trajectory.txt')[:, :3]      # (N, 3)
orientations = np.loadtxt('trajectory.txt')[:, 3:6]  # (N, 3)

# Batch render
results_list = api.render_batch(positions, orientations)

# Process each frame
for idx, results in enumerate(results_list):
    # Save or process results
    cv2.imwrite(f"frame_{idx:06d}.png", results['rgb'])
```

## 📚 API Documentation

### PerceptionAPI Class

Main class for interfacing with the perception system.

#### Initialization

```python
from perception.API import PerceptionAPI

api = PerceptionAPI(
    config_path: str,           # Path to nerfstudio config
    json_path: str,             # Path to camera settings JSON
    aspect_ratio: float = 16/9, # Image aspect ratio
    enable_stereo: bool = False,# Enable stereo mode
    stereo_baseline: float = 0.05  # Baseline in meters
)
```

#### Core Methods

**`render(position, orientation_rpy, modalities=None)`**
- Render from a single pose
- Returns: `Dict[str, np.ndarray]` with RGB and depth data

**`render_batch(positions, orientations_rpy, modalities=None)`**
- Render multiple poses
- Returns: `List[Dict[str, np.ndarray]]`

**`enable_stereo(baseline=None)`**
- Switch to stereo rendering mode
- Optionally update baseline

**`disable_stereo()`**
- Switch to mono rendering mode

**`set_stereo_baseline(baseline)`**
- Update stereo baseline distance

**`get_stereo_baseline()`**
- Get current baseline distance

**`get_image_dimensions()`**
- Returns: `(height, width)` tuple

**`get_enabled_modalities()`**
- Returns: List of enabled modality names

**`get_config_info()`**
- Returns: Dict with configuration details

#### Module Composition Methods

**`compose_modules(*modules, name=None)`**
- Compose multiple modules into a pipeline
- Modules can be objects or (name, module) tuples
- Returns: `ComposedModule` instance

**`create_stereo_events_module(event_module)`**
- Convenience method for stereo + events composition
- Returns: `ComposedModule` with stereo and event camera

**`create_noise_flow_module(noise_module, flow_module)`**
- Convenience method for noise + flow composition
- Returns: `ComposedModule` that computes flow on noisy images

#### Extensibility Methods

**`register_custom_module(name, module)`**
- Add custom perception module
- Module must have a `render(position, orientation_rpy)` method

**`unregister_custom_module(name)`**
- Remove custom module

**`add_pre_render_hook(hook)`**
- Add callback before rendering

**`add_post_render_hook(hook)`**
- Add callback after rendering

**`clear_hooks()`**
- Remove all hooks

### StereoCamera Class

Stereo camera system using SplatRenderer.

```python
from perception.stereo_camera import StereoCamera

stereo_cam = StereoCamera(
    config_path: str,
    json_path: str,
    baseline: float = 0.05,
    aspect_ratio: float = 16/9
)

# Render stereo views
results = stereo_cam.render_stereo(position, orientation_rpy)

# Render mono view
results = stereo_cam.render_mono(position, orientation_rpy)
```

### SplatRenderer Class

Low-level Gaussian Splatting renderer.

```python
from perception.splat_render import SplatRenderer

renderer = SplatRenderer(
    config_path: str,
    json_path: str,
    aspect_ratio: float = 16/9
)

# Render single view
rgb, depth_viz, depth_raw = renderer.render(position, orientation_rpy)
```

## 💡 Usage Example

See `perception/example_usage.py` for a simple example that renders a few images around the origin:

```bash
cd perception
python example_usage.py
```

This will:
- Render 5 mono images moving forward from the origin
- Render 5 stereo pairs moving forward from the origin
- Save outputs to `renders/mono/` and `renders/stereo/`

## 📁 File Structure

```
VizFlyt2/
├── README.md
├── QUICK_REFERENCE.md
├── COMPOSITION_GUIDE.md
└── perception/
    ├── modules.py               # Base classes (BaseModule, VisionModule)
    ├── API.py                   # Main perception API with composition
    ├── stereo_camera.py         # Stereo camera class
    ├── splat_render.py          # Gaussian splatting renderer
    ├── utils.py                 # Utility functions
    ├── example_usage.py         # Simple usage examples
    ├── composition_examples.py  # Module composition examples
    ├── api_example.py           # API feature demonstrations
    └── render_images.py         # Legacy rendering script
```

## 🏗️ Module Architecture

VizFlyt2 uses a hierarchical module system:

### BaseModule
Renders RGB and depth directly from the Gaussian Splat scene. These are first-stage modules:
- **MonoRenderer**: Single camera RGB + depth
- **StereoRenderer**: Stereo camera RGB + depth

### VisionModule  
Processes RGB/depth to produce derived outputs. These are second-stage modules:
- **EventCamera**: Simulates DVS events from intensity changes
- **OpticalFlow**: Computes motion between frames
- **SnowModule**: Adds weather effects (snow simulation)
- **Custom modules**: Create via subclassing or decorator factories

**Creating Custom VisionModules:**
```python
# Quick way: Use decorator factories
from modules import rgb_vision_module_factory

@rgb_vision_module_factory
def edge_detector(rgb_image):
    edges = cv2.Canny(rgb_image, 100, 200)
    return {'edges': edges}

pipeline = renderer + edge_detector

# Full control: Subclass for stateful modules
class MyVisionModule(VisionModule):
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def render(self, position, orientation_rpy, rgb=None, **kwargs):
        # Custom processing with state
        return {'output': result}
```

Modules can be composed together using the `+` operator or `.compose()` method to create complex pipelines.

## ⚙️ Configuration

### Camera Settings JSON Format

```json
{
  "camera": {
    "c2w_matrix": [[...], [...], [...], [...]],
    "fov_radians": 1.3089969389957472,
    "render_resolution": 1080
  }
}
```

### Trajectory File Format

Text file with one pose per line:
```
x y z roll pitch yaw
0.0 0.0 0.0 0.0 0.0 0.0
1.0 0.0 0.0 0.0 0.0 0.0
...
```
- Position: meters (NED frame)
- Orientation: radians (roll-pitch-yaw)

## 🧭 Coordinate Systems

### NED Frame (North-East-Down)
- **X**: North (forward)
- **Y**: East (right)
- **Z**: Down

### Body Frame
- **X**: Forward
- **Y**: Right
- **Z**: Down

### Orientation
- **Roll**: Rotation about X-axis (right-wing down is positive)
- **Pitch**: Rotation about Y-axis (nose up is positive)
- **Yaw**: Rotation about Z-axis (nose right is positive)

## 🔧 Advanced Usage

### Module Composition

Compose multiple modules to create complex perception pipelines where outputs flow through each module:

```python
# Example 1: Snow + Flow (compute optical flow on snowy images)
from composition_demo import SnowModule, FlowModule

api = PerceptionAPI("config.yml", "camera.json")
mono = api.get_mono_renderer()

# Create vision modules
snow = SnowModule(intensity=0.4)
flow = FlowModule()

# Compose: snow + flow (VisionModule)
snowy_flow = snow + flow

# Add to renderer
pipeline = mono + snowy_flow

# Render
results = pipeline.render(position, orientation_rpy)
# Access: results['rgb'], results['depth'], results['snow_mask'], results['flow']
```

```python
# Example 2: Stereo + Events (stereo event cameras)
event_module = EventCameraModule(threshold=15, stereo=True)
composed = api.create_stereo_events_module(event_module)

api.register_custom_module('stereo_events', composed)
results = api.render(position, orientation_rpy)

# Access: results['stereo_events_events_events_left'], results['stereo_events_events_events_right']
```

The composition system automatically:
- Passes outputs from one module as inputs to the next
- Prefixes output keys to avoid conflicts
- Handles position and orientation for all modules

**📖 See [COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md) for detailed documentation and examples.**

See `perception/composition_examples.py` for complete examples.

### Custom Perception Module

```python
class MyCustomModule:
    def __init__(self, model_path):
        # Initialize your module
        pass
    
    def render(self, position, orientation_rpy):
        # Your rendering logic
        return {
            'output_key': np.array(...),
            'another_output': np.array(...)
        }

# Register module
api.register_custom_module("my_module", MyCustomModule("model.pth"))

# Render with custom module
results = api.render(position, orientation_rpy)
custom_output = results['my_module_output_key']
```

### Using Hooks

```python
# Pre-render hook for logging
def log_pose(position, orientation_rpy):
    print(f"Rendering at {position}")

# Post-render hook for analysis
def analyze_output(results):
    depth = results['depth']
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")

api.add_pre_render_hook(log_pose)
api.add_post_render_hook(analyze_output)
```

### Selective Modality Rendering

```python
# Only render specific modalities to save computation
results = api.render(position, orientation_rpy, modalities=['rgb'])
```

## 📊 Performance

Typical rendering times (RTX 3090):
- Mono rendering: ~20-40ms per frame
- Stereo rendering: ~40-80ms per frame

Tips for optimization:
- Use batch rendering for trajectories
- Reduce resolution for faster rendering
- Use selective modality rendering when possible

## 🐛 Troubleshooting

**Issue: "Config file not found"**
- Ensure the config path points to your trained splatfacto model
- Check that the path is correct relative to your working directory

**Issue: Slow rendering**
- Verify GPU is being used (check CUDA availability)
- Try reducing image resolution
- Check that the model is in eval mode

**Issue: Incorrect camera orientation**
- Verify NED coordinate system is being used correctly
- Check that roll-pitch-yaw convention matches (aerospace convention)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Built on [Nerfstudio](https://docs.nerf.studio/)
- Uses Gaussian Splatting for high-quality rendering
- Developed by WPI PEAR Lab

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.