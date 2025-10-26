# Module Composition Guide# Module Composition Guide



## Overview## Overview



VizFlyt2 supports **module composition** - chain perception modules together using the `+` operator. Outputs from one module automatically flow as inputs to the next.The VizFlyt2 Perception API supports **module composition**, allowing you to chain multiple perception modules together. Outputs from one module automatically flow as inputs to the next module.



## Module Types## Module Hierarchy



### BaseModule (First-Stage: Rendering)VizFlyt2 uses a two-tier module system defined in `perception/modules.py`:

Renders RGB and depth from the Gaussian Splat scene:

- **SplatRenderer**: Single mono camera### BaseModule

- **StereoCamera**: Stereo camera pairRenders RGB and depth directly from the Gaussian Splat scene:

- **MonoRenderer**: Single camera view

### VisionModule (Second-Stage: Processing)- **StereoRenderer**: Stereo camera views

Processes RGB/depth to produce derived outputs:

- **EventCamera**: DVS event simulationBaseModules are **first-stage** modules that generate fundamental RGB/depth data from the 3D scene.

- **OpticalFlow**: Motion estimation

- **SnowModule**: Weather effects### VisionModule  

- **Custom modules**: Your own processing (decorators or subclassing)Processes RGB/depth images to produce derived outputs:

- **EventCamera**: DVS event simulation

## Quick Start- **OpticalFlow**: Motion estimation

- **SnowModule**: Weather effect simulation (snow)

### Basic Composition- **Custom modules**: Segmentation, detection, etc.



```pythonVisionModules are **second-stage** modules that process existing image data rather than rendering from the splat.

from perception.splat_render import SplatRenderer

from perception.modules import rgb_vision_module_factory## Why Use Composition?

import cv2

Instead of manually connecting modules and passing data between them, composition lets you:

# Create renderer

renderer = SplatRenderer("config.yml", "cam_settings.json")- ✅ Create complex pipelines declaratively

- ✅ Automatically handle data flow between modules

# Create vision module with decorator- ✅ Reuse existing modules in new combinations

@rgb_vision_module_factory- ✅ Avoid namespace conflicts with automatic prefixing

def edge_detector(rgb_image):

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)## Basic Concept

    edges = cv2.Canny(gray, 50, 150)

    return {'edges': edges}```

Module A → Module B → Module C

# Compose with + operator   ↓          ↓          ↓

pipeline = renderer + edge_detectoroutput_a → output_b → output_c

   └─────→   (uses output_a as input)

# Render              └─────→ (uses output_a and output_b as inputs)

results = pipeline.render(position, orientation_rpy)```

# Access: results['rgb'], results['depth'], results['edges']

```## Quick Examples



### Chaining Multiple Modules### Example 1: Snow + Flow



```pythonAdd snow effect to images, then compute optical flow on the snowy images:

# Create vision modules

@rgb_vision_module_factory```python

def add_noise(rgb_image):# Create modules

    noise = np.random.normal(0, 15, rgb_image.shape)snow_module = SnowModule(intensity=0.4)

    noisy = np.clip(rgb_image + noise, 0, 255).astype(np.uint8)flow_module = FlowModule()

    return {'rgb': noisy}

# Compose them

@rgb_vision_module_factorycomposed = api.compose_modules(

def blur(rgb_image):    ('snow', snow_module),

    blurred = cv2.GaussianBlur(rgb_image, (11, 11), 2.0)    ('flow', flow_module),

    return {'rgb': blurred}    name='snowy_flow'

)

@rgb_vision_module_factory

def detect_edges(rgb_image):# Register and use

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)api.register_custom_module('snowy_flow', composed)

    edges = cv2.Canny(gray, 50, 150)results = api.render(position, orientation_rpy)

    return {'edges': edges, 'rgb': rgb_image}

# Access outputs (with automatic prefixing)

# Chain them: noise → blur → edgessnowy_rgb = results['snowy_flow_snow_rgb']

pipeline = renderer + add_noise + blur + detect_edgesflow = results['snowy_flow_flow_flow']

```

# Results flow through automatically

results = pipeline.render(position, orientation_rpy)### Example 2: Stereo + Events

```

Render stereo cameras, then simulate event cameras on both views:

## Composition Rules

```python

### ✅ Allowed# Create event camera module

event_module = EventCameraModule(threshold=15, stereo=True)

1. **BaseModule + VisionModule → ComposedBase**

   ```python# Use convenience method

   renderer + vision_module  # Returns ComposedBase (acts as BaseModule)composed = api.create_stereo_events_module(event_module)

   ```

# Register and use

2. **VisionModule + VisionModule → ComposedVision**api.register_custom_module('stereo_events', composed)

   ```pythonresults = api.render(position, orientation_rpy)

   vision1 + vision2  # Returns ComposedVision (acts as VisionModule)

   ```# Access stereo event outputs

events_left = results['stereo_events_events_events_left']

3. **BaseModule + (VisionModule + VisionModule)**events_right = results['stereo_events_events_events_right']

   ```python```

   renderer + (vision1 + vision2)

   # or equivalently:### Example 3: Multi-Stage Pipeline

   renderer + vision1 + vision2

   ```Create a 3-stage pipeline:



### ❌ Not Allowed```python

composed = api.compose_modules(

**BaseModule + BaseModule**    ('snow', SnowModule()),

```python    ('events', EventCameraModule()),

renderer1 + renderer2  # ERROR: Can't chain two renderers    ('flow', FlowModule()),

```    name='complex_pipeline'

)

*Rationale: Each pipeline needs exactly one renderer at the start.*```



## Data FlowThis creates: `snow → events → flow`



```## How to Write Composable Modules

BaseModule                VisionModule 1           VisionModule 2

(SplatRenderer)           (add noise)              (detect edges)### VisionModule Template

    ↓                          ↓                         ↓

render RGB/depth  →   receives rgb/depth  →   receives rgb/depth/noiseYou can create vision modules in two ways: by **subclassing** or using **factory decorators**.

    ↓                          ↓                         ↓

returns:              returns:                  returns:#### Option 1: Factory Decorators (Simple Functions)

{'rgb': ...,          {'rgb': noisy,            {'edges': ...,

 'depth': ...}         'noise': ...}             'rgb': ...}For simple vision processing, use the decorator factories:

    └──────────────→ merged into kwargs ──────→ merged into kwargs

                                                      ↓```python

                                            Final combined resultsfrom modules import vision_module_factory, rgb_vision_module_factory

```

# Generic vision module (access all kwargs)

Each module:@vision_module_factory

1. Receives `position`, `orientation_rpy`, and `**kwargs` (outputs from previous modules)def my_processor(position, orientation_rpy, **kwargs):

2. Processes the data    """Process any inputs from previous modules."""

3. Returns a dictionary that gets merged with previous outputs    rgb = kwargs.get('rgb')

    depth = kwargs.get('depth')

## Creating Custom Modules    if rgb is None:

        return {}

### Method 1: Decorator Factories (Quick & Easy)    

    result = process(rgb, depth)

#### RGB-Only Processing    return {'output': result}



```python# RGB-specific vision module (simplified)

from perception.modules import rgb_vision_module_factory@rgb_vision_module_factory

def my_rgb_processor(rgb_image):

@rgb_vision_module_factory    """Process only RGB input (automatically extracted)."""

def my_processor(rgb_image):    # rgb_image is automatically extracted from kwargs['rgb']

    """    # Returns None if 'rgb' not available

    Automatically receives rgb from kwargs.    result = process(rgb_image)

    Returns None if rgb not available.    return {'output': result}

    """

    # Your processing here# Use with composition

    result = process(rgb_image)pipeline = renderer + my_processor

    return {'output': result}# or

```pipeline = renderer + my_rgb_processor

```

#### Generic Processing (Access All Inputs)

**When to use decorators:**

```python- Simple, stateless processing functions

from perception.modules import vision_module_factory- Quick prototyping

- One-off image transformations

@vision_module_factory- When you don't need `__init__` parameters

def my_processor(position, orientation_rpy, **kwargs):

    """**When to use subclassing:**

    Access any inputs from previous modules.- Need initialization parameters or state

    """- Complex logic requiring multiple methods

    rgb = kwargs.get('rgb')- Reusable modules with configuration

    depth = kwargs.get('depth')

    #### Option 2: Subclassing (Full Control)

    if rgb is None:

        return {}For modules with state or complex logic, inherit from `VisionModule`:

    

    # Your processing here```python

    result = process(rgb, depth, position)from modules import VisionModule

    return {'output': result}import numpy as np

```

class MyVisionModule(VisionModule):

**When to use decorators:**    """

- Simple, stateless functions    Example vision module that processes RGB images.

- Quick prototyping    """

- One-off transformations    

- Don't need `__init__` parameters    def __init__(self, param1=10):

        self.param1 = param1

### Method 2: Subclassing (Full Control)    

    def render(self, position, orientation_rpy, rgb=None, **kwargs):

```python        """

from perception.modules import VisionModule        Process RGB input to generate output.

import cv2        

        Args:

class EdgeDetector(VisionModule):            position: Camera position (always passed)

    """Edge detection with configurable thresholds."""            orientation_rpy: Camera orientation (always passed)

                rgb: RGB image from previous module (may be None)

    def __init__(self, low_threshold=50, high_threshold=150):            **kwargs: Other outputs from previous modules

        self.low_threshold = low_threshold        

        self.high_threshold = high_threshold        Returns:

                Dictionary of outputs to pass to next module

    def render(self, position, orientation_rpy, rgb=None, **kwargs):        """

        """        # Check for required inputs

        Args:        if rgb is None:

            position: Camera position (always provided)            return {}

            orientation_rpy: Camera orientation (always provided)        

            rgb: RGB image from previous module (may be None)        # Process data

            **kwargs: Other outputs from previous modules        result = self.process(rgb)

                

        Returns:        # Return outputs

            Dictionary of outputs to pass to next module        return {

        """            'my_output': result,

        if rgb is None:            'rgb': rgb  # Pass through for next module if needed

            return {}        }

        ```

        # Convert to grayscale

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)### BaseModule Template

        

        # Detect edgesIf you need a custom renderer (rare), inherit from `BaseModule`:

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        ```python

        return {from modules import BaseModule

            'edges': edges,import numpy as np

            'rgb': rgb  # Pass through if needed by next module

        }class MyCustomRenderer(BaseModule):

    """

# Usage    Custom renderer that generates RGB and depth from a source.

edge_detector = EdgeDetector(low_threshold=100, high_threshold=200)    """

pipeline = renderer + edge_detector    

```    def __init__(self, config):

        self.config = config

**When to use subclassing:**        # Initialize rendering backend

- Need initialization parameters    

- Stateful processing (e.g., temporal filtering, previous frame tracking)    def render(self, position, orientation_rpy, **kwargs):

- Complex logic requiring helper methods        """

- Reusable, configurable modules        Render RGB and depth from position/orientation.

        

## Stereo Composition        Returns:

            Dictionary with 'rgb' and 'depth' keys at minimum

When composing with `StereoCamera`, vision modules run **independently** on left and right views:        """

        # Render from your backend

```python        rgb = self.render_rgb(position, orientation_rpy)

from perception.stereo_camera import StereoCamera        depth = self.render_depth(position, orientation_rpy)

from perception.modules import rgb_vision_module_factory        

        return {'rgb': rgb, 'depth': depth}

stereo = StereoCamera("config.yml", "cam.json", baseline=0.065)```



@rgb_vision_module_factory### Key Guidelines

def add_snow(rgb_image):

    # Add snow effect1. **Inherit from the right base class**:

    mask = np.random.random(rgb_image.shape[:2]) < 0.2   - `BaseModule` for renderers that create RGB/depth from scratch

    rgb_snow = rgb_image.copy()   - `VisionModule` for processors that use RGB/depth as input

    rgb_snow[mask] = [255, 255, 255]

    return {'rgb': rgb_snow}2. **Accept `**kwargs`**: This receives outputs from previous modules

3. **Return dictionary**: All outputs should be in a dict with descriptive keys

# Compose: vision module runs on BOTH left and right independently4. **Handle None gracefully**: Input data might not always be available

pipeline = stereo + add_snow5. **Pass through relevant data**: If next modules need certain inputs, include them in your output



results = pipeline.render(position, orientation_rpy)### Example: Snow Module (VisionModule)



# Outputs have _left and _right suffixes```python

# results['rgb_left']   - left view with snowfrom modules import VisionModule

# results['rgb_right']  - right view with snowimport numpy as np

# results['depth_left']

# results['depth_right']class SnowModule(VisionModule):

```    """Adds snow effect to RGB images."""

    

**How it works internally:**    def __init__(self, intensity=0.3, flake_size=2):

1. StereoCamera renders both views → `rgb_left`, `rgb_right`, `depth_left`, `depth_right`        """

2. Vision module runs on left: `{'rgb': rgb_left, 'depth': depth_left}` → outputs        Args:

3. Vision module runs on right: `{'rgb': rgb_right, 'depth': depth_right}` → outputs            intensity: Snow intensity (0.0 to 1.0), controls density

4. Outputs merged with `_left` and `_right` suffixes            flake_size: Size of snowflakes in pixels

        """

## Examples        self.intensity = intensity

        self.flake_size = flake_size

See [`perception/EXAMPLES.md`](perception/EXAMPLES.md) for complete runnable examples:    

    def render(self, position, orientation_rpy, rgb=None, **kwargs):

- **basic_examples.py** - Mono, stereo, basic composition        if rgb is None:

- **custom_module_example.py** - Decorators vs subclassing            return {}

- **composition_examples.py** - Noise+flow, stereo+events, multi-stage pipelines        

- **stereo_composition_example.py** - Stereo-specific compositions        h, w = rgb.shape[:2]

        snowy_rgb = rgb.copy().astype(float)

## Best Practices        

        # Create snow mask (random snowflakes)

### 1. Pass Through RGB When Needed        num_flakes = int(h * w * self.intensity * 0.001)

        snow_mask = np.zeros((h, w), dtype=np.float32)

If subsequent modules need RGB, pass it through:        

        for _ in range(num_flakes):

```python            x = np.random.randint(0, w)

def render(self, position, orientation_rpy, rgb=None, **kwargs):            y = np.random.randint(0, h)

    if rgb is None:            brightness = np.random.uniform(0.7, 1.0)

        return {}            

                # Draw snowflake

    result = process(rgb)            y_min = max(0, y - self.flake_size // 2)

    return {            y_max = min(h, y + self.flake_size // 2 + 1)

        'my_output': result,            x_min = max(0, x - self.flake_size // 2)

        'rgb': rgb  # Pass through for next module            x_max = min(w, x + self.flake_size // 2 + 1)

    }            snow_mask[y_min:y_max, x_min:x_max] = brightness

```        

        # Apply snow (white overlay)

### 2. Check for Required Inputs        for c in range(3):

            snowy_rgb[:, :, c] = snowy_rgb[:, :, c] * (1 - snow_mask * 0.8) + snow_mask * 255 * 0.8

```python        

def render(self, position, orientation_rpy, **kwargs):        snowy_rgb = np.clip(snowy_rgb, 0, 255).astype(np.uint8)

    rgb = kwargs.get('rgb')        

    if rgb is None:        # Return snowy RGB for next module + snow mask

        return {}  # Can't process without RGB        return {

                'rgb': snowy_rgb,      # Override rgb for next module

    # Continue processing...            'snow_mask': snow_mask  # Additional output

```        }

```

### 3. Use Descriptive Output Keys

### Example: Event Camera Module (VisionModule)

```python

return {```python

    'edges': edges,           # ✓ Goodfrom modules import VisionModule

    'output': result,         # ✗ Too genericimport cv2

    'segmentation_mask': mask # ✓ Goodimport numpy as np

}

```class EventCameraModule(VisionModule):

    """Simulates event camera from RGB intensity changes."""

### 4. Keep Modules Focused    

    def __init__(self, threshold=20):

Each module should do one thing well:        self.threshold = threshold

- ✓ `EdgeDetector` - detects edges        self.prev_frame = None

- ✓ `NoiseAdder` - adds noise    

- ✗ `EdgeDetectorAndNoiseAdder` - too much    def render(self, position, orientation_rpy, rgb=None, **kwargs):

        if rgb is None:

Compose simple modules to create complex pipelines.            return {}

        

## Advanced: Custom BaseModule        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        

Only needed if you have a new rendering method (rare):        if self.prev_frame is not None:

            # Compute events

```python            diff = gray.astype(float) - self.prev_frame.astype(float)

from perception.modules import BaseModule            events = (np.abs(diff) > self.threshold).astype(np.uint8)

            

class MyCustomRenderer(BaseModule):            self.prev_frame = gray

    def __init__(self, config):            return {'events': events, 'polarity': np.sign(diff)}

        self.config = config        

            self.prev_frame = gray

    def render(self, position, orientation_rpy, **kwargs):        return {}

        """Must return rgb and depth at minimum."""```

        rgb = my_rendering_method(position, orientation_rpy)

        depth = my_depth_method(position, orientation_rpy)## Data Flow Example

        

        return {Here's how data flows through a composed pipeline:

            'rgb': rgb,

            'depth': depth```python

        }# Create pipeline: Base Renderer → Snow → Flow

```api = create_mono_api(config, json_path)

composed = api.compose_modules(

## Troubleshooting    ('snow', SnowModule()),

    ('flow', FlowModule())

**Q: My vision module isn't receiving rgb**)

api.register_custom_module('snowy_flow', composed)

A: Check that the previous module returns `'rgb'` in its output dictionary.

# Render

**Q: Outputs have weird names like `_left` or `_right`**results = api.render(position, orientation_rpy)



A: You're using stereo composition. Vision modules run independently on each view.# Data flow:

# 1. Base renderer outputs: {'rgb': <image>, 'depth': <depth>}

**Q: Can I compose BaseModule + BaseModule?**# 2. Snow module receives rgb, outputs: {'rgb': <snowy>, 'snow_mask': <mask>}

# 3. Flow module receives rgb (snowy), outputs: {'flow': <flow>}

A: No, this will raise a TypeError. Use one renderer per pipeline.# 4. Final results with prefixing:

#    - Base: {'rgb': <original>, 'depth': <depth>}

**Q: How do I access intermediate outputs?**#    - Composed: {'snowy_flow_snow_rgb': <snowy>,

#                 'snowy_flow_snow_snow_mask': <mask>,

A: All outputs from all modules are merged into the final results dictionary. Just access by key.#                 'snowy_flow_flow_flow': <flow>}

```

---

## Output Key Naming

For more details, see:

- [README.md](README.md) - Main documentationComposed module outputs are prefixed to avoid conflicts:

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API quick reference

- [perception/EXAMPLES.md](perception/EXAMPLES.md) - Example scripts guide```

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
