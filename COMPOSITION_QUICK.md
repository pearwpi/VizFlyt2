# Module Composition - Quick Summary

## What is it?

Chain perception modules together so outputs from one module flow into the next.

## Simple Example

```python
from perception.API import PerceptionAPI
from composition_demo import SnowModule, FlowModule

# Setup
api = PerceptionAPI("config.yml", "cam_settings.json")
mono = api.get_mono_renderer()

# Create modules
snow = SnowModule(intensity=0.4)
flow = FlowModule()

# Compose: snow + flow (using + operator)
snowy_flow = snow + flow  # VisionModule

# Add to renderer
pipeline = mono + snowy_flow  # BaseModule

# Render
results = pipeline.render(position, orientation_rpy)

# Get outputs
snowy_image = results['rgb']
snow_mask = results['snow_mask']
optical_flow = results['flow']
```

## How it Works

```
BaseModule → VisionModule A → VisionModule B → results
(renderer)   processes RGB    processes A's
             outputs           outputs
```

Composition rules:
- **BaseModule + VisionModule → BaseModule** (rendering + processing)
- **VisionModule + VisionModule → VisionModule** (chained processing)
- **BaseModule + BaseModule → ERROR** (can't chain renderers)

## Writing Composable Modules

```python
class MyModule:
    def render(self, position, orientation_rpy, **kwargs):
        # Get input from previous module
        rgb = kwargs.get('rgb')
        
        # Process
        result = do_something(rgb)
        
        # Return for next module
        return {
            'my_output': result,
            'rgb': rgb  # pass through if needed
        }
```

## Key Points

✅ Modules are chained automatically  
✅ Output keys are auto-prefixed to avoid conflicts  
✅ Create complex pipelines easily  
✅ Reuse modules in different combinations  

## Full Documentation

- **Detailed Guide**: [COMPOSITION_GUIDE.md](COMPOSITION_GUIDE.md)
- **Examples**: `perception/composition_examples.py`
- **API Docs**: [README.md](README.md#module-composition-methods)
