# VizFlyt2 Perception Examples

Quick guide to the example scripts.

## 📁 Example Files

### 1. **basic_examples.py** - Start Here!
Simple examples to get started with rendering and basic composition.

**What it covers:**
- Mono camera rendering (RGB + depth)
- Stereo camera rendering (left/right pairs)
- Basic module composition with `+` operator

**Run it:**
```bash
python basic_examples.py
```

---

### 2. **custom_module_example.py** - Creating Your Own Modules
Shows TWO ways to create custom VisionModules.

**What it covers:**
- **Method 1:** Factory decorators (`@rgb_vision_module_factory`, `@vision_module_factory`)
  - Quick & simple for stateless functions
  - Examples: edge detection, brightness adjustment, RGB+depth blending
- **Method 2:** Subclassing (`class MyModule(VisionModule)`)
  - Full control with initialization parameters
  - Examples: configurable edge detection, depth colorizer, temporal filtering
- When to use each method
- Both work seamlessly with `+` operator composition

**Run it:**
```bash
python custom_module_example.py
```

---

### 3. **composition_examples.py** - Complex Pipelines
Advanced composition patterns combining multiple modules.

**What it covers:**
- **Noise + Optical Flow:** Adds noise to images, computes optical flow, saves both RGB and flow visualization
- **Stereo + Event Camera:** Simulates event cameras on stereo pairs, saves both RGB and events
- **Custom Multi-Stage Pipeline:** noise → blur → edge detection, saves intermediate and final results
- Interactive menu to select examples

**Run it:**
```bash
python composition_examples.py
```

---

### 4. **stereo_composition_example.py** - Stereo-Specific Composition
Demonstrates how vision modules work with stereo cameras.

**What it covers:**
- Base stereo rendering (no composition)
- Stereo + Snow effect (applied to both views)
- Stereo + Edge detection (applied to both views)
- Stereo + chained modules (snow → edges on both views)
- How StereoComposedBase works internally

**Key concept:** Vision modules run independently on left and right views, producing `_left` and `_right` outputs.

**Run it:**
```bash
python stereo_composition_example.py
```

---

## 🚀 Quick Start

1. **Update the paths** in each example to point to your splat config and camera settings:
   ```python
   config_path = "../splats/your_splat/config.yml"
   json_path = "../splats/cam_settings.json"
   ```

2. **Run the basic example first:**
   ```bash
   python basic_examples.py
   ```

3. **Try creating a custom module** with decorators:
   ```bash
   python custom_module_example.py
   ```

4. **Explore advanced compositions:**
   ```bash
   python composition_examples.py
   ```

---

## 📊 What Each Example Demonstrates

| Example | Decorators | Subclassing | Composition | Stereo | Complexity |
|---------|-----------|-------------|-------------|--------|------------|
| basic_examples.py | ❌ | ❌ | ✅ Basic | ✅ | ⭐ Easy |
| custom_module_example.py | ✅ | ✅ | ✅ | ❌ | ⭐⭐ Medium |
| composition_examples.py | ❌ | ✅ | ✅ Advanced | ✅ | ⭐⭐⭐ Advanced |
| stereo_composition_example.py | ❌ | ✅ | ✅ Stereo | ✅ | ⭐⭐ Medium |

---

## 💡 Learning Path

**Beginner:**
1. Start with `basic_examples.py` to understand rendering basics
2. Try `custom_module_example.py` to learn decorator pattern

**Intermediate:**
3. Explore `stereo_composition_example.py` for stereo-specific features
4. Run `composition_examples.py` for complex pipelines

**Advanced:**
- Read `COMPOSITION_GUIDE.md` for architecture details
- Check `QUICK_REFERENCE.md` for API reference
- Create your own custom modules!

---

## 📝 All Examples Use

- **Direct module instantiation** (no API layer)
- **`+` operator** for composition
- **Outputs folder** for all saved images
- **Origin rendering** (position [0, 0, 0] for simplicity)
- **Parameter-based paths** (no hardcoded paths in multiple places)
