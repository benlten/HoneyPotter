# Honeypotter 0.2.0

**Albumentations for *distractors*.** Honeypotter generates **non-semantic categories** that exist solely to **misclassify** models and stress robustness. It supports **in-distribution and OOD color modes**, patterns, primitives, textures, and blur variants.

> ⚠️ **Important for evaluation**: Honeypotter classes (e.g., `checker_sz16_bw`, `stripes_45deg_w12_ood`) are NOT semantic object classes. To test properly, **append these category names to your dataset’s label space** (e.g., ImageNet-1k → ImageNet-1k + K honeypot classes) and measure how often the classifier **falls for** these distractors (steal-rate).

## Features
- **Pattern families**: `checker`, `stripes`, `dots`, `blob`, `solid`, optional `circle`, `perlin`, `texture`, and `moire`
- **Color modes**:
  - `color` - Random RGB or sophisticated gamut-based sampling
  - `gray` - Grayscale variations
  - `bw` - Black and white high-contrast
  - `ood` - Neon/unrealistic out-of-distribution colors
- **Gamut support**: Sophisticated color boundary detection using histogram-HDR or GMM methods
- **Frequency control**:
  - `--pattern_blur_sigma` → embedded blur with label tracking (e.g., `_blurS3`)
  - `--global_blur_sigma` → defocus blur (unlabeled)
  - Pattern size controls for high/low frequency testing
- **Texture integration**: `--texture_dir`, `--include_texture_family`, `--texture_mix_prob`
- **Output formats**: `labelmap.json`, `labels.txt`, `metadata.csv` for easy integration

## Installation

### From PyPI (Recommended)
```bash
pip install honeypotter
```

### Development Install
```bash
git clone https://github.com/yourusername/honeypotter
cd honeypotter
pip install -e .[dev]
```

## Quick Start

### Basic Usage
```bash
# Generate 20 distractor categories with 100 images each
honeypotter generate --out ./distractors

# Quick test with fewer images
honeypotter generate --out ./test --n_categories 5 --per_class 10
```

### Advanced Usage
```bash
# High-frequency patterns with gamut-based colors
honeypotter generate --out ./high_freq \
  --n_categories 50 --per_class 200 \
  --families checker,stripes \
  --checker_size_min 4 --checker_size_max 8 \
  --stripe_width_min 3 --stripe_width_max 6 \
  --gamut ./my_dataset_gamut.json \
  --seed 42

# Low-frequency with blur testing
honeypotter generate --out ./blur_test \
  --families blob,perlin \
  --pattern_blur_sigma 3.0 \
  --global_blur_sigma 1.5 \
  --color_mode ood

# ViT-aligned moiré distractors
honeypotter generate --out ./moire_vit \
  --n_categories 20 --per_class 100 \
  --enable_moire --families moire \
  --moire_grid_size 16 \
  --no-moire-random-hue --moire_hues 15,195,305
```

Output creates organized directories like: `checker_sz8_ood/`, `stripes_45deg_w4_blurS3/`, etc.

### OOD Color Palette Format
`--ood_palette` accepts a JSON with either:
```json
{"rgb_hex": ["#00FFFF", "#FF00FF", "#00FF66"]}
```
or
```json  
{"hsv": [[0.5, 0.9, 1.0], [0.9, 0.9, 1.0]]}
```

If omitted, Honeypotter uses a **default neon OOD palette**.

## Programmatic API

### Basic Usage
```python
from honeypotter import GenerateDistractor, CategorySpec, ColorSampler

# Simple random RGB generation
sampler = ColorSampler(seed=42)
spec = CategorySpec.default(
    families=["checker", "stripes", "dots", "blob", "solid"],
    n_categories=20,
    checker_size_range=(8, 32),
    stripe_width_range=(6, 28),
    color_mode="color"
)

gen = GenerateDistractor(
    image_size=224,
    spec=spec,
    color_sampler=sampler,
    seed=42
)

# Generate single image
img, name = gen()  # Returns PIL Image and category name like "checker_sz16"
```

### Advanced Usage with Gamut
```python
from honeypotter import ColorSampler

# Gamut-based color sampling
gamut_sampler = ColorSampler(gamut_path="./dataset_gamut.json", seed=42)

spec = CategorySpec.default(
    families=["checker", "stripes", "perlin"],
    n_categories=50,
    pattern_blur_sigma=2.0,  # Adds blur to category names
    color_mode="ood"         # Use OOD colors
)

gen = GenerateDistractor(
    image_size=224,
    spec=spec, 
    color_sampler=gamut_sampler,
    global_blur_sigma=1.0,   # Unlabeled defocus blur
    seed=42
)

# Generate batch
for category_config in gen.categories:
    for i in range(10):
        img, name = gen()
        img.save(f"{name}_{i:03d}.png")
```

## Evaluation Protocol

### Core Principle: Honeypots are NOT semantic classes
Honeypotter generates **distractor categories** designed to test robustness, not represent real objects. They should **never** be used as training targets.

### Dataset Integration

#### 1. Generate Distractors
```bash
# Generate 50 honeypot categories for ImageNet evaluation
honeypotter generate \
  --out ./imagenet_distractors \
  --n_categories 50 --per_class 100 \
  --families checker,stripes,dots,blob \
  --color_mode ood --seed 42
```

#### 2. Extend Label Space 
```python
# Original ImageNet: 1000 classes (0-999)
# Extended: 1050 classes (0-999: ImageNet, 1000-1049: Honeypots)

original_labels = list(range(1000))  # ImageNet classes 0-999
honeypot_labels = list(range(1000, 1050))  # Honeypot classes 1000-1049

# Load honeypot label mapping
import json
with open("./imagenet_distractors/labelmap.json", "r") as f:
    honeypot_mapping = json.load(f)

# Create full label space
full_label_space = {
    **{f"imagenet_class_{i}": i for i in original_labels},
    **{name: 1000 + idx for name, idx in honeypot_mapping.items()}
}
```

#### 3. Training Exclusion (CRITICAL)
```python
def create_dataloaders(dataset_path, honeypot_path):
    """Create train/val loaders with honeypots EXCLUDED from training."""
    
    # Training: ONLY original dataset
    train_dataset = ImageFolder(
        root=f"{dataset_path}/train",
        transform=train_transforms
    )
    
    # Validation: Original + Honeypots
    val_original = ImageFolder(
        root=f"{dataset_path}/val", 
        transform=val_transforms
    )
    
    val_honeypots = ImageFolder(
        root=honeypot_path,
        transform=val_transforms,
        # CRITICAL: Map honeypot labels to extended range
        target_transform=lambda x: x + 1000  
    )
    
    # Combined validation set
    val_dataset = ConcatDataset([val_original, val_honeypots])
    
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(val_dataset, batch_size=32, shuffle=False)
    )
```

### Evaluation Metrics

#### Required Metrics
```python
def evaluate_with_honeypots(model, val_loader, num_original_classes=1000):
    """Evaluate model on original + honeypot classes."""
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Separate original and honeypot predictions
    original_mask = np.array(all_labels) < num_original_classes
    honeypot_mask = ~original_mask
    
    # Core metrics
    clean_accuracy = accuracy_score(
        np.array(all_labels)[original_mask], 
        np.array(all_preds)[original_mask]
    )
    
    # Honeypot "accuracy" (should be low!)
    honeypot_predictions = np.array(all_preds)[honeypot_mask]
    honeypot_in_original_space = (honeypot_predictions < num_original_classes).mean()
    
    # Steal rate: how often model predicts original class for honeypots
    steal_rate = honeypot_in_original_space
    
    return {
        "clean_accuracy": clean_accuracy,
        "steal_rate": steal_rate,  # Lower is better
        "honeypot_confusion": 1 - honeypot_in_original_space  # Higher is better
    }
```

#### Report These Metrics
1. **Clean Accuracy**: Performance on original validation set
2. **Steal Rate**: Fraction of honeypots classified as original classes (lower is better)
3. **Honeypot Confusion**: Model's uncertainty on distractors (higher is better)
4. **Calibration**: ECE/reliability on honeypots vs. clean samples

### Common Mistakes to Avoid

#### DON'T: Train on Honeypots
```python
# WRONG: Including honeypots in training
train_dataset = ConcatDataset([original_train, honeypots])  # Never do this!
```

#### DON'T: Map Honeypots to Semantic Classes  
```python
# WRONG: Treating honeypots as "background" or existing class
honeypot_labels = [0] * len(honeypots)  # Maps to class 0 - incorrect!
```

#### DO: Keep Separate Label Space
```python  
# CORRECT: Extend label space for honeypots
original_labels = list(range(1000))          # ImageNet: 0-999
honeypot_labels = list(range(1000, 1050))    # Distractors: 1000-1049
```

### Research Applications

- **Robustness Testing**: How do models handle non-semantic distractors?
- **Calibration Studies**: Are models well-calibrated on unexpected inputs?
- **Architecture Comparison**: Which architectures are more/less susceptible?
- **Training Method Evaluation**: Do robust training methods help with honeypots?

## Docs & Paper
- Static site under `/docs` (GitHub Pages ready) with palette vs OOD samples.
- LaTeX paper in `/paper` includes a section on **OOD colors** and the evaluation protocol.


## Contributing

We welcome contributions! Please see our [GitHub Issues](https://github.com/yourusername/honeypotter/issues) for:

- 🐛 **Bug reports** - Report unexpected behavior
- 💡 **Feature requests** - Suggest new capabilities  
- 📚 **Documentation** - Help improve our docs
- 🔬 **Research ideas** - Propose new distractor types

### Quick Start for Contributors
```bash
git clone https://github.com/yourusername/honeypotter
cd honeypotter
pip install -e .[dev]
pytest tests/
```

## Related Work
- Albumentations \cite{albumentations18, albumentations20}
- Stylized-ImageNet / texture bias \cite{geirhos19}  
- ImageNet-C/P \cite{hendrycks19c}
- AugMix \cite{augmix20} and DeepAugment \cite{manyfaces21}
- ImageNet-A/O/R, ObjectNet, ImageNet-Sketch \cite{imageneta19, imagenetr21, objectnet19, wang2019sketch}
- WILDS benchmark \cite{koh2021wilds}
- THINGS (naturalistic palette reference) \cite{things19, things23}
### Moires for ViT Evaluation

Moiré patterns are now available as a first-class family. Enable them with `--enable_moire` (or include `moire` in `--families`). By default Honeypotter snaps moiré intensities to a `16×16` grid so each ViT patch observes a consistent signal. Adjust the grid with `--moire_grid_size` (set to `0` or `1` to disable).

- **Hue control**: Random hues are used by default. Opt into deterministic palettes with `--no-moire-random-hue` and provide a comma-separated list via `--moire_hues 15,195,305`. If omitted, Honeypotter falls back to an internal neon-inspired list.
- **Frequency controls**: Tune interference strength with `--moire_frequency_min/max` and `--moire_frequency_delta_min/max`. Small deltas intensify the moiré banding.
- **Contrast**: Use `--moire_saturation_*` and `--moire_value_*` to adjust colorfulness and brightness of the bands.

Programmatic usage mirrors the CLI:

```python
from honeypotter import GenerateDistractor, CategorySpec

spec = CategorySpec.default(
    families=["moire"],
    n_categories=10,
    moire_grid_size=16,
    moire_random_hue=False,
    moire_hue_list=[15.0, 195.0, 305.0],
)

gen = GenerateDistractor(image_size=224, spec=spec, seed=123)
img, name = gen()
```
