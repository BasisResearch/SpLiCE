# SpLiCE Examples

This directory contains examples demonstrating how to integrate and use SpLiCE with various CLIP models.

## Quick Start

Install requirements:
```bash
pip install datasets torch pillow tqdm
```

## Examples

### 1. Command Line Integration Tool

`integrate_model.py` - General-purpose script to integrate any CLIP model with SpLiCE.

```bash
# Basic usage
python examples/integrate_model.py --model "ViT-L/14@336px" --library clip --vocabulary laion

# Test with subset (faster)
python examples/integrate_model.py --model "ViT-L/14@336px" --num-samples 5000

# Custom options
python examples/integrate_model.py --model "ViT-B/32" --batch-size 64 --test-image path/to/image.jpg
```

### 2. Python Integration Examples

**`integrate_vitl14_336px.py`** - Integrate higher-resolution CLIP model
```bash
python examples/integrate_vitl14_336px.py
```

**`integrate_maskclip.py`** - Integrate MaskCLIP (for collab-splats users)
```bash
python examples/integrate_maskclip.py
```

### 3. Comparison Demo

**`clip_vs_maskclip_demo.ipynb`** - Interactive notebook comparing CLIP and MaskCLIP decompositions

Open in Jupyter to explore visual decompositions side-by-side.

## Custom Integration

Integrate any CLIP model with a few lines:

```python
from splice import integrate_custom_model, SPLICE
import clip

# Load your model
model, preprocess = clip.load("YourModel", device="cuda")

# Integrate
components = integrate_custom_model(
    model_name="YourModel",
    clip_model=model,
    preprocess_fn=preprocess,
    tokenizer=clip.tokenize,
    vocabulary="laion",
    library="clip",
    device="cuda",
)

# Create SpLiCE model
splice_model = SPLICE(
    image_mean=components['image_mean'],
    dictionary=components['dictionary'],
    clip=components['clip'],
    device="cuda",
    l1_penalty=0.01,
    return_weights=True,
)
```

## Understanding the Integration

### Image Mean (Cone Center)
The image mean represents the center of the CLIP embedding cone, computed from MSCOCO images. This is cached after first computation.

**Default:** All ~118K images from MSCOCO train2017 (~2-3 hours first run, then instant)
**Testing:** Set `num_mscoco_samples=5000` for faster testing (~15 minutes)

### Concept Dictionary
Mean-centered text embeddings representing concepts. SpLiCE decomposes images as sparse combinations of these.

### Caching
Files are cached to avoid recomputation:
- Image means: `data/means/{library}_{model}_image.pt`
- MSCOCO dataset: `~/.cache/huggingface/datasets`

## Parameters

**Integration:**
- `num_mscoco_samples`: Number of images (None = all ~118K, 5000 for testing)
- `batch_size`: Reduce if OOM (default: 32)
- `force_recompute`: Recompute even if cached

**SpLiCE Model:**
- `l1_penalty`: Sparsity control (higher = sparser, default: 0.01)
- `return_weights`: Return concept weights instead of reconstructed embeddings
