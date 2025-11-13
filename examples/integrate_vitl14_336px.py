"""
Example: Integrate ViT-L/14@336px CLIP model with SpLiCE

This example shows how to integrate a higher-resolution CLIP model
and test it on an image.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import clip
from splice import integrate_custom_model, SPLICE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14@336px"

    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(model_name, device=device)

    print("\nIntegrating with SpLiCE...")
    components = integrate_custom_model(
        model_name="ViT-L-14-336px",
        clip_model=clip_model,
        preprocess_fn=preprocess,
        tokenizer=clip.tokenize,
        vocabulary="laion",
        library="clip",
        device=device,
    )

    print("\nCreating SpLiCE model...")
    splice_model = SPLICE(
        image_mean=components['image_mean'],
        dictionary=components['dictionary'],
        clip=components['clip'],
        device=device,
        l1_penalty=0.01,
        return_weights=True,
    )

    print("\nIntegration complete!")
    print(f"Model ready for use with {len(components['dictionary'])} concepts")

    return splice_model


if __name__ == "__main__":
    main()
