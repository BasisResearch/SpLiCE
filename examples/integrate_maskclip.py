"""
Example: Integrate MaskCLIP with SpLiCE

This example demonstrates how to integrate MaskCLIP (used in collab-splats)
with SpLiCE for semantic decomposition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import maskclip_onnx.clip as clip
from splice import integrate_custom_model, SPLICE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14@336px"

    print("Loading MaskCLIP model...")
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
    print("You can now use this with MaskCLIPExtractor from collab-splats")

    return splice_model


if __name__ == "__main__":
    main()
