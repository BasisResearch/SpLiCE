#!/usr/bin/env python3
"""
General script to integrate any CLIP model with SpLiCE.

This script computes the image mean from MSCOCO train2017 (all ~118K images)
using HuggingFace datasets and creates a concept dictionary for the specified vocabulary.

Memory-efficient implementation: Images are streamed and processed in batches without
loading the entire dataset into memory. Only embeddings are accumulated for computing
the mean, significantly reducing memory requirements.

Requirements:
    pip install datasets

Usage:
    python integrate_model.py --model ViT-L/14@336px --library clip --vocabulary laion
    python integrate_model.py --model ViT-B-32 --library open_clip --vocabulary mscoco
    python integrate_model.py --model ViT-L/14@336px --num-samples 5000  # Use subset for testing
    python integrate_model.py --model ViT-L/14@336px --batch-size 16  # Reduce batch size if OOM
"""

import argparse
import sys
from pathlib import Path
import torch

# Add splice to path
sys.path.insert(0, str(Path(__file__).parent))

from splice import integrate_custom_model, SPLICE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrate a CLIP model with SpLiCE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="CLIP model name (e.g., 'ViT-L/14@336px', 'ViT-B/32', 'RN50')"
    )
    parser.add_argument(
        "--library",
        type=str,
        default="clip",
        choices=["clip", "open_clip"],
        help="CLIP library to use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for saving (defaults to model with slashes replaced by hyphens)"
    )

    # Vocabulary configuration
    parser.add_argument(
        "--vocabulary",
        type=str,
        default="laion",
        help="Vocabulary to use ('laion', 'mscoco', 'laion_bigrams', or path to custom vocab file)"
    )

    # MSCOCO configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of MSCOCO images to use (default: None = use all images)"
    )
    parser.add_argument(
        "--mscoco-split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
        help="MSCOCO split to use"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing images (reduce if running out of memory)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/means/)"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cached files exist"
    )

    # Testing options
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to test image (optional, to verify the integration)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top concepts to display for test image"
    )

    return parser.parse_args()


def load_clip_model(model_name: str, library: str, device: str):
    """Load CLIP model from specified library."""
    print(f"Loading {library}:{model_name}...")

    if library == "clip":
        import clip
        model, preprocess = clip.load(model_name, device=device)
        tokenizer = clip.tokenize
    elif library == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='laion2b_s34b_b79k',
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        raise ValueError(f"Unknown library: {library}")

    return model, preprocess, tokenizer


def load_vocabulary(vocab_arg: str):
    """Load vocabulary from name or file path."""
    # Check if it's a file path
    vocab_path = Path(vocab_arg)
    if vocab_path.exists():
        print(f"Loading vocabulary from file: {vocab_path}")
        with open(vocab_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    # Otherwise treat as vocabulary name
    return vocab_arg


def test_on_image(splice_model, preprocess, image_path: str, vocabulary: str, top_k: int = 10):
    """Test the SpLiCE model on a sample image."""
    from PIL import Image

    print(f"\nTesting on image: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(splice_model.device)

    # Get decomposition
    with torch.no_grad():
        weights = splice_model(image_input)

    # Get top concepts
    top_weights, top_indices = torch.topk(weights[0], k=top_k)

    print(f"\nTop {top_k} concepts:")
    print("-" * 70)

    # Load vocabulary to get concept names
    if isinstance(vocabulary, str):
        vocab_path = Path(__file__).parent / "data" / "vocab" / f"{vocabulary}.txt"
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f.readlines()]
    else:
        vocab = vocabulary

    for i, (idx, weight) in enumerate(zip(top_indices, top_weights), 1):
        concept = vocab[idx.item()]
        print(f"{i:2d}. {concept:40s} (weight: {weight.item():.4f})")


def main():
    args = parse_args()

    print("=" * 70)
    print("SpLiCE Model Integration")
    print("=" * 70)
    print(f"Model: {args.library}:{args.model}")
    print(f"Vocabulary: {args.vocabulary}")
    print(f"MSCOCO split: {args.mscoco_split}")
    if args.num_samples is None:
        print("Using ALL images in MSCOCO split (~118K for train2017)")
    else:
        print(f"Using {args.num_samples} sampled images")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Determine model name for saving
    if args.model_name is None:
        model_name = args.model.replace("/", "-").replace("@", "-")
    else:
        model_name = args.model_name

    # Load CLIP model
    clip_model, preprocess, tokenizer = load_clip_model(
        args.model, args.library, args.device
    )

    # Load vocabulary
    vocabulary = load_vocabulary(args.vocabulary)

    # Integrate with SpLiCE
    print("\nIntegrating model with SpLiCE...")
    print("Using HuggingFace datasets for efficient MSCOCO loading")

    components = integrate_custom_model(
        model_name=model_name,
        clip_model=clip_model,
        preprocess_fn=preprocess,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        library=args.library,
        num_mscoco_samples=args.num_samples,
        mscoco_split=args.mscoco_split,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
    )

    # Create SpLiCE model
    print("\nCreating SpLiCE model...")
    splice_model = SPLICE(
        image_mean=components['image_mean'],
        dictionary=components['dictionary'],
        clip=components['clip'],
        device=args.device,
        l1_penalty=0.01,
        return_weights=True,
    )

    print("\n" + "=" * 70)
    print("Integration complete!")
    print(f"Image mean saved to: data/means/{args.library}_{model_name}_image.pt")
    print("=" * 70)

    # Test on image if provided
    if args.test_image:
        test_on_image(
            splice_model=splice_model,
            preprocess=preprocess,
            image_path=args.test_image,
            vocabulary=args.vocabulary,
            top_k=args.top_k,
        )

    print("\nDone! You can now use this model with SpLiCE:")
    print(f"  from splice import load_image_mean")
    print(f"  image_mean = load_image_mean('{model_name}', library='{args.library}')")


if __name__ == "__main__":
    main()
