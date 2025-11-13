"""
Utility functions for SpLiCE.

This module provides utilities for:
1. Computing vocabulary embeddings
2. Integrating custom CLIP models into SpLiCE
3. Computing image means from MSCOCO dataset
4. Creating mean-centered concept dictionaries
5. Saving/loading model-specific data
"""

import torch
import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Union
from tqdm.auto import tqdm
from PIL import Image


def compute_vocabulary_embeddings(
    clip_model,
    tokenizer: Callable,
    vocabulary: str,
    model_name: str,
    library: str = "clip",
    vocabulary_size: int = -1,
    data_path: str = None,
    device: str = "cuda",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Compute and cache vocabulary embeddings for any CLIP model.

    Args:
        clip_model: CLIP model with encode_text method
        tokenizer: Tokenizer function (e.g., clip.tokenize)
        vocabulary: Vocabulary name ('laion', 'mscoco', 'laion_bigrams')
        model_name: Model identifier (e.g., 'ViT-L-14-336px', 'ViT-B-32')
        library: Library name for file naming ('clip', 'open_clip')
        vocabulary_size: Number of concepts to use (-1 for full vocabulary)
        data_path: Path to SpLiCE data directory (default: ~/.cache/splice/)
        device: Device to use for computation
        batch_size: Batch size for processing

    Returns:
        torch.Tensor: Normalized vocabulary embeddings [vocab_size, embedding_dim]
    """

    # Setup paths
    if data_path is None:
        data_path = os.path.expanduser("~/.cache/splice/")

    vocab_path = os.path.join(data_path, "vocab", f"{vocabulary}.txt")
    embeddings_dir = os.path.join(data_path, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Create filename
    vocab_size_str = "full" if vocabulary_size <= 0 else str(vocabulary_size)
    embeddings_file = f"{library}_{model_name}_{vocabulary}_{vocab_size_str}_embeddings.pt"
    embeddings_path = os.path.join(embeddings_dir, embeddings_file)

    # Check cache
    if os.path.isfile(embeddings_path):
        print(f"Loading cached embeddings: {embeddings_file}")
        embeddings = torch.load(embeddings_path, map_location=device)
        # Ensure float32 dtype for compatibility
        return embeddings.float()

    # Load vocabulary
    print(f"Computing embeddings for vocabulary: {vocabulary}")
    with open(vocab_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        if vocabulary_size > 0:
            lines = lines[-vocabulary_size:]

    # Compute embeddings in batches
    clip_model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            batch_emb = clip_model.encode_text(tokens)
            embeddings.append(batch_emb.cpu())

            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, len(lines))}/{len(lines)}")

    # Stack and normalize
    concepts = torch.cat(embeddings, dim=0).to(device)
    if concepts.dim() > 2:
        concepts = concepts.squeeze()

    # Ensure float32 dtype
    concepts = concepts.float()

    # Normalize: L2 → center → L2
    concepts = torch.nn.functional.normalize(concepts, dim=1)
    concepts = concepts - torch.mean(concepts, dim=0, keepdim=True)
    concepts = torch.nn.functional.normalize(concepts, dim=1)

    # Cache result
    torch.save(concepts, embeddings_path)
    print(f"Saved embeddings: {embeddings_file} (shape: {concepts.shape}, dtype: {concepts.dtype})")

    return concepts


# ============================================================================
# Integration functions for custom CLIP models
# ============================================================================

def _process_image_batch(
    images: List[Image.Image],
    clip_model: torch.nn.Module,
    preprocess_fn: Callable,
    device: str,
) -> torch.Tensor:
    """
    Process a batch of images and return normalized embeddings.

    Args:
        images: List of PIL Images
        clip_model: CLIP model with encode_image method
        preprocess_fn: Preprocessing function for images
        device: Device to use for computation

    Returns:
        torch.Tensor of normalized embeddings (batch_size, embedding_dim)
    """
    batch_tensors = torch.stack([preprocess_fn(img) for img in images]).to(device)

    with torch.no_grad():
        embeddings = clip_model.encode_image(batch_tensors).float()
        # Normalize to unit sphere
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu()


def stream_embeddings(
    image_generator,
    clip_model: torch.nn.Module,
    preprocess_fn: Callable,
    device: str = "cuda",
    batch_size: int = 32,
    total_images: Optional[int] = None,
    progress_bar: Optional[tqdm] = None,
):
    """
    Stream CLIP embeddings from images without loading all into memory.

    Args:
        image_generator: Generator that yields PIL Images
        clip_model: CLIP model with encode_image method
        preprocess_fn: Preprocessing function for images
        device: Device to use for computation
        batch_size: Batch size for processing
        total_images: Total number of images (for progress bar)
        progress_bar: Optional existing tqdm progress bar to update

    Yields:
        torch.Tensor batches of normalized embeddings
    """
    import gc

    batch_images = []

    for img in image_generator:
        batch_images.append(img)

        # Update progress bar for each image loaded
        if progress_bar is not None:
            progress_bar.update(1)

        if len(batch_images) >= batch_size:
            yield _process_image_batch(batch_images, clip_model, preprocess_fn, device)
            batch_images = []

            # Free memory for large batches
            if batch_size > 16:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    # Process remaining images
    if batch_images:
        yield _process_image_batch(batch_images, clip_model, preprocess_fn, device)


def compute_streaming_mean(embedding_generator):
    """
    Compute mean from embedding stream using incremental calculation.

    Args:
        embedding_generator: Generator yielding embedding batches

    Returns:
        Tuple of (mean_tensor, total_count)
    """
    running_sum = None
    total_count = 0

    for batch_embeddings in embedding_generator:
        if running_sum is None:
            running_sum = batch_embeddings.sum(dim=0)
        else:
            running_sum += batch_embeddings.sum(dim=0)

        total_count += batch_embeddings.shape[0]

    if running_sum is None or total_count == 0:
        raise ValueError("No embeddings were processed")

    return running_sum / total_count, total_count


def _get_mscoco_dataset(
    split: str = "train2017",
    num_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
):
    """
    Get MSCOCO dataset and determine total count.

    Returns:
        Tuple of (dataset, total_count) where total_count can be None if unknown.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library not found. "
            "Install with: pip install datasets"
        )

    print(f"Loading MSCOCO {split} using HuggingFace datasets...")

    # Map split names
    hf_split = "train" if split == "train2017" else "validation"

    # Try multiple COCO dataset sources
    dataset_sources = [
        ("detection-datasets/coco", {}),  # Newer parquet-based format
        ("HuggingFaceM4/COCO", {"trust_remote_code": True}),  # Original with trust flag
    ]

    dataset = None
    for source, kwargs in dataset_sources:
        try:
            print(f"Trying dataset source: {source}")
            dataset = load_dataset(
                source,
                split=hf_split,
                cache_dir=str(cache_dir) if cache_dir else None,
                **kwargs
            )
            print(f"Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"Failed to load from {source}: {e}")
            continue

    if dataset is None:
        # Fallback to streaming mode if all else fails
        print("Falling back to streaming mode...")
        try:
            dataset = load_dataset(
                "HuggingFaceM4/COCO",
                split=hf_split,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MSCOCO dataset from all sources. "
                f"Last error: {e}\n"
                f"Please check your datasets library version or use a manual download approach."
            )

    # Determine total count
    total_count = None
    if hasattr(dataset, '__len__'):
        total_size = len(dataset)
        if num_samples is not None and num_samples < total_size:
            indices = np.random.choice(total_size, num_samples, replace=False)
            dataset = dataset.select(indices)
            total_count = num_samples
        else:
            total_count = total_size
        print(f"Will process {total_count} images")
    else:
        # Streaming mode - only know count if num_samples specified
        total_count = num_samples
        if num_samples:
            print(f"Streaming images (target: {num_samples})...")
        else:
            print(f"Streaming images (all available)...")

    return dataset, total_count


def stream_mscoco_images(
    split: str = "train2017",
    num_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
):
    """
    Stream MSCOCO images from HuggingFace datasets.

    Args:
        split: MSCOCO split ("train2017" or "val2017")
        num_samples: Number of samples (None = all)
        cache_dir: HuggingFace datasets cache directory

    Yields:
        PIL Images one at a time
    """
    dataset, _ = _get_mscoco_dataset(split, num_samples, cache_dir)

    # Generator to yield images one at a time
    count = 0
    target_count = num_samples if num_samples is not None else float('inf')

    for item in dataset:
        if count >= target_count:
            break
        if 'image' in item:
            img = item['image']
            if isinstance(img, Image.Image):
                yield img.convert('RGB')
                count += 1


def compute_mscoco_mean(
    clip_model: torch.nn.Module,
    preprocess_fn: Callable,
    num_samples: Optional[int] = None,
    split: str = "train2017",
    device: str = "cuda",
    batch_size: int = 32,
    cache_dir: Optional[Path] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Compute normalized image mean from MSCOCO using streaming.

    Args:
        clip_model: CLIP model with encode_image method
        preprocess_fn: Image preprocessing function
        num_samples: Number of images (None = all)
        split: MSCOCO split ("train2017" or "val2017")
        device: Device for computation
        batch_size: Batch size for processing
        cache_dir: HuggingFace datasets cache directory
        seed: Random seed for sampling

    Returns:
        Normalized image mean tensor (embedding_dim,)
    """
    print("=" * 70)
    print(f"Computing image mean from MSCOCO {split} (memory-efficient streaming)")
    if num_samples is None:
        print("Using ALL images in the split")
    else:
        print(f"Using {num_samples} sampled images")
    print("=" * 70)

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    else:
        cache_dir = Path(cache_dir)

    # Set random seed
    np.random.seed(seed)

    # Get dataset and total count
    print("\n[1/2] Setting up MSCOCO image stream...")
    dataset, total_count = _get_mscoco_dataset(
        split=split,
        num_samples=num_samples,
        cache_dir=cache_dir,
    )

    # Create image generator from dataset
    def image_generator():
        count = 0
        target = num_samples if num_samples is not None else float('inf')
        for item in dataset:
            if count >= target:
                break
            if 'image' in item:
                img = item['image']
                if isinstance(img, Image.Image):
                    yield img.convert('RGB')
                    count += 1

    # Stream embeddings and compute mean incrementally
    print(f"\n[2/2] Processing images and computing mean (batch_size={batch_size})...")

    # Create progress bar with total count
    pbar_kwargs = {
        "desc": "Processing images",
        "unit": "img",
    }
    if total_count is not None:
        pbar_kwargs["total"] = total_count

    with tqdm(**pbar_kwargs) as pbar:
        embedding_stream = stream_embeddings(
            image_generator=image_generator(),
            clip_model=clip_model,
            preprocess_fn=preprocess_fn,
            device=device,
            batch_size=batch_size,
            total_images=total_count,
            progress_bar=pbar,
        )

        # Compute and normalize mean
        image_mean, processed_count = compute_streaming_mean(embedding_stream)

    image_mean = image_mean / image_mean.norm()

    print("\n" + "=" * 70)
    print(f"Image mean computed successfully from {processed_count} images!")
    print(f"Shape: {image_mean.shape}")
    print(f"Norm: {image_mean.norm().item():.6f}")
    print("=" * 70)

    return image_mean


def create_concept_dictionary(
    clip_model: torch.nn.Module,
    tokenizer: Callable,
    vocabulary: List[str],
    device: str = "cuda",
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Create mean-centered, normalized concept dictionary.

    Args:
        clip_model: CLIP model with encode_text method
        tokenizer: Text tokenizer function
        vocabulary: List of concept strings
        device: Device for computation
        batch_size: Batch size for encoding

    Returns:
        Normalized concept dictionary (num_concepts, embedding_dim)
    """
    print(f"Creating concept dictionary for {len(vocabulary)} concepts...")

    # Encode concepts in batches
    concept_embeddings = []
    for i in tqdm(range(0, len(vocabulary), batch_size), desc="Encoding concepts"):
        batch = vocabulary[i:i + batch_size]

        # Tokenize and encode
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens).float()
            # Normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            concept_embeddings.append(feats.cpu())

    # Concatenate all embeddings
    concepts = torch.cat(concept_embeddings, dim=0)

    # Mean-center the concepts (compute μcon and subtract)
    concept_mean = concepts.mean(dim=0, keepdim=True)
    concepts_centered = concepts - concept_mean

    # Normalize to unit sphere
    concepts_normalized = concepts_centered / concepts_centered.norm(dim=-1, keepdim=True)

    print(f"Concept dictionary created: {concepts_normalized.shape}")
    return concepts_normalized


def save_image_mean(
    image_mean: torch.Tensor,
    model_name: str,
    library: str = "clip",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save image mean following SpLiCE naming conventions.

    Args:
        image_mean: Image mean tensor
        model_name: Model name (e.g., "ViT-L-14-336px")
        library: Library name ("clip" or "open_clip")
        output_dir: Output directory (default: data/means/)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        # Save to SpLiCE data directory
        output_dir = Path(__file__).parent.parent / "data" / "means"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Follow SpLiCE naming convention: {library}_{model}_image.pt
    filename = f"{library}_{model_name}_image.pt"
    output_path = output_dir / filename

    torch.save(image_mean, output_path)
    print(f"Saved image mean to: {output_path}")

    return output_path


def load_image_mean(
    model_name: str,
    library: str = "clip",
    means_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load a saved image mean.

    Args:
        model_name: Model name (e.g., "ViT-L-14-336px")
        library: Library name ("clip" or "open_clip")
        means_dir: Directory with cached means

    Returns:
        Image mean tensor
    """
    if means_dir is None:
        means_dir = Path(__file__).parent.parent / "data" / "means"
    else:
        means_dir = Path(means_dir)

    filename = f"{library}_{model_name}_image.pt"
    mean_path = means_dir / filename

    if not mean_path.exists():
        raise FileNotFoundError(
            f"Image mean not found at {mean_path}. "
            f"Please compute it first using compute_mscoco_mean()"
        )

    return torch.load(mean_path)


def get_image_mean(
    model_name: str,
    clip_model: torch.nn.Module,
    preprocess_fn: Callable,
    library: str = "clip",
    num_samples: Optional[int] = None,
    split: str = "train2017",
    device: str = "cuda",
    batch_size: int = 32,
    means_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    Get image mean from cache or compute it from MSCOCO.

    Args:
        model_name: Model name (e.g., "ViT-L-14-336px")
        clip_model: CLIP model (used if computing)
        preprocess_fn: Image preprocessing function
        library: Library name ("clip" or "open_clip")
        num_samples: Number of images (None = all)
        split: MSCOCO split
        device: Device for computation
        batch_size: Batch size for processing
        means_dir: Directory for cached means
        force_recompute: Force recomputation

    Returns:
        Tuple of (image_mean, was_cached)
    """
    if means_dir is None:
        means_dir = Path(__file__).parent.parent / "data" / "means"
    else:
        means_dir = Path(means_dir)

    filename = f"{library}_{model_name}_image.pt"
    mean_path = means_dir / filename

    # Try to load if exists and not forcing recompute
    if mean_path.exists() and not force_recompute:
        print(f"Loading cached image mean from: {mean_path}")
        image_mean = torch.load(mean_path)
        return image_mean, True

    # Compute
    print(f"Computing image mean (will be cached to {mean_path})...")
    image_mean = compute_mscoco_mean(
        clip_model=clip_model,
        preprocess_fn=preprocess_fn,
        num_samples=num_samples,
        split=split,
        device=device,
        batch_size=batch_size,
    )

    # Save
    save_image_mean(
        image_mean=image_mean,
        model_name=model_name,
        library=library,
        output_dir=means_dir,
    )

    return image_mean, False


def integrate_custom_model(
    model_name: str,
    clip_model: torch.nn.Module,
    preprocess_fn: Callable,
    tokenizer: Callable,
    vocabulary: Union[str, List[str]],
    library: str = "clip",
    num_mscoco_samples: Optional[int] = None,
    mscoco_split: str = "train2017",
    device: str = "cuda",
    batch_size: int = 32,
    output_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> dict:
    """
    Integrate a custom CLIP model into SpLiCE.

    Gets/computes image mean from MSCOCO, creates concept dictionary,
    and returns all components needed for a SPLICE model.

    Args:
        model_name: Model name (e.g., "ViT-L-14-336px")
        clip_model: CLIP model with encode_image/encode_text methods
        preprocess_fn: Image preprocessing function
        tokenizer: Text tokenizer function
        vocabulary: Vocabulary name or list of concepts
        library: Library name ("clip" or "open_clip")
        num_mscoco_samples: Number of MSCOCO images (None = all)
        mscoco_split: MSCOCO split ("train2017" or "val2017")
        device: Device for computation
        batch_size: Batch size for processing
        output_dir: Directory for saving files
        force_recompute: Force recomputation of cached files

    Returns:
        Dict with: image_mean, dictionary, clip, device, model_name, library
    """
    print("=" * 70)
    print(f"Integrating custom CLIP model: {library}:{model_name}")
    print("=" * 70)

    # Step 1: Get image mean (from cache or compute)
    print("\n[1/3] Getting image mean from MSCOCO...")
    image_mean, was_cached = get_image_mean(
        model_name=model_name,
        clip_model=clip_model,
        preprocess_fn=preprocess_fn,
        library=library,
        num_samples=num_mscoco_samples,
        split=mscoco_split,
        device=device,
        batch_size=batch_size,
        means_dir=output_dir / "means" if output_dir else None,
        force_recompute=force_recompute,
    )

    # Step 2: Load vocabulary
    print("\n[2/3] Loading vocabulary...")
    if isinstance(vocabulary, str):
        # Load from SpLiCE vocabularies
        vocab_path = Path(__file__).parent.parent / "data" / "vocab" / f"{vocabulary}.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary '{vocabulary}' not found at {vocab_path}. "
                f"Please provide a list of concepts instead."
            )
        with open(vocab_path, 'r') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        print(f"Loaded vocabulary '{vocabulary}' with {len(vocab_list)} concepts")
    else:
        vocab_list = vocabulary
        print(f"Using custom vocabulary with {len(vocab_list)} concepts")

    # Step 3: Create concept dictionary
    print("\n[3/3] Creating concept dictionary...")
    concept_dict = create_concept_dictionary(
        clip_model=clip_model,
        tokenizer=tokenizer,
        vocabulary=vocab_list,
        device=device,
    )

    print("\n" + "=" * 70)
    print("Model integration complete!")
    print(f"  - Image mean shape: {image_mean.shape}")
    print(f"  - Dictionary shape: {concept_dict.shape}")
    print("=" * 70)

    return {
        'image_mean': image_mean,
        'dictionary': concept_dict,
        'clip': clip_model,
        'device': device,
        'model_name': model_name,
        'library': library,
    }
