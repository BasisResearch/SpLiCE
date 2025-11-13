from .splice import *
from .utils import (
    # Vocabulary embeddings
    compute_vocabulary_embeddings,
    # Model integration
    integrate_custom_model,
    create_concept_dictionary,
    save_image_mean,
    load_image_mean,
    # MSCOCO mean computation
    compute_mscoco_mean,
    get_image_mean,
    stream_embeddings,
    stream_mscoco_images,
    compute_streaming_mean,
)

# Backward compatibility aliases for old function names
compute_image_mean_from_mscoco = compute_mscoco_mean
compute_or_load_image_mean = get_image_mean