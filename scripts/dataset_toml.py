"""Dataset TOML config generator for kohya-ss training scripts.

Supports both single-folder datasets and multi-folder datasets
(e.g., base character + outfit variations in subdirectories).
"""

import toml
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _count_images_in_dir(dir_path: Path) -> int:
    """Count image files in a single directory (non-recursive)."""
    return sum(
        1 for entry in dir_path.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )


def _count_captions_in_dir(dir_path: Path) -> int:
    """Count .txt caption files in a single directory (non-recursive)."""
    return sum(
        1 for entry in dir_path.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".txt"
    )


def discover_subsets(image_dir: str) -> list[dict[str, Any]]:
    """Discover all directories containing images (root + subdirectories).

    Scans the root directory and each immediate subdirectory for images.
    Returns a list of subset dicts:
    [
        {"image_dir": "/path/to/root", "num_images": 10, "num_captions": 10},
        {"image_dir": "/path/to/root/outfit1", "num_images": 8, "num_captions": 8},
        ...
    ]
    Only directories with >= 1 image are included.
    """
    root = Path(image_dir).resolve()
    if not root.exists() or not root.is_dir():
        return []

    subsets: list[dict[str, Any]] = []

    # Check root directory
    root_images = _count_images_in_dir(root)
    if root_images > 0:
        subsets.append({
            "image_dir": str(root),
            "num_images": root_images,
            "num_captions": _count_captions_in_dir(root),
        })

    # Check each immediate subdirectory
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            sub_images = _count_images_in_dir(entry)
            if sub_images > 0:
                subsets.append({
                    "image_dir": str(entry),
                    "num_images": sub_images,
                    "num_captions": _count_captions_in_dir(entry),
                })

    return subsets


def generate_dataset_toml(
    image_dir: str | None = None,
    batch_size: int = 4,
    num_images: int = 0,
    epochs: int = 2,
    num_repeats: int = 1,
    output_path: str = "dataset.toml",
    resolution: int = 1024,
    cache_text_encoder_outputs: bool = False,
    caption_tag_dropout_rate: float = 0.1,
    keep_tokens: int = 1,
    subsets: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a .toml dataset config for training.

    Uses the toml library for proper serialization so that paths with
    special characters (quotes, backslashes, spaces) are handled safely.

    Args:
        image_dir: Path to directory containing training images.
                   Used as a single subset when *subsets* is None.
        batch_size: Training batch size.
        num_images: Number of images in the dataset (legacy, ignored when subsets provided).
        epochs: Number of training epochs (passed for API compat, set via CLI).
        num_repeats: How many times each image is repeated per epoch.
        output_path: Path to write the TOML file.
        resolution: Image resolution (default: 1024).
        cache_text_encoder_outputs: If True, disable shuffle_caption,
            token_warmup_step, and caption_tag_dropout_rate (required by
            kohya-ss when --cache_text_encoder_outputs is used).
        caption_tag_dropout_rate: Probability of dropping each caption tag
            (default: 0.05). Overridden to 0.0 when caching is enabled.
        keep_tokens: Number of leading tokens to preserve from caption shuffle.
        subsets: Optional list of subset dicts for multi-folder datasets.
            Each dict: {"image_dir": str, "num_repeats": int}.
            When provided, *image_dir* and *num_repeats* are ignored.

    Returns:
        Path to the generated TOML file.
    """
    # Clamp resolution to Anima-supported bucket range
    min_reso = 768
    max_reso = 1024
    bucket_steps = 16  # Anima: WanVAE spatial downscale=8, patch_size=8 -> 16

    clamped = max(min_reso, min(resolution, max_reso))
    if resolution != clamped:
        import logging
        logging.warning(
            f"Resolution {resolution} clamped to {clamped} "
            f"(Anima supports {min_reso}-{max_reso}px)"
        )

    # When caching text encoder outputs, certain caption randomization
    # options are incompatible (kohya-ss AssertionError)
    if cache_text_encoder_outputs:
        shuffle_caption = False
        token_warmup_step = 0
        effective_dropout = 0.0
    else:
        shuffle_caption = True
        token_warmup_step = 0
        effective_dropout = caption_tag_dropout_rate

    # Build subsets list
    if subsets is not None:
        subsets_list = subsets
    else:
        # Legacy: single folder
        subsets_list = [
            {
                "image_dir": image_dir,
                "num_repeats": num_repeats,
            }
        ]

    config = {
        "general": {
            "shuffle_caption": shuffle_caption,
            "caption_extension": ".txt",
            "keep_tokens": keep_tokens,
            "token_warmup_step": token_warmup_step,
            "caption_tag_dropout_rate": effective_dropout,
        },
        "datasets": [
            {
                "enable_bucket": True,
                "resolution": clamped,
                "min_bucket_reso": min_reso,
                "max_bucket_reso": max_reso,
                "bucket_reso_steps": bucket_steps,
                "bucket_no_upscale": True,
                "batch_size": batch_size,
                "subsets": subsets_list,
            }
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return str(output)
