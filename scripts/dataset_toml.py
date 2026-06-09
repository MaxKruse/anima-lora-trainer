"""Dataset TOML config generator for kohya-ss training scripts."""

import toml
from pathlib import Path


def generate_dataset_toml(
    image_dir: str,
    batch_size: int,
    num_images: int,
    epochs: int,
    num_repeats: int,
    output_path: str,
    resolution: int = 1024,
) -> str:
    """Generate a .toml dataset config for training.

    Uses the toml library for proper serialization so that paths with
    special characters (quotes, backslashes, spaces) are handled safely.

    Args:
        image_dir: Path to directory containing training images.
        batch_size: Training batch size.
        num_images: Number of images in the dataset.
        epochs: Number of training epochs (passed for API compat, set via CLI).
        num_repeats: How many times each image is repeated per epoch.
        output_path: Path to write the TOML file.
        resolution: Image resolution (default: 1024).

    Returns:
        Path to the generated TOML file.
    """
    # Clamp resolution to Anima-supported bucket range
    min_reso = 768
    max_reso = 1024
    bucket_steps = 16  # Anima: WanVAE spatial downscale=8, patch_size=2 -> 16

    clamped = max(min_reso, min(resolution, max_reso))

    config = {
        "general": {
            "shuffle_caption": True,
            "caption_extension": ".txt",
            "keep_tokens": 1,
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
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "num_repeats": num_repeats,
                    }
                ],
            }
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return str(output)
