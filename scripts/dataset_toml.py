"""Dataset TOML config generator for kohya-ss training scripts."""

import math
from pathlib import Path


def generate_dataset_toml(
    image_dir: str,
    batch_size: int,
    num_images: int,
    epochs: int,
    steps_per_epoch: int,
    output_path: str,
    resolution: int = 512,
) -> str:
    """Generate a .toml dataset config for training.

    Args:
        image_dir: Path to directory containing training images.
        batch_size: Training batch size.
        num_images: Number of images in the dataset.
        epochs: Number of training epochs.
        steps_per_epoch: Desired training steps per epoch.
        output_path: Path to write the TOML file.
        resolution: Image resolution (default: 512).

    Returns:
        Path to the generated TOML file.
    """
    # Calculate num_repeats: ceil(steps_per_epoch / num_images)
    num_repeats = math.ceil(steps_per_epoch / num_images) if num_images > 0 else 1

    toml_content = f"""\
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_dir}'
  num_repeats = {num_repeats}
"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_content)

    return str(output)
