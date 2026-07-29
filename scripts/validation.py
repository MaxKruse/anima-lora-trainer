"""Dataset validation for LoRA training.

Checks image counts, caption coverage, and writes validation markers.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from scripts.constants import (
    IMAGE_EXTENSIONS,
    MAX_IMAGES_BASE,
    MAX_IMAGES_PER_FOLDER,
    MIN_IMAGES_PER_FOLDER,
    MIN_IMAGES_PER_OUTFIT,
    VALIDATION_MARKER,
)
from scripts.dataset_toml import discover_subsets


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def get_dataset_out_dir(dataset_dir: str) -> Path:
    """Get the dataset's output directory (datasets/<name>/out/)."""
    dataset_dir_path = Path(dataset_dir).resolve()
    parent = dataset_dir_path.parent
    if parent.name == "img":
        parent = parent.parent
    return parent / "out"


def get_validation_marker_path(dataset_dir: str) -> Path:
    """Get the path to the validation marker file."""
    return get_dataset_out_dir(dataset_dir) / VALIDATION_MARKER


def check_validation(dataset_dir: str) -> bool:
    """Check if the dataset has been validated. Returns True if valid."""
    marker = get_validation_marker_path(dataset_dir)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
        if data.get("dataset") != str(Path(dataset_dir).resolve()):
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def write_validation_marker(
    dataset_dir: str,
    subsets: list[dict[str, Any]],
    params: dict[str, Any],
    warnings: list[str],
) -> Path:
    """Write the validation marker file after successful validation."""
    out_dir = get_dataset_out_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_path = out_dir / VALIDATION_MARKER

    total_images = sum(s["num_images"] for s in subsets)
    total_captions = sum(s.get("num_captions", 0) for s in subsets)

    marker_data: dict[str, Any] = {
        "dataset": str(Path(dataset_dir).resolve()),
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total_images,
        "total_captions": total_captions,
        "num_subsets": len(subsets),
        "subsets": [
            {
                "folder": Path(s["image_dir"]).name,
                "images": s["num_images"],
                "captions": s.get("num_captions", 0),
            }
            for s in subsets
        ],
        "params": {
            "max_steps": params.get("max_steps", 800),
            "batch_size": params.get("batch_size", 4),
        },
        "warnings": warnings,
    }

    _write_json(marker_path, marker_data)
    return marker_path


def calculate_max_steps(batch_size: int = 4) -> int:
    """Calculate auto max_steps based on batch size.

    Higher batch sizes converge faster, so they need fewer total steps.
    Scaling is slightly less than linear to keep theoretical training
    stable across batch sizes.

      batch_size=4 -> max_steps=600  (default, sweet spot)
      batch_size=3 -> max_steps=800
      batch_size=2 -> max_steps=1000
      batch_size=1 -> max_steps=1600
    """
    return {
        4: 600,
        3: 800,
        2: 1000,
        1: 1600,
    }.get(batch_size, 600)


def calculate_repeats(
    num_images: int,
    batch_size: int = 4,
    target_steps_per_epoch: int = 12,
) -> int:
    """Calculate num_repeats so that (num_images * repeats) / batch_size falls in 10-15 range.

    The target is ~12 steps per epoch (midpoint of 10-15).

    Derivation:
      steps_per_epoch = (num_images * num_repeats) / batch_size
      num_repeats = (target_steps_per_epoch * batch_size) / num_images
    """
    if num_images <= 0 or batch_size <= 0:
        return 1
    return max(1, math.ceil((target_steps_per_epoch * batch_size) / num_images))


def validate_dataset(
    image_dir: str,
    batch_size: int = 4,
) -> tuple[bool, list[str]]:
    """Validate a training dataset.

    Returns (is_valid, warnings) tuple. is_valid is False for hard errors
    (missing directory, no images, no captions). Warnings are collected
    for non-fatal issues.

    Rules:
      - Each folder: 10-25 images (warnings outside this range)
      - Total: 25 (base) + 15 x N_outfits is the max recommended
      - NO captions at all = hard error
      - Some missing captions = warning
    """
    path = Path(image_dir)
    warnings: list[str] = []

    if not path.exists():
        return False, ["Directory does not exist"]
    if not path.is_dir():
        return False, ["Not a directory"]

    subsets = discover_subsets(image_dir)

    if not subsets:
        return False, ["No image files found"]

    total_images = sum(s["num_images"] for s in subsets)
    total_captions = sum(s.get("num_captions", 0) for s in subsets)

    # ── Per-folder checks ──────────────────────────────────────────────
    num_subsets = len(subsets)
    is_base = True

    for s in subsets:
        folder = Path(s["image_dir"])
        rel = (
            folder.relative_to(path)
            if str(folder).startswith(str(path))
            else folder.name
        )
        img_count = s["num_images"]
        cap_count = s.get("num_captions", 0)
        cap_status = (
            "\u2713" if cap_count >= img_count
            else f"\u26a0 {cap_count}/{img_count} captions"
        )
        print(f"  {rel}/  \u2014  {img_count} images, {cap_status}")

        max_allowed = MAX_IMAGES_BASE if is_base else MIN_IMAGES_PER_OUTFIT

        if img_count < MIN_IMAGES_PER_FOLDER:
            msg = (
                f"{rel}/ has only {img_count} images "
                f"(recommended min: {MIN_IMAGES_PER_FOLDER})"
            )
            warnings.append(msg)
        elif img_count > max_allowed:
            msg = (
                f"{rel}/ has {img_count} images "
                f"(recommended max: {max_allowed})"
            )
            warnings.append(msg)

        is_base = False

    print(f"  Total: {total_images} images, {total_captions} captions")

    # ── Total image count check ────────────────────────────────────────
    num_outfits = max(0, num_subsets - 1)
    max_recommended = MAX_IMAGES_BASE + (MIN_IMAGES_PER_OUTFIT * num_outfits)

    if total_images > max_recommended:
        msg = (
            f"Total images ({total_images}) exceeds recommended max "
            f"({max_recommended} = {MAX_IMAGES_BASE} base + "
            f"{MIN_IMAGES_PER_OUTFIT} x {num_outfits} outfits)"
        )
        print(f"  WARNING: {msg}", file=sys.stderr)
        warnings.append(msg)

    # ── Caption checks ─────────────────────────────────────────────────
    if total_captions == 0:
        return False, [
            "No caption files found — every image needs a .txt caption"
        ]
    elif total_captions < total_images:
        missing = total_images - total_captions
        msg = (
            f"{missing} image(s) may be missing captions "
            f"({total_captions}/{total_images} have .txt files)"
        )
        warnings.append(msg)

    # Write validation marker
    auto_max_steps = calculate_max_steps(batch_size)
    params: dict[str, Any] = {"max_steps": auto_max_steps, "batch_size": batch_size}
    marker_path = write_validation_marker(
        image_dir, subsets, params, warnings
    )
    print("\n  \u2713 Validated")

    return True, warnings
