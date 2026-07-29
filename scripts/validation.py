"""Dataset validation for LoRA training.

Checks folder structure, image counts, caption coverage, and writes validation markers.
Converts non-JPG images to JPG during validation.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.constants import (
    IMAGE_EXTENSIONS,
    MAX_IMAGES_BASE,
    MAX_IMAGES_PER_FOLDER,
    MIN_IMAGES_PER_FOLDER,
    MIN_IMAGES_PER_OUTFIT,
    VALIDATION_MARKER,
)
from scripts.dataset_toml import discover_subsets

# Extensions that should be converted to .jpg during validation
_CONVERTIBLE_EXTENSIONS = {".png", ".webp", ".bmp", ".tiff", ".tif", ".jfif", ".jif"}

# Extensions that are already jpg and don't need conversion
_JPG_EXTENSIONS = {".jpg", ".jpeg"}


def _convert_to_jpg(img_path: Path) -> bool:
    """Convert a single image file to JPG in place.

    Renames the file to .jpg (compositing alpha onto white if needed).
    Returns True if conversion succeeded, False otherwise.
    """
    try:
        with Image.open(img_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                # Composite onto white background for formats with alpha
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if "A" in img.mode:
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            jpg_path = img_path.with_suffix(".jpg")
            img.save(jpg_path, "JPEG", quality=95)
            img_path.unlink()
            return True
    except Exception as e:
        print(f"  WARNING: Failed to convert {img_path.name}: {e}", file=sys.stderr)
        return False


def convert_images_to_jpg(img_dir: Path) -> int:
    """Convert all non-JPG images in img_dir to JPG.

    Scans the directory and any immediate subdirectories.
    Also renames matching .txt caption files if the image filename changes.
    Returns the number of files successfully converted.
    """
    converted = 0
    dirs_to_scan = [img_dir]

    # Also scan immediate subdirectories
    if img_dir.is_dir():
        for entry in sorted(img_dir.iterdir()):
            if entry.is_dir():
                dirs_to_scan.append(entry)

    for directory in dirs_to_scan:
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in _CONVERTIBLE_EXTENSIONS:
                continue

            # If a caption file exists for the old name, rename it for the new name
            old_caption = entry.with_suffix(".txt")
            new_caption = entry.with_suffix(".jpg").with_suffix(".txt")

            success = _convert_to_jpg(entry)
            if success:
                if old_caption.exists() and not new_caption.exists():
                    old_caption.rename(new_caption)
                converted += 1
                print(f"  Converted {entry.name} -> {entry.with_suffix('.jpg').name}")

    return converted


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def get_dataset_img_dir(dataset_dir: str) -> Path:
    """Get the dataset's image directory (<dataset>/img/)."""
    return Path(dataset_dir).resolve() / "img"


def get_dataset_out_dir(dataset_dir: str) -> Path:
    """Get the dataset's output directory (<dataset>/out/)."""
    return Path(dataset_dir).resolve() / "out"


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
        expected_img = str(get_dataset_img_dir(dataset_dir))
        if data.get("dataset") != expected_img:
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
        "dataset": str(get_dataset_img_dir(dataset_dir)),
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
            "max_steps": params.get("max_steps", 600),
            "batch_size": params.get("batch_size", 4),
        },
        "warnings": warnings,
    }

    _write_json(marker_path, marker_data)
    return marker_path


def calculate_max_steps(batch_size: int = 4, training_type: str = "character") -> int:
    """Calculate auto max_steps based on batch size and training type.

    Character training (bs=4): 600 steps
    Style training (bs=4): 1200 steps (2x character)

    Scaling with batch size (slightly less than linear):
      Character: bs4=600, bs3=800, bs2=1000, bs1=1600
      Style:     bs4=1200, bs3=1600, bs2=2000, bs1=3200
    """
    character_steps = {
        4: 600,
        3: 800,
        2: 1000,
        1: 1600,
    }.get(batch_size, 600)

    multiplier = 2 if training_type == "style" else 1
    return character_steps * multiplier


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
    dataset_dir: str,
    batch_size: int = 4,
) -> tuple[bool, list[str]]:
    """Validate a training dataset.

    Expects the dataset directory to contain:
      <dataset>/img/   - training images and .txt captions
      <dataset>/out/   - output directory (created if missing)

    Returns (is_valid, warnings) tuple. is_valid is False for hard errors.
    Warnings are collected for non-fatal issues.
    """
    path = Path(dataset_dir).resolve()
    warnings: list[str] = []

    if not path.exists():
        return False, [f"Directory does not exist: {path}"]
    if not path.is_dir():
        return False, [f"Not a directory: {path}"]

    # ── Check folder structure ─────────────────────────────────────────
    img_dir = path / "img"
    out_dir = path / "out"

    if not img_dir.exists():
        print(f"\n  ERROR: Missing 'img/' subdirectory in dataset folder.", file=sys.stderr)
        print(f"\n  Expected structure:", file=sys.stderr)
        print(f"    {path}/", file=sys.stderr)
        print(f"      img/     <- training images + .txt captions", file=sys.stderr)
        print(f"      out/     <- training output (created automatically)", file=sys.stderr)
        return False, ["Missing img/ subdirectory"]

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created output directory: {out_dir}")

    # ── Convert non-JPG images to JPG ──────────────────────────────────
    converted_count = convert_images_to_jpg(img_dir)
    if converted_count > 0:
        print(f"  Converted {converted_count} image(s) to JPG\n")

    # ── Validate images in img/ ────────────────────────────────────────
    subsets = discover_subsets(str(img_dir))

    if not subsets:
        return False, ["No image files found in img/"]

    total_images = sum(s["num_images"] for s in subsets)
    total_captions = sum(s.get("num_captions", 0) for s in subsets)

    # ── Per-folder checks ──────────────────────────────────────────────
    num_subsets = len(subsets)
    is_base = True

    for s in subsets:
        folder = Path(s["image_dir"])
        rel = (
            folder.relative_to(img_dir)
            if str(folder).startswith(str(img_dir))
            else folder.name
        )
        rel_str = str(rel) if str(rel) != "." else ""
        img_count = s["num_images"]
        cap_count = s.get("num_captions", 0)
        cap_status = (
            "\u2713" if cap_count >= img_count
            else f"\u26a0 {cap_count}/{img_count} captions"
        )
        folder_label = f"img/{rel_str}/" if rel_str else "img/"
        print(f"  {folder_label}  \u2014  {img_count} images, {cap_status}")

        max_allowed = MAX_IMAGES_BASE if is_base else MIN_IMAGES_PER_OUTFIT

        if img_count < MIN_IMAGES_PER_FOLDER:
            msg = (
                f"{folder_label} has only {img_count} images "
                f"(recommended min: {MIN_IMAGES_PER_FOLDER})"
            )
            warnings.append(msg)
        elif img_count > max_allowed:
            msg = (
                f"{folder_label} has {img_count} images "
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
            "No caption files found in img/ \u2014 every image needs a .txt caption"
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
        dataset_dir, subsets, params, warnings
    )
    print(f"\n  \u2713 Validated -> {marker_path}")

    return True, warnings
