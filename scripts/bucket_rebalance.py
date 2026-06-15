"""Bucket rebalancing for kohya-ss training datasets.

Detects when a single aspect-ratio bucket dominates the dataset and
redistributes images by cropping to adjacent bucket resolutions.
"""

import logging
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from scripts.constants import IMAGE_EXTENSIONS
from scripts.dataset_toml import discover_subsets

logger = logging.getLogger(__name__)


# ── Image I/O helpers ────────────────────────────────────────────────────
def _imread(path: Path):
    """Read an image with Windows-safe path handling."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite(path: Path, image) -> bool:
    """Write an image with Windows-safe path handling."""
    ext = path.suffix.lower() if path.suffix.lower() in IMAGE_EXTENSIONS else ".png"
    try:
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


# ── Kohya bucket manager ─────────────────────────────────────────────────
def _get_kohya_bucket_manager(resolution, min_bucket_reso, max_bucket_reso, reso_steps):
    """Create a kohya BucketManager with our dataset.toml settings."""
    from scripts.constants import PROJECT_ROOT
    sd_scripts_dir = str(PROJECT_ROOT / "sd-scripts")
    import sys
    if sd_scripts_dir not in sys.path:
        sys.path.insert(0, sd_scripts_dir)
    from library.train_util import BucketManager
    return BucketManager(
        True, (resolution, resolution), min_bucket_reso, max_bucket_reso, reso_steps,
    )


def assign_bucket_resolution(
    width: int, height: int,
    resolution: int = 1024, min_bucket_reso: int = 768,
    max_bucket_reso: int = 1024, reso_steps: int = 16,
) -> tuple[int, int]:
    """Assign bucket resolution using kohya's select_bucket behavior."""
    try:
        manager = _get_kohya_bucket_manager(
            resolution, min_bucket_reso, max_bucket_reso, reso_steps,
        )
        bucket, _, _ = manager.select_bucket(width, height)
        return bucket
    except Exception:
        # Fallback: round to nearest reso_steps
        return (
            max(reso_steps, int(round(width / reso_steps) * reso_steps)),
            max(reso_steps, int(round(height / reso_steps) * reso_steps)),
        )


def plan_bucket_rebalance(
    bucket_counts: dict[tuple[int, int], int],
    dominance_threshold: float,
    max_augmented_images: int,
) -> dict | None:
    """Create a rebalance plan when one bucket dominates too strongly.

    Returns a plan dict with keys: dominant_bucket, dominant_share,
    augment_count, target_buckets. Returns None if no rebalancing needed.
    """
    if len(bucket_counts) < 3:
        return None
    total = sum(bucket_counts.values())
    if total <= 0:
        return None

    dominant_bucket, dominant_count = max(bucket_counts.items(), key=lambda x: x[1])
    dominant_share = dominant_count / total
    if dominant_share <= dominance_threshold:
        return None

    needed = max(0, math.ceil(dominant_count / dominance_threshold - total))
    available = dominant_count
    augment_count = min(available, max_augmented_images, max(1, needed))

    # Target buckets: all non-dominant buckets
    target_buckets = [
        b for b in bucket_counts if b != dominant_bucket
    ]
    # Ensure min coverage: at least one augmented image per target bucket
    if target_buckets:
        augment_count = max(augment_count, len(target_buckets))

    return {
        "dominant_bucket": dominant_bucket,
        "dominant_count": dominant_count,
        "dominant_share": dominant_share,
        "augment_count": augment_count,
        "target_buckets": target_buckets,
    }


def collect_bucket_members(
    training_images, resolution=1024, min_bucket_reso=768,
    max_bucket_reso=1024, reso_steps=16,
):
    """Collect bucket counts and source members from dataset root + subdirs.

    Returns (counts, members, skipped) where:
      - counts: {bucket_resolution: count}
      - members: {bucket_resolution: [image_paths]}
      - skipped: number of unreadable images
    """
    counts: dict[tuple[int, int], int] = {}
    members: dict[tuple[int, int], list[Path]] = {}
    skipped = 0
    manager = None
    try:
        manager = _get_kohya_bucket_manager(
            resolution, min_bucket_reso, max_bucket_reso, reso_steps,
        )
    except Exception:
        pass

    for subset in discover_subsets(training_images):
        subset_dir = Path(subset["image_dir"])
        for entry in subset_dir.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = _imread(entry)
            if image is None:
                skipped += 1
                continue
            h, w = image.shape[:2]
            if manager is not None:
                bucket, _, _ = manager.select_bucket(w, h)
            else:
                # Fallback: round to nearest reso_steps
                bucket = (
                    max(reso_steps, int(round(w / reso_steps) * reso_steps)),
                    max(reso_steps, int(round(h / reso_steps) * reso_steps)),
                )
            counts[bucket] = counts.get(bucket, 0) + 1
            members.setdefault(bucket, []).append(entry)
    return counts, members, skipped


# ── Crop target generation ───────────────────────────────────────────────
def _find_crop_targets(dominant_bucket, dominant_members, all_buckets, reso_steps=16):
    """Generate crop target resolutions that escape the dominant bucket.

    Uses two strategies:
    1. Adjacent resolutions (±reso_steps from dominant bucket)
    2. Existing minority bucket resolutions (fallback when strategy 1 fails)

    A valid crop target must not be assigned back to the dominant bucket
    by kohya's bucket_no_upscale logic.
    """
    dw, dh = dominant_bucket
    avg_w, avg_h = dw, dh
    if dominant_members:
        sizes = []
        for path in dominant_members[:20]:
            img = _imread(path)
            if img is not None:
                sizes.append((img.shape[1], img.shape[0]))
        if sizes:
            avg_w = int(sum(s[0] for s in sizes) / len(sizes))
            avg_h = int(sum(s[1] for s in sizes) / len(sizes))

    def _kohya_assigns_to(target):
        """Find which existing bucket kohya would assign an image to."""
        tw, th = target
        fitting = [b for b in all_buckets if b[0] >= tw and b[1] >= th]
        if not fitting:
            return None
        return min(fitting, key=lambda b: b[0] * b[1])

    def _is_valid(target):
        """Check if target escapes dominant bucket.

        _random_crop_to_bucket crops to aspect ratio then resizes to exact
        target, so the target doesn't need to fit within the source image.
        We only need to ensure the cropped+resized image won't be assigned
        back to the dominant bucket.
        """
        if target[0] < 256 or target[1] < 256:
            return False
        assigned = _kohya_assigns_to(target)
        return assigned is None or assigned != dominant_bucket

    # Strategy 1: adjacent resolutions (±reso_steps)
    candidates = []
    for delta_w in (-reso_steps, reso_steps):
        for delta_h in (-reso_steps, 0, reso_steps):
            if delta_w == 0 and delta_h == 0:
                continue
            tw, th = dw + delta_w, dh + delta_h
            if tw >= 256 and th >= 256:
                candidates.append((tw, th))

    valid = [t for t in candidates if _is_valid(t)]
    if valid:
        return valid

    # Strategy 2: use existing minority bucket resolutions directly
    minority_buckets = [b for b in all_buckets if b != dominant_bucket]
    return [b for b in minority_buckets if _is_valid(b)]


def _random_crop_to_bucket(image, target_bucket, rng):
    """Crop then resize to exact target bucket resolution."""
    target_w, target_h = target_bucket
    target_ratio = target_w / target_h
    h, w = image.shape[:2]
    source_ratio = w / h

    if source_ratio > target_ratio:
        crop_h, crop_w = h, max(32, int(round(h * target_ratio)))
    else:
        crop_w, crop_h = w, max(32, int(round(w / target_ratio)))

    crop_w, crop_h = min(crop_w, w), min(crop_h, h)
    if crop_w < 32 or crop_h < 32:
        return None

    x = rng.randint(0, w - crop_w) if w > crop_w else 0
    y = rng.randint(0, h - crop_h) if h > crop_h else 0
    cropped = image[y : y + crop_h, x : x + crop_w]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


# ── Main rebalance entry point ───────────────────────────────────────────
def maybe_build_bucket_rebalance_subset(
    training_images, output_dir, num_repeats, enabled,
    dominance_threshold, max_augmented_images, seed, resolution=1024,
):
    """Redistribute excess images from a dominant bucket.

    When a single bucket holds more than dominance_threshold of all images:
    1. Copy dominant images to a subset with LOWER repeats
    2. Copy non-dominant images to a subset with NORMAL repeats
    3. Create cropped variants and place in a third subset

    Returns list of subset dicts (ready for dataset TOML), or None.
    """
    if not enabled:
        return None

    bucket_counts, bucket_members, skipped = collect_bucket_members(
        training_images, resolution=resolution,
    )
    if skipped > 0:
        logger.debug("Bucket check skipped %d unreadable image(s)", skipped)

    if len(bucket_counts) < 3:
        return None
    total = sum(bucket_counts.values())
    if total <= 0:
        return None

    dominant_bucket, dominant_count = max(bucket_counts.items(), key=lambda x: x[1])
    dominant_share = dominant_count / total
    if dominant_share <= dominance_threshold:
        logger.debug("Bucket distribution within threshold — no rebalance needed")
        return None

    needed = max(0, math.ceil(dominant_count / dominance_threshold - total))
    images_to_move = min(dominant_count, max_augmented_images, max(1, needed))
    if images_to_move <= 0:
        return None

    dominant_members = bucket_members.get(dominant_bucket, [])
    if not dominant_members:
        return None

    crop_targets = _find_crop_targets(
        dominant_bucket, dominant_members, set(bucket_counts.keys()),
    )
    if not crop_targets:
        logger.debug("Bucket rebalance: no valid crop targets")
        return None

    # Calculate per-subset repeats
    non_dominant_count = total - dominant_count
    crops_to_create = min(images_to_move, len(dominant_members))
    r_nd = num_repeats
    r_c = num_repeats
    max_r_d = (
        (non_dominant_count * r_nd + crops_to_create * r_c) * dominance_threshold
        / (dominant_count * (1 - dominance_threshold))
    )
    r_d = max(1, int(max_r_d))

    dom_eff = dominant_count * r_d
    non_dom_eff = non_dominant_count * r_nd
    crop_eff = crops_to_create * r_c
    total_eff = dom_eff + non_dom_eff + crop_eff
    actual_share = dom_eff / total_eff * 100 if total_eff > 0 else 0

    logger.warning(
        "Bucket rebalance: dominant %s at %.0f%% — "
        "splitting into 3 subsets (dominant r=%d, others r=%d, %d crops)",
        dominant_bucket, dominant_share * 100,
        r_d, r_nd, crops_to_create,
    )

    # Create subset directories
    rebalance_base = output_dir / "bucket-rebalance"
    if rebalance_base.exists():
        shutil.rmtree(rebalance_base)
    dominant_dir = rebalance_base / "dominant"
    non_dominant_dir = rebalance_base / "non_dominant"
    crops_dir = rebalance_base / "crops"
    for d in (dominant_dir, non_dominant_dir, crops_dir):
        d.mkdir(parents=True)

    rng = random.Random(seed)
    sources = rng.sample(dominant_members, crops_to_create)

    # Copy dominant images
    for src in dominant_members:
        dst = dominant_dir / src.name
        shutil.copy2(str(src), str(dst))
        cap = src.with_suffix(".txt")
        if cap.exists():
            shutil.copy2(str(cap), str(dominant_dir / cap.name))

    # Copy non-dominant images
    for bucket, bmembers in bucket_members.items():
        if bucket == dominant_bucket:
            continue
        for src in bmembers:
            dst = non_dominant_dir / src.name
            shutil.copy2(str(src), str(dst))
            cap = src.with_suffix(".txt")
            if cap.exists():
                shutil.copy2(str(cap), str(non_dominant_dir / cap.name))

    # Create cropped images
    generated = 0
    for i, src in enumerate(sources):
        target = crop_targets[i % len(crop_targets)]
        image = _imread(src)
        if image is None:
            continue
        cropped = _random_crop_to_bucket(image, target, rng)
        if cropped is None:
            continue
        out_suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTENSIONS else ".png"
        out_path = crops_dir / f"crop_{i:04d}_{src.stem}{out_suffix}"
        if not _imwrite(out_path, cropped):
            continue
        cap = src.with_suffix(".txt")
        if cap.exists():
            out_path.with_suffix(".txt").write_text(
                cap.read_text(encoding="utf-8", errors="replace"), encoding="utf-8",
            )
        generated += 1

    if generated == 0:
        logger.debug("Bucket rebalance: no cropped samples generated")
        shutil.rmtree(rebalance_base, ignore_errors=True)
        return None



    return [
        {"image_dir": str(non_dominant_dir), "num_repeats": r_nd},
        {"image_dir": str(dominant_dir), "num_repeats": r_d},
        {"image_dir": str(crops_dir), "num_repeats": r_c},
    ]
