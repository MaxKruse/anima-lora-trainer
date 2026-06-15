"""Unified LoRA training CLI — single and matrix modes.

Wraps kohya-ss training scripts with sensible defaults derived from
proven training runs (froot: 44 images, mari_setogaya: 14 images).

Usage:
    # Validate a dataset
    uv run python scripts/train.py --validate --dataset datasets/froot/img

    # Single training run
    uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima

    # Single run with custom params
    uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot --lr 0.0001 --bs 2

    # Matrix run (comma-separated values generate all permutations)
    uv run python scripts/train.py --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,32 --alpha 1,16 --lr 0.0001,0.0002
"""

import argparse
import shutil
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from itertools import product

import cv2
import numpy as np

# ── Project setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.dataset_toml import generate_dataset_toml, discover_subsets as _discover_subsets
from scripts.zip_training_data import zip_training_data
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants / Defaults ─────────────────────────────────────────────────
# Derived from proven training runs (froot 44 imgs, mari_setogaya 14 imgs)
DEFAULTS = {
    "network_dim": 20,
    "network_alpha": 1,
    "learning_rate": 0.0002,
    "batch_size": 4,
    "max_steps": 800,
    "optimizer": "AdamW8Bit",
    "scheduler": "cosine",
    "resolution": 1024,
    "mixed_precision": "bf16",
    "timestep_sampling": "sigmoid",
    "gradient_checkpointing": True,
    "cache_latents": True,
    "cache_text_encoder": False,
    "caption_tag_dropout_rate": 0.05,
    "keep_tokens": 1,
}

# Image count guidelines
MIN_IMAGES_PER_FOLDER = 10
MAX_IMAGES_PER_FOLDER = 25
MIN_IMAGES_PER_OUTFIT = 15  # max per outfit folder (non-base)
MAX_IMAGES_BASE = 25        # max for base folder

# Validation marker filename
VALIDATION_MARKER = ".validation.json"

# Model paths
MODEL_PATHS = {
    "diffusion_model": "models/diffusion_model/anima-base-v1.0.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
    "text_encoder": "models/text_encoder/qwen_3_06b_base.safetensors",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ── Helpers ──────────────────────────────────────────────────────────────
def count_images(image_dir: str) -> int:
    """Count ALL image files recursively under a directory."""
    root = Path(image_dir)
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*") if _.is_file() and _.suffix.lower() in IMAGE_EXTENSIONS)


def count_captions(image_dir: str) -> int:
    """Count ALL .txt caption files recursively under a directory."""
    root = Path(image_dir)
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.txt") if _.is_file())


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


def _get_kohya_bucket_manager(resolution: int, min_bucket_reso: int, max_bucket_reso: int, reso_steps: int):
    """Create a kohya BucketManager configured like our dataset.toml settings."""
    sd_scripts_dir = str(_project_root / "sd-scripts")
    if sd_scripts_dir not in sys.path:
        sys.path.insert(0, sd_scripts_dir)

    from library.train_util import BucketManager

    manager = BucketManager(
        True,  # bucket_no_upscale
        (resolution, resolution),
        min_bucket_reso,
        max_bucket_reso,
        reso_steps,
    )
    return manager


def assign_bucket_resolution(
    width: int,
    height: int,
    resolution: int = 1024,
    min_bucket_reso: int = 768,
    max_bucket_reso: int = 1024,
    reso_steps: int = 16,
) -> tuple[int, int]:
    """Assign bucket resolution using kohya's select_bucket behavior."""
    try:
        manager = _get_kohya_bucket_manager(
            resolution=resolution,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            reso_steps=reso_steps,
        )
        bucket_reso, _resized_size, _ar_error = manager.select_bucket(width, height)
        return bucket_reso
    except Exception:
        # Fallback keeps behavior deterministic if kohya import fails.
        q_w = max(reso_steps, int(round(width / reso_steps) * reso_steps))
        q_h = max(reso_steps, int(round(height / reso_steps) * reso_steps))
        return q_w, q_h


def collect_bucket_members(
    training_images: str,
    resolution: int = 1024,
    min_bucket_reso: int = 768,
    max_bucket_reso: int = 1024,
    reso_steps: int = 16,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], list[Path]], int]:
    """Collect bucket counts and source members from dataset root + immediate subdirs."""
    counts: dict[tuple[int, int], int] = {}
    members: dict[tuple[int, int], list[Path]] = {}
    skipped = 0

    manager = None
    try:
        manager = _get_kohya_bucket_manager(
            resolution=resolution,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            reso_steps=reso_steps,
        )
    except Exception:
        manager = None

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
                bucket, _resized_size, _ar_error = manager.select_bucket(w, h)
            else:
                bucket = assign_bucket_resolution(
                    w,
                    h,
                    resolution=resolution,
                    min_bucket_reso=min_bucket_reso,
                    max_bucket_reso=max_bucket_reso,
                    reso_steps=reso_steps,
                )
            counts[bucket] = counts.get(bucket, 0) + 1
            members.setdefault(bucket, []).append(entry)

    return counts, members, skipped


def _find_crop_targets(
    dominant_bucket: tuple[int, int],
    dominant_members: list[Path],
    all_buckets: set[tuple[int, int]],
    reso_steps: int = 16,
) -> list[tuple[int, int]]:
    """Generate crop target resolutions that escape the dominant bucket.

    A valid crop target must be larger than the dominant bucket in at least
    one dimension — otherwise kohya (with bucket_no_upscale) would assign
    the cropped image back to the dominant bucket.

    Targets don't need to be existing buckets — kohya will assign cropped images
    to the nearest fitting bucket. This spreads redistributed images across
    multiple nearby buckets instead of piling them into one.

    Targets are filtered to ensure they fit within the average source image size
    (since bucket_no_upscale means images are typically larger than bucket reso).
    """
    dw, dh = dominant_bucket

    # Estimate average source image size from dominant members.
    avg_w, avg_h = dw, dh  # fallback to bucket resolution
    if dominant_members:
        sizes = []
        for path in dominant_members[:20]:
            img = _imread(path)
            if img is not None:
                sizes.append((img.shape[1], img.shape[0]))
        if sizes:
            avg_w = int(sum(s[0] for s in sizes) / len(sizes))
            avg_h = int(sum(s[1] for s in sizes) / len(sizes))

    # Generate candidates: ±reso_steps in each dimension.
    candidates = []
    for delta_w in (-reso_steps, reso_steps):
        for delta_h in (-reso_steps, 0, reso_steps):
            if delta_w == 0 and delta_h == 0:
                continue
            tw, th = dw + delta_w, dh + delta_h
            if tw >= 256 and th >= 256:
                candidates.append((tw, th))

    def _kohya_assigns_to(target: tuple[int, int]) -> tuple[int, int] | None:
        """Find which existing bucket kohya would assign an image of given size to."""
        tw, th = target
        fitting = [
            b for b in all_buckets
            if b[0] >= tw and b[1] >= th
        ]
        if not fitting:
            return None  # would create new bucket
        # Kohya picks smallest fitting bucket (by area)
        return min(fitting, key=lambda b: b[0] * b[1])

    # Filter: only keep targets that escape the dominant bucket.
    valid = []
    for target in candidates:
        # Must fit within source image
        if target[0] > avg_w or target[1] > avg_h:
            continue
        # Must not be assigned back to dominant bucket
        assigned = _kohya_assigns_to(target)
        if assigned is None:
            # No existing bucket fits — kohya will create one. Safe.
            valid.append(target)
        elif assigned != dominant_bucket:
            # Assigned to a different bucket. Good.
            valid.append(target)
        # else: would go back to dominant bucket — skip

    return valid


def plan_bucket_rebalance(
    bucket_counts: dict[tuple[int, int], int],
    dominance_threshold: float,
    max_augmented_images: int,
) -> dict | None:
    """Create a rebalance plan when one bucket dominates too strongly.

    Strategy: crop excess images from the dominant bucket to adjacent bucket
    resolutions (±16px in one dimension) and place them in a rebalance subset.
    This directly reduces the dominant bucket's share of the training data.

    Returns a plan dict with images_to_move and target buckets, or None if
    no rebalancing is needed.
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

    # Calculate source images to move.
    # Each moved image becomes a crop placed in an adjacent bucket.
    # With repeats=r: new_total = total + moved*r, dominant_share = dominant / new_total
    # Solve: dominant / (total + moved*r) <= threshold
    #   => moved >= (dominant / threshold - total) / r
    # We don't know repeats yet, so estimate with repeats=1 (conservative upper bound).
    needed = max(0, math.ceil(dominant_count / dominance_threshold - total))
    available = dominant_count  # can't move more source images than exist
    images_to_move = min(available, max_augmented_images, max(1, needed))

    if images_to_move <= 0:
        return None

    return {
        "dominant_bucket": dominant_bucket,
        "dominant_count": dominant_count,
        "dominant_share": dominant_share,
        "images_to_move": images_to_move,
    }


def _random_crop_to_bucket(
    image,
    target_bucket: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    """Crop then resize to exact target bucket resolution for deterministic assignment.

    The target_bucket is always a multiple-of-16 resolution (kohya bucket key),
    so the augmented image will be assigned to the correct bucket by kohya.
    """
    target_w, target_h = target_bucket
    target_ratio = target_w / target_h

    h, w = image.shape[:2]
    source_ratio = w / h

    if source_ratio > target_ratio:
        crop_h = h
        crop_w = max(32, int(round(h * target_ratio)))
    else:
        crop_w = w
        crop_h = max(32, int(round(w / target_ratio)))

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    if crop_w < 32 or crop_h < 32:
        return None

    max_x = w - crop_w
    max_y = h - crop_h
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0

    cropped = image[y : y + crop_h, x : x + crop_w]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def maybe_build_bucket_rebalance_subset(
    training_images: str,
    output_dir: Path,
    num_repeats: int,
    enabled: bool,
    dominance_threshold: float,
    max_augmented_images: int,
    seed: int,
    resolution: int = 1024,
) -> list[dict] | None:
    """Redistribute excess images from a dominant bucket using per-subset repeats.

    When a single bucket holds more than dominance_threshold of all images:
    1. Copy dominant images to a dedicated subset with LOWER repeats
    2. Copy non-dominant images to a dedicated subset with NORMAL repeats
    3. Create cropped variants (±16px) and place in a third subset

    The caller should REPLACE the original dataset subsets with these configs
    to avoid double-counting.

    Returns a list of subset dicts (ready for dataset TOML), or None if
    no rebalancing is needed.
    """
    if not enabled:
        return None

    bucket_counts, bucket_members, skipped = collect_bucket_members(
        training_images,
        resolution=resolution,
        min_bucket_reso=768,
        max_bucket_reso=resolution,
        reso_steps=16,
    )
    if skipped > 0:
        logger.warning(f"Bucket check skipped {skipped} unreadable image(s)")

    plan = plan_bucket_rebalance(bucket_counts, dominance_threshold, max_augmented_images)
    if plan is None:
        logger.info("Bucket rebalance check: distribution within threshold")
        return None

    dominant_bucket = plan["dominant_bucket"]
    dominant_members = bucket_members.get(dominant_bucket, [])
    images_to_move = plan["images_to_move"]

    if not dominant_members:
        return None

    # Find crop targets that escape the dominant bucket.
    all_buckets = set(bucket_counts.keys())
    crop_targets = _find_crop_targets(dominant_bucket, dominant_members, all_buckets)
    if not crop_targets:
        logger.info(
            "Bucket rebalance: no valid crop targets for %s — skipping",
            dominant_bucket,
        )
        return None

    # Calculate per-subset repeats to bring dominant share below threshold.
    # dominant_eff = dom_count * r_d
    # non_dom_eff = non_dom_count * r_nd
    # crop_eff = crops * r_c
    # Want: dominant_eff / (dominant_eff + non_dom_eff + crop_eff) <= threshold
    non_dominant_count = sum(bucket_counts.values()) - plan["dominant_count"]
    crops_to_create = min(images_to_move, len(dominant_members))

    # r_nd = num_repeats, r_c = num_repeats, solve for r_d:
    #   dom * r_d <= threshold * (dom * r_d + non_dom * r_nd + crops * r_c)
    #   r_d <= (non_dom * r_nd + crops * r_c) * threshold / (dom * (1 - threshold))
    r_nd = num_repeats
    r_c = num_repeats
    max_r_d = (
        (non_dominant_count * r_nd + crops_to_create * r_c) * dominance_threshold
        / (plan["dominant_count"] * (1 - dominance_threshold))
    )
    r_d = max(1, int(max_r_d))  # floor to ensure we're below threshold

    # Recalculate actual share with floored r_d
    dom_eff = plan["dominant_count"] * r_d
    non_dom_eff = non_dominant_count * r_nd
    crop_eff = crops_to_create * r_c
    total_eff = dom_eff + non_dom_eff + crop_eff
    actual_share = dom_eff / total_eff * 100 if total_eff > 0 else 0

    logger.warning(
        "Bucket rebalance: dominant bucket %s at %.1f%% (%d source image(s)); "
        "splitting into per-subset repeats — dominant r=%d, others r=%d, crops r=%d "
        "(%d crops to %s) — new share ~%.1f%%",
        dominant_bucket,
        plan["dominant_share"] * 100,
        plan["dominant_count"],
        r_d,
        r_nd,
        r_c,
        crops_to_create,
        crop_targets,
        actual_share,
    )

    # ── Create subset directories ──────────────────────────────────────
    rebalance_base = output_dir / "bucket-rebalance"
    if rebalance_base.exists():
        shutil.rmtree(rebalance_base)

    dominant_dir = rebalance_base / "dominant"
    non_dominant_dir = rebalance_base / "non_dominant"
    crops_dir = rebalance_base / "crops"
    dominant_dir.mkdir(parents=True)
    non_dominant_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)

    dominant_member_set = set(dominant_members)
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
    for bucket, members in bucket_members.items():
        if bucket == dominant_bucket:
            continue
        for src in members:
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
                cap.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
        generated += 1

    if generated == 0:
        logger.warning("Bucket rebalance could not generate any cropped samples")
        shutil.rmtree(rebalance_base, ignore_errors=True)
        return None

    logger.info(
        "Bucket rebalance: dominant r=%d (%s), non-dominant r=%d (%s), crops r=%d (%s)",
        r_d, dominant_dir,
        r_nd, non_dominant_dir,
        r_c, crops_dir,
    )

    # Return subset configs that REPLACE the original dataset subsets.
    subsets = [
        {
            "image_dir": str(non_dominant_dir),
            "num_repeats": r_nd,
        },
        {
            "image_dir": str(dominant_dir),
            "num_repeats": r_d,
        },
        {
            "image_dir": str(crops_dir),
            "num_repeats": r_c,
        },
    ]
    return subsets


# Re-export discover_subsets from dataset_toml for convenience
discover_subsets = _discover_subsets


def calculate_repeats(num_images: int, max_steps: int = 800, batch_size: int = 4) -> int:
    """Calculate num_repeats so one epoch covers ~max_steps.

    With max_steps taking precedence over epochs in kohya-ss, repeats
    controls how many epochs it takes to reach max_steps. We target
    ~3-5 epochs for smooth convergence.

    Formula: effective_images = num_images * repeats
             batches_per_epoch = effective_images / batch_size
             epochs_to_target = max_steps / batches_per_epoch
             Target 4 epochs: repeats = ceil(max_steps * batch_size / (num_images * 4))
    """
    batches_per_epoch_needed = max_steps / 4  # target ~4 epochs
    return max(1, math.ceil(batches_per_epoch_needed / num_images))


def generate_job_id() -> str:
    """Generate a unique job ID: timestamp + random suffix."""
    ts = int(time.time() * 1000)
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"job-{ts}-{suffix}"


def parse_list_value(value: str) -> list:
    """Parse a comma-separated value string into a list of mixed types."""
    parts = [p.strip() for p in value.split(",")]
    result = []
    for part in parts:
        if not part:
            continue
        # Try int
        try:
            int_val = int(part)
            if str(int_val) == part:
                result.append(int_val)
                continue
        except ValueError:
            pass
        # Try float
        try:
            result.append(float(part))
            continue
        except ValueError:
            pass
        result.append(part)
    return result


def generate_permutations(param_dict: dict) -> list[dict]:
    """Generate Cartesian product of parameter lists."""
    if not param_dict:
        return [{}]
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def extract_tags(directory: str) -> list[str]:
    """Extract unique tags from .txt caption files (recursive)."""
    tags: set[str] = set()
    for file_path in sorted(Path(directory).rglob("*.txt")):
        if not file_path.is_file():
            continue
        content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        for tag in content.split(","):
            tag = tag.strip().lower()
            if tag:
                tags.add(tag)
    return sorted(tags)


def generate_test_prompt(tags: list[str], num_tags: int = 10, seed: int = 42) -> str:
    """Generate a test prompt from training data tags."""
    if not tags:
        return "masterpiece"
    rng = random.Random(seed)
    selected = rng.sample(tags, min(len(tags), num_tags))
    return "masterpiece, " + ", ".join(selected)


def build_output_dir(dataset_dir: str, output_base: str | None, job_id: str) -> Path:
    """Build the output directory path."""
    dataset_dir_path = Path(dataset_dir).resolve()
    if output_base:
        return Path(output_base) / job_id
    # Default: datasets/<name>/out/<job_id>
    datasets_parent = dataset_dir_path.parent
    if datasets_parent.name == "img":
        datasets_parent = datasets_parent.parent
    return datasets_parent / "out" / job_id


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
        # Check that the marker's dataset path matches
        if data.get("dataset") != str(Path(dataset_dir).resolve()):
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def write_validation_marker(dataset_dir: str, subsets: list[dict], params: dict, warnings: list[str]) -> Path:
    """Write the validation marker file after successful validation."""
    out_dir = get_dataset_out_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_path = out_dir / VALIDATION_MARKER

    total_images = sum(s["num_images"] for s in subsets)
    total_captions = sum(s.get("num_captions", 0) for s in subsets)

    marker_data = {
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


# ── Validation ───────────────────────────────────────────────────────────
def validate_dataset(image_dir: str, max_steps: int = 800, batch_size: int = 4) -> tuple[bool, list[str]]:
    """Validate a training dataset.

    Returns (is_valid, warnings) tuple. is_valid is False for hard errors
    (missing directory, no images, no captions). Warnings are collected
    for non-fatal issues (image count out of range, missing some captions).

    Rules:
      - Each folder: 10-25 images (warnings outside this range)
      - Total: 25 (base) + 15 × N_outfits is the max recommended
      - NO captions at all = hard error (showstopper)
      - Some missing captions = warning
    """
    path = Path(image_dir)
    warnings: list[str] = []

    if not path.exists():
        print(f"ERROR: Directory does not exist: {image_dir}", file=sys.stderr)
        return False, ["Directory does not exist"]
    if not path.is_dir():
        print(f"ERROR: Not a directory: {image_dir}", file=sys.stderr)
        return False, ["Not a directory"]

    # Discover all subsets (root + subdirectories with images)
    subsets = discover_subsets(image_dir)

    if not subsets:
        print(f"ERROR: No image files found in {image_dir}!", file=sys.stderr)
        return False, ["No image files found"]

    total_images = sum(s["num_images"] for s in subsets)
    total_captions = sum(s.get("num_captions", 0) for s in subsets)

    print(f"Dataset: {image_dir}")

    # ── Per-folder checks ──────────────────────────────────────────────
    num_subsets = len(subsets)
    is_base = True  # first subset is always the root/base

    if num_subsets > 1:
        print(f"  Folders discovered ({num_subsets} subsets):")
    else:
        print(f"  (single folder)")

    for s in subsets:
        folder = Path(s["image_dir"])
        rel = folder.relative_to(path) if str(folder).startswith(str(path)) else folder.name
        img_count = s["num_images"]
        cap_count = s.get("num_captions", 0)
        cap_status = "✓" if cap_count >= img_count else f"⚠ {cap_count}/{img_count} captions"
        print(f"    {rel}/  —  {img_count} images, {cap_status}")

        # Per-folder image count checks
        if is_base:
            max_allowed = MAX_IMAGES_BASE
        else:
            max_allowed = MIN_IMAGES_PER_OUTFIT

        if img_count < MIN_IMAGES_PER_FOLDER:
            msg = f"{rel}/ has only {img_count} images (recommended min: {MIN_IMAGES_PER_FOLDER})"
            print(f"  WARNING: {msg}", file=sys.stderr)
            warnings.append(msg)
        elif img_count > max_allowed:
            msg = f"{rel}/ has {img_count} images (recommended max: {max_allowed})"
            print(f"  WARNING: {msg}", file=sys.stderr)
            warnings.append(msg)
        else:
            print(f"    ✓ {rel}/ image count is in range ({MIN_IMAGES_PER_FOLDER}-{max_allowed})")

        is_base = False

    print(f"  Total images: {total_images}")
    print(f"  Total captions: {total_captions}")

    # ── Total image count check ────────────────────────────────────────
    # Max recommended: 25 (base) + 15 × (num_outfits)
    num_outfits = max(0, num_subsets - 1)
    max_recommended = MAX_IMAGES_BASE + (MIN_IMAGES_PER_OUTFIT * num_outfits)

    if total_images > max_recommended:
        msg = f"Total images ({total_images}) exceeds recommended max ({max_recommended} = {MAX_IMAGES_BASE} base + {MIN_IMAGES_PER_OUTFIT} × {num_outfits} outfits)"
        print(f"  WARNING: {msg}", file=sys.stderr)
        warnings.append(msg)

    # ── Caption checks ─────────────────────────────────────────────────
    if total_captions == 0:
        print(f"ERROR: No caption files (.txt) found!", file=sys.stderr)
        print(f"  Every image needs a matching .txt caption file.", file=sys.stderr)
        return False, ["No caption files found — every image needs a .txt caption"]
    elif total_captions < total_images:
        missing = total_images - total_captions
        msg = f"{missing} image(s) may be missing captions ({total_captions}/{total_images} have .txt files)"
        print(f"  WARNING: {msg}", file=sys.stderr)
        warnings.append(msg)
    else:
        print(f"  All images have captions.")

    # ── Training math ──────────────────────────────────────────────────
    repeats = calculate_repeats(total_images, max_steps, batch_size)
    effective = total_images * repeats
    batches_per_epoch = effective / batch_size
    epochs_for_target = max_steps / batches_per_epoch if batches_per_epoch > 0 else float("inf")

    print(f"  With maxSteps={max_steps}, batchSize={batch_size}:")
    print(f"    Recommended repeats: {repeats}")
    print(f"    Effective images:    {effective}")
    print(f"    Batches/epoch:       {batches_per_epoch:.0f}")
    print(f"    Epochs for {max_steps} steps: ~{epochs_for_target:.1f}")

    # Create output directory and write validation marker
    params = {"max_steps": max_steps, "batch_size": batch_size}
    marker_path = write_validation_marker(image_dir, subsets, params, warnings)
    print(f"\n  ✓ Validated — marker written to {marker_path}")
    print(f"    Output directory: {get_dataset_out_dir(image_dir)}")

    return True, warnings


# ── In-process training ──────────────────────────────────────────────────

class _TqdmProgressWrapper(tqdm):
    """Wrapper around tqdm that tracks progress and checks for cancel signals.

    Installed as a drop-in replacement for tqdm before kohya-ss training starts.
    Every call to .update() fires callbacks so the CLI can update manifests.
    """
    _callbacks: list = []
    _cancel_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0

    @classmethod
    def set_callbacks(cls, callbacks: list, cancel_path: Path | None = None):
        cls._callbacks = callbacks
        cls._cancel_path = cancel_path

    def update(self, n=1):
        self._step_count += n
        for cb in self._callbacks:
            try:
                cb(self._step_count, self)
            except Exception:
                pass
        # Check cancel signal
        if self._cancel_path and self._cancel_path.exists():
            logger.info("Cancel signal detected")
            raise KeyboardInterrupt("Training cancelled")
        return super().update(n)


def _build_kohya_args(params: dict, kohya_parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Build a full kohya-ss Namespace using parser defaults + wrapper overrides."""
    p = params
    args = kohya_parser.parse_args([])

    # Override only the values controlled by this wrapper.
    args.pretrained_model_name_or_path = MODEL_PATHS["diffusion_model"]
    args.qwen3 = MODEL_PATHS["text_encoder"]
    args.vae = MODEL_PATHS["vae"]
    args.dataset_config = p["dataset_config"]
    args.output_dir = p["output_dir"]
    args.output_name = p["lora_name"]
    args.save_model_as = "safetensors"
    args.network_module = "networks.lora_anima"
    args.network_dim = p["network_dim"]
    args.network_alpha = p["network_alpha"]
    args.learning_rate = p["learning_rate"]
    args.train_batch_size = p["batch_size"]
    args.optimizer_type = p["optimizer"]
    args.lr_scheduler = p["scheduler"]
    args.timestep_sampling = p["timestep_sampling"]
    args.discrete_flow_shift = 1.0
    args.mixed_precision = p["mixed_precision"]
    args.max_train_steps = p["max_steps"]
    args.save_every_n_steps = max(1, p["max_steps"] // 10)
    args.gradient_checkpointing = p.get("gradient_checkpointing", True)
    args.cache_latents = p.get("cache_latents", True)
    args.cache_text_encoder_outputs = p.get("cache_text_encoder", False)
    args.network_train_unet_only = p.get("cache_text_encoder", False)
    args.vae_chunk_size = 64
    args.vae_disable_cache = True

    return args


def run_single_training(params: dict, output_dir: Path, job_id: str) -> dict:
    """Run a single training job in-process (no subprocess spawning).

    Directly imports and calls kohya-ss's AnimaNetworkTrainer.train(),
    wrapping tqdm for progress tracking and cancel support.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "training.log"
    manifest_path = output_dir / "job_manifest.json"

    # Write job manifest
    manifest = {
        "jobId": job_id,
        "status": "running",
        "params": {k: v for k, v in params.items() if k != "output_dir"},
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "current_step": 0,
        "total_steps": params.get("max_steps", 800),
        "avg_loss": None,
    }
    _write_json(manifest_path, manifest)

    # Discover all image folders (root + subdirectories)
    subsets = discover_subsets(params["training_images"])
    if not subsets:
        logger.error(f"No images found in {params['training_images']}")
        return {"status": "failed", "error": "No images found", "output_dir": str(output_dir)}

    total_images = sum(s["num_images"] for s in subsets)
    num_repeats = params.get("repeats") or calculate_repeats(
        total_images, params.get("max_steps", 800), params.get("batch_size", 4)
    )

    # Build subset list for dataset TOML
    subset_configs = [
        {"image_dir": s["image_dir"], "num_repeats": num_repeats}
        for s in subsets
    ]

    rebalance_subsets = maybe_build_bucket_rebalance_subset(
        training_images=params["training_images"],
        output_dir=output_dir,
        num_repeats=num_repeats,
        enabled=params.get("rebalance_buckets", False),
        dominance_threshold=params.get("bucket_dominance_threshold", 0.20),
        max_augmented_images=params.get("bucket_rebalance_max_aug", 64),
        seed=params.get("bucket_rebalance_seed", 42),
        resolution=params.get("resolution", 1024),
    )
    if rebalance_subsets is not None:
        # Rebalance returns subsets that REPLACE the original ones
        # (dominant + non-dominant + crops with per-subset repeats)
        subset_configs = rebalance_subsets

    # Generate dataset TOML
    dataset_toml_path = output_dir / "dataset.toml"
    generate_dataset_toml(
        batch_size=params["batch_size"],
        num_images=total_images,
        epochs=params.get("epochs", 10),
        num_repeats=num_repeats,
        output_path=str(dataset_toml_path),
        resolution=params.get("resolution", 1024),
        cache_text_encoder_outputs=params.get("cache_text_encoder", False),
        caption_tag_dropout_rate=params.get("caption_tag_dropout_rate", 0.05),
        keep_tokens=params.get("keep_tokens", 1),
        subsets=subset_configs,
    )
    logger.info(f"Dataset TOML: {dataset_toml_path} ({len(subsets)} subset(s), repeats={num_repeats}, {total_images} images)")

    # Create zip backup of training data (preserves folder structure)
    try:
        zip_training_data(params["training_images"], str(output_dir))
        logger.info(f"Training data zipped to {output_dir}/training-data.zip")
    except Exception as e:
        logger.warning(f"Zip creation skipped: {e}")

    # ── Setup in-process Kohya-ss training ─────────────────────────────
    params["dataset_config"] = str(dataset_toml_path)

    # Add sd-scripts to path so we can import kohya-ss modules
    sd_scripts_dir = str(_project_root / "sd-scripts")
    if sd_scripts_dir not in sys.path:
        sys.path.insert(0, sd_scripts_dir)

    # Progress tracking state (set up BEFORE importing kohya-ss)
    current_step = 0
    total_steps = params.get("max_steps", 800)
    avg_loss = None
    cancel_path = _project_root / "jobs" / f"{job_id}.cancel"

    def _on_step(step: int, bar):
        nonlocal current_step, total_steps, avg_loss
        current_step = step
        total_steps = bar.total if bar.total else total_steps
        # Try to read loss from tqdm's internal format dict
        if hasattr(bar, "format_dict") and "avr_loss" in bar.format_dict:
            avg_loss = round(float(bar.format_dict["avr_loss"]), 6)
        # Update manifest every 20 steps
        if step % 20 == 0 and step > 0:
            manifest["current_step"] = current_step
            manifest["total_steps"] = total_steps
            manifest["avg_loss"] = avg_loss
            _write_json(manifest_path, manifest)

    # Install tqdm wrapper BEFORE importing kohya-ss modules,
    # so their "from tqdm import tqdm" gets our wrapper
    _TqdmProgressWrapper.set_callbacks([_on_step], cancel_path)
    import tqdm as tqdm_module
    _real_tqdm = tqdm_module.tqdm
    tqdm_module.tqdm = _TqdmProgressWrapper

    # Now import kohya-ss modules — they'll use our patched tqdm
    import torch
    from anima_train_network import AnimaNetworkTrainer, setup_parser as setup_anima_parser
    import library.train_util as train_util

    # Build args namespace for kohya-ss
    args = _build_kohya_args(params, setup_anima_parser())

    # Set environment for single-process CPU thread control
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    logger.info(f"Starting in-process training: {params['lora_name']}")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Output: {output_dir}")

    exit_code = 0
    try:
        # Kohya-ss arg validation
        train_util.verify_command_line_training_args(args)

        # Handle attn_mode backward compat (same as anima_train_network.py __main__)
        if hasattr(args, "attn_mode") and args.attn_mode == "sdpa":
            args.attn_mode = "torch"

        # Run training in-process
        trainer = AnimaNetworkTrainer()
        trainer.train(args)

    except KeyboardInterrupt:
        logger.info("Training cancelled")
        exit_code = -1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exit_code = 1
    finally:
        # Final manifest update
        manifest["current_step"] = current_step
        manifest["total_steps"] = total_steps
        manifest["avg_loss"] = avg_loss
        manifest["exit_code"] = exit_code
        manifest["status"] = "completed" if exit_code == 0 else (
            "cancelled" if cancel_path.exists() else "failed"
        )
        _write_json(manifest_path, manifest)

        # Restore original tqdm
        tqdm_module.tqdm = _real_tqdm

        # Clean up cancel file
        if cancel_path.exists():
            cancel_path.unlink(missing_ok=True)

        return {
            "status": manifest["status"],
            "output_dir": str(output_dir),
            "exit_code": exit_code,
        }


def _write_json(path: Path, data: dict) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── CLI ──────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA Matrix Trainer — single and matrix training modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a dataset
  %(prog)s --validate --dataset datasets/froot/img

  # Single training run with defaults
  %(prog)s --mode single --dataset datasets/froot/img --name Froot-Anima

  # Custom parameters
  %(prog)s --mode single --dataset datasets/froot/img --name Froot --lr 0.0001 --bs 2 --network-dim 32

  # Matrix run (all permutations of given values)
  %(prog)s --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,32 --alpha 1,16 --lr 0.0001,0.0002

  # Matrix with output dir
  %(prog)s --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,20,32 --alpha 1,20 -o datasets/froot/out/matrix-run
""",
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        choices=["single", "matrix"],
        default="single",
        help="Training mode: single run or matrix (permutation) sweep [default: single]",
    )

    # Validate
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate dataset (check image count, captions) and exit",
    )

    # Required
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to directory containing training images and .txt captions",
    )

    # Name
    parser.add_argument(
        "--name", "-n",
        help="LoRA output name (default: dataset folder name)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output directory base (default: datasets/<name>/out/<job-id>)",
    )

    # ── Training parameters (accept comma-separated values in matrix mode) ──
    parser.add_argument(
        "--network-dim",
        type=str,
        default=str(DEFAULTS["network_dim"]),
        help=f"LoRA dimension(s) [default: {DEFAULTS['network_dim']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=str,
        default=str(DEFAULTS["network_alpha"]),
        help=f"LoRA alpha(s) [default: {DEFAULTS['network_alpha']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=str,
        default=str(DEFAULTS["learning_rate"]),
        help=f"Learning rate(s) [default: {DEFAULTS['learning_rate']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--batch-size", "--bs",
        type=str,
        default=str(DEFAULTS["batch_size"]),
        help=f"Batch size(s) [default: {DEFAULTS['batch_size']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--max-steps", "--ss",
        type=str,
        default=str(DEFAULTS["max_steps"]),
        help=f"Max training step(s) [default: {DEFAULTS['max_steps']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULTS["optimizer"],
        help=f"Optimizer(s) [default: {DEFAULTS['optimizer']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--scheduler", "-s",
        type=str,
        default=DEFAULTS["scheduler"],
        help=f"LR scheduler(s) [default: {DEFAULTS['scheduler']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=str(DEFAULTS["resolution"]),
        help=f"Resolution(s) [default: {DEFAULTS['resolution']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=None,
        help="Override num_repeats (auto-calculated from image count if omitted)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--no-cache-latents",
        action="store_true",
        help="Disable latent caching",
    )
    parser.add_argument(
        "--cache-text-encoder",
        action="store_true",
        help="Enable text encoder output caching",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=DEFAULTS["mixed_precision"],
        help=f"Mixed precision [default: {DEFAULTS['mixed_precision']}]",
    )
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default=DEFAULTS["timestep_sampling"],
        help=f"Timestep sampling [default: {DEFAULTS['timestep_sampling']}]",
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=DEFAULTS["caption_tag_dropout_rate"],
        help=f"Caption tag dropout rate [default: {DEFAULTS['caption_tag_dropout_rate']}]",
    )
    parser.add_argument(
        "--keep-tokens",
        type=int,
        default=DEFAULTS["keep_tokens"],
        help=f"Keep first N tokens from shuffle [default: {DEFAULTS['keep_tokens']}]",
    )
    parser.add_argument(
        "--rebalance-buckets",
        action="store_true",
        help="Detect dominant bucket skew (>20% default) and redistribute by cropping excess images to adjacent buckets",
    )
    parser.add_argument(
        "--bucket-dominance-threshold",
        type=float,
        default=0.20,
        help="Dominant bucket share threshold that triggers rebalancing [default: 0.20]",
    )
    parser.add_argument(
        "--bucket-rebalance-max-aug",
        type=int,
        default=64,
        help="Maximum random-crop augmented samples to generate for bucket rebalance [default: 64]",
    )
    parser.add_argument(
        "--bucket-rebalance-seed",
        type=int,
        default=42,
        help="Random seed for bucket rebalance crop selection [default: 42]",
    )

    # Matrix resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="(Matrix mode) Resume from existing manifest",
    )

    return parser


def parse_params(args) -> dict:
    """Parse CLI args into a training params dict (single value each)."""
    return {
        "network_dim": int(args.network_dim),
        "network_alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "max_steps": int(args.max_steps),
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "resolution": int(args.resolution),
        "mixed_precision": args.mixed_precision,
        "timestep_sampling": args.timestep_sampling,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "cache_latents": not args.no_cache_latents,
        "cache_text_encoder": args.cache_text_encoder,
        "caption_tag_dropout_rate": args.caption_dropout,
        "keep_tokens": args.keep_tokens,
        "repeats": args.repeats,
        "rebalance_buckets": args.rebalance_buckets,
        "bucket_dominance_threshold": args.bucket_dominance_threshold,
        "bucket_rebalance_max_aug": args.bucket_rebalance_max_aug,
        "bucket_rebalance_seed": args.bucket_rebalance_seed,
    }


def parse_param_ranges(args) -> dict:
    """Parse CLI args into param ranges dict (lists for matrix mode)."""
    range_keys = [
        ("network_dim", args.network_dim),
        ("network_alpha", args.alpha),
        ("learning_rate", args.learning_rate),
        ("batch_size", args.batch_size),
        ("max_steps", args.max_steps),
        ("optimizer", args.optimizer),
        ("scheduler", args.scheduler),
        ("resolution", args.resolution),
    ]
    ranges = {}
    for key, value in range_keys:
        parsed = parse_list_value(value)
        if len(parsed) > 1:
            ranges[key] = parsed
    # Boolean flags are not ranged
    return ranges


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Validate mode ──────────────────────────────────────────────────
    if args.validate:
        valid, warnings = validate_dataset(
            args.dataset,
            max_steps=int(args.max_steps),
            batch_size=int(args.batch_size),
        )
        if warnings:
            print(f"\n  ({len(warnings)} warning(s) — training is still allowed)")
        sys.exit(0 if valid else 1)

    # ── Training gate: check validation marker ─────────────────────────
    if not check_validation(args.dataset):
        out_dir = get_dataset_out_dir(args.dataset)
        print(f"ERROR: Dataset has not been validated.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Run validation first:", file=sys.stderr)
        print(f"    uv run python scripts/train.py --validate --dataset {args.dataset}", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  This creates the output directory ({out_dir}/) and verifies", file=sys.stderr)
        print(f"  your dataset structure before training.", file=sys.stderr)
        sys.exit(1)

    # ── Resolve lora name ──────────────────────────────────────────────
    dataset_path = Path(args.dataset).resolve()
    if args.name:
        lora_name = args.name
    else:
        # Use parent folder name (e.g., "datasets/froot/img" → "froot")
        name = dataset_path.name
        if name == "img":
            name = dataset_path.parent.name
        lora_name = name

    # ── Single mode ────────────────────────────────────────────────────
    if args.mode == "single":
        job_id = generate_job_id()
        output_dir = build_output_dir(args.dataset, args.output, job_id)

        params = parse_params(args)
        params["lora_name"] = lora_name
        params["training_images"] = str(dataset_path)
        params["output_dir"] = str(output_dir)
        params["job_id"] = job_id

        # Ensure jobs dir exists for cancel signal
        (_project_root / "jobs").mkdir(exist_ok=True)

        logger.info(f"Starting single training: {lora_name}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Params: {json.dumps(params, indent=2, default=str)}")

        result = run_single_training(params, output_dir, job_id)

        if result["status"] == "completed":
            print(f"\nTraining completed: {result['output_dir']}")
        else:
            print(f"\nTraining {result['status']}: exit code {result.get('exit_code')}", file=sys.stderr)
            sys.exit(1)

    # ── Matrix mode ────────────────────────────────────────────────────
    else:
        job_id = generate_job_id()
        output_base = build_output_dir(args.dataset, args.output, job_id)
        output_base.mkdir(parents=True, exist_ok=True)

        # Parse param ranges
        param_ranges = parse_param_ranges(args)
        if not param_ranges:
            print("ERROR: Matrix mode requires at least one parameter with multiple values (comma-separated).", file=sys.stderr)
            print("Example: --network-dim 16,20,32 --alpha 1,20", file=sys.stderr)
            sys.exit(1)

        # Generate permutations
        permutations = generate_permutations(param_ranges)
        logger.info(f"Generated {len(permutations)} permutations")

        # Base params (single values, non-ranged)
        base_params = parse_params(args)
        base_params["lora_name"] = lora_name
        base_params["training_images"] = str(dataset_path)
        base_params["job_id"] = job_id

        # Generate test prompt from training data tags
        all_tags = extract_tags(str(dataset_path))
        test_prompt = generate_test_prompt(all_tags, num_tags=10, seed=42)
        logger.info(f"Test prompt: {test_prompt}")
        logger.info(f"Extracted {len(all_tags)} tags from training data")

        # Manifest
        manifest_path = output_base / "manifest.json"
        manifest = {
            "jobId": job_id,
            "mode": "matrix",
            "param_ranges": {k: list(v) for k, v in param_ranges.items()},
            "total": len(permutations),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "test_prompt": test_prompt,
            "permutations": [
                {
                    "index": i,
                    "params": perm,
                    "status": "pending",
                    "output_dir": None,
                    "error": None,
                }
                for i, perm in enumerate(permutations)
            ],
        }

        # Resume support
        pending_indices = list(range(len(permutations)))
        if args.resume and manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
                if "permutations" in existing and len(existing["permutations"]) == len(permutations):
                    manifest = existing
                    pending_indices = [
                        i for i, p in enumerate(permutations)
                        if existing["permutations"][i]["status"] == "pending"
                    ]
                    logger.info(f"Resuming: {len(pending_indices)} pending permutations")
            except (json.JSONDecodeError, KeyError):
                pass

        _write_json(manifest_path, manifest)

        # Ensure jobs dir exists
        (_project_root / "jobs").mkdir(exist_ok=True)

        cancel_path = output_base / "cancel"
        completed = 0
        failed = 0

        for idx in pending_indices:
            if cancel_path.exists():
                logger.info("Cancel signal detected. Stopping.")
                manifest["cancelled"] = len(permutations) - idx
                break

            perm = permutations[idx]

            # Build permutation-specific param name
            perm_parts = [f"{k}-{v}" for k, v in perm.items()]
            perm_name = "x".join(perm_parts)
            perm_dir = output_base / perm_name

            status_str = f"[{idx + 1}/{len(permutations)}] {perm_name}"
            print(f"\n{'='*60}")
            print(f"{status_str}")
            print(f"{'='*60}")

            # Merge base + permutation params
            run_params = {**base_params, **perm}
            run_params["output_dir"] = str(perm_dir)

            # Update manifest
            manifest["permutations"][idx]["status"] = "running"
            manifest["permutations"][idx]["output_dir"] = str(perm_dir)
            _write_json(manifest_path, manifest)

            try:
                result = run_single_training(run_params, perm_dir, f"{job_id}-{idx}")

                if result["status"] == "completed":
                    manifest["permutations"][idx]["status"] = "completed"
                    manifest["completed"] += 1
                    completed += 1
                    print(f"  Completed: {perm_name}")
                else:
                    error = f"exit code {result.get('exit_code', 'unknown')}"
                    manifest["permutations"][idx]["status"] = "failed"
                    manifest["permutations"][idx]["error"] = error
                    manifest["failed"] += 1
                    failed += 1
                    print(f"  Failed: {perm_name} — {error}", file=sys.stderr)

                _write_json(manifest_path, manifest)

            except Exception as e:
                manifest["permutations"][idx]["status"] = "failed"
                manifest["permutations"][idx]["error"] = str(e)
                manifest["failed"] += 1
                failed += 1
                print(f"  Exception: {perm_name} — {e}", file=sys.stderr)
                _write_json(manifest_path, manifest)

        # Summary
        skipped = len(permutations) - completed - failed - manifest.get("cancelled", 0)
        print(f"\n{'='*60}")
        print(f"Matrix training finished:")
        print(f"  Completed: {completed}")
        print(f"  Failed:    {failed}")
        print(f"  Skipped:   {skipped}")
        if manifest.get("cancelled", 0) > 0:
            print(f"  Cancelled: {manifest['cancelled']}")
        print(f"  Output:    {output_base}")
        print(f"{'='*60}")

        if failed > 0 and completed == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
