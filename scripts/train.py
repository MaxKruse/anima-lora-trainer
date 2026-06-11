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
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from itertools import product

# ── Project setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.command_builder import build_training_command
from scripts.dataset_toml import generate_dataset_toml, discover_subsets as _discover_subsets
from scripts.zip_training_data import zip_training_data

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


# ── Training ─────────────────────────────────────────────────────────────
def run_single_training(params: dict, output_dir: Path, job_id: str) -> dict:
    """Run a single training job with progress tracking."""
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

    # Build training command
    params["dataset_config"] = str(dataset_toml_path)
    cmd = build_training_command(params)

    # Launch via uv run
    full_cmd = ["uv", "run"] + cmd
    logger.info(f"Launching: {' '.join(full_cmd[:10])}...")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # Parse tqdm progress from kohya-ss output
    tqdm_re = re.compile(r"steps:\s+(\d+)%.*?\|\s+(\d+)/(\d+)\s+.*?avr_loss=([\d.]+)")
    tqdm_min_re = re.compile(r"steps:\s+(\d+)%.*?\|\s+(\d+)/(\d+)")

    current_step = 0
    total_steps = params.get("max_steps", 800)
    avg_loss = None

    proc = subprocess.Popen(
        full_cmd,
        cwd=str(_project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    cancel_path = _project_root / "jobs" / f"{job_id}.cancel"

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for raw_line in iter(proc.stdout.readline, b""):
                # Check cancel
                if cancel_path.exists():
                    logger.info("Cancelled")
                    proc.terminate()
                    break

                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                log_file.write(line + "\n")
                log_file.flush()

                # Parse progress
                m = tqdm_re.search(line)
                if m:
                    pct, cur, tot, loss = m.groups()
                    current_step = int(cur)
                    total_steps = int(tot)
                    avg_loss = round(float(loss), 6)
                else:
                    m = tqdm_min_re.search(line)
                    if m:
                        pct, cur, tot = m.groups()
                        current_step = int(cur)
                        total_steps = int(tot)

                # Update manifest periodically
                if current_step % 20 == 0 and current_step > 0:
                    manifest["current_step"] = current_step
                    manifest["total_steps"] = total_steps
                    manifest["avg_loss"] = avg_loss
                    _write_json(manifest_path, manifest)

    finally:
        exit_code = proc.wait()

        # Final manifest update
        manifest["current_step"] = current_step
        manifest["total_steps"] = total_steps
        manifest["avg_loss"] = avg_loss
        manifest["exit_code"] = exit_code
        manifest["status"] = "completed" if exit_code == 0 else "failed"
        if cancel_path.exists():
            manifest["status"] = "cancelled"
        _write_json(manifest_path, manifest)

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
