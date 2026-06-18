"""Unified LoRA training CLI — single and matrix modes.

Entry point that orchestrates validation, dataset config generation,
and in-process kohya-ss training.

Usage:
    uv run python scripts/train.py --validate --dataset datasets/froot/img
    uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima
    uv run python scripts/train.py --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,32 --alpha 1,16
"""

import json
import logging
import os
import random
import sys
import time
import traceback
from itertools import product
from pathlib import Path

from tqdm import tqdm

# ── Project setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.bucket_rebalance import (
    maybe_build_bucket_rebalance_subset,
)
from scripts.cli_args import (
    build_parser,
    ensure_dataset,
    parse_params,
    parse_param_ranges,
)
from scripts.constants import (
    IMAGE_EXTENSIONS,
    MODEL_PATHS,
    PROJECT_ROOT,
)
from scripts.dataset_toml import (
    discover_subsets,
    generate_dataset_toml,
)
from scripts.validation import (
    calculate_max_steps,
    calculate_repeats,
    check_validation,
    get_dataset_out_dir,
    validate_dataset,
)
from scripts.zip_training_data import zip_training_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _write_json(path: Path, data: dict) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── Utility functions ────────────────────────────────────────────────────
def generate_job_id() -> str:
    """Generate a unique job ID: timestamp + random suffix."""
    ts = int(time.time() * 1000)
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"job-{ts}-{suffix}"


def generate_permutations(param_dict: dict) -> list[dict]:
    """Generate Cartesian product of parameter lists."""
    if not param_dict:
        return [{}]
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_output_dir(dataset_dir: str, output_base) -> Path:
    """Build the output directory path (no job_id nesting)."""
    dataset_dir_path = Path(dataset_dir).resolve()
    if output_base:
        return Path(output_base)
    datasets_parent = dataset_dir_path.parent
    if datasets_parent.name == "img":
        datasets_parent = datasets_parent.parent
    return datasets_parent / "out"


def _perm_suffix(perm: dict) -> str:
    """Build a short filename suffix from permutation params, e.g. 'lr-0.0001-bs-2'."""
    aliases = {
        "learning_rate": "lr",
        "batch_size": "bs",
        "network_dim": "dim",
        "network_alpha": "alpha",
        "max_steps": "steps",
        "optimizer": "opt",
        "scheduler": "sched",
        "resolution": "res",
    }
    parts = []
    for k, v in perm.items():
        short = aliases.get(k, k)
        parts.append(f"{short}-{v}")
    return "-".join(parts)


def _copy_final_model(perm_dir: Path, output_base: Path, lora_name: str, perm: dict) -> None:
    """Copy the final .safetensors from a working dir to the output folder."""
    final_model = perm_dir / f"{lora_name}.safetensors"
    if not final_model.exists():
        return
    suffix = _perm_suffix(perm)
    name = f"{lora_name}-{suffix}.safetensors" if suffix else f"{lora_name}.safetensors"
    dest = output_base / name
    dest.write_bytes(final_model.read_bytes())
    logger.info("Copied final model: %s", dest.name)


def _count_images_in_dir(image_dir: str) -> int:
    """Count image files in a directory (non-recursive)."""
    path = Path(image_dir)
    if not path.exists() or not path.is_dir():
        return 0
    return sum(
        1
        for entry in path.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_latest_state_dir(perm_dir: Path, lora_name: str) -> Path | None:
    """Find the latest accelerator state directory for resume.

    kohya-ss saves state dirs like:
      {output_dir}/{output_name}-step{NNNNNNNN}-state/
      {output_dir}/{output_name}-state/  (final)

    Returns the path to the latest state dir, or None if not found.
    """
    if not perm_dir.exists():
        return None

    # Look for step-based state dirs: {lora_name}-step{NNNNNNNN}-state
    pattern_prefix = f"{lora_name}-step"
    suffix = "-state"
    state_dirs = [
        d for d in perm_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(pattern_prefix)
        and d.name.endswith(suffix)
    ]
    if state_dirs:
        # Sort by name (step number is zero-padded, so lexicographic works)
        state_dirs.sort(key=lambda d: d.name)
        return state_dirs[-1]

    # Fallback: final state dir
    final_state = perm_dir / f"{lora_name}-state"
    if final_state.is_dir():
        return final_state

    return None


def _is_incomplete(perm_dir: Path) -> bool:
    """Check if a permutation run is incomplete (has no final model)."""
    manifest = perm_dir / "job_manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
        status = data.get("status", "")
        # "running" means aborted mid-run, "failed"/"cancelled" are incomplete
        return status in ("running", "failed", "cancelled")
    except (json.JSONDecodeError, OSError):
        return False


def _get_resume_info(perm_dir: Path, lora_name: str) -> dict:
    """Get resume info for an incomplete run.

    Returns dict with:
      - incomplete: bool
      - current_step: int
      - total_steps: int
      - state_dir: Path | None
    """
    manifest = perm_dir / "job_manifest.json"
    if not manifest.exists():
        return {"incomplete": False, "current_step": 0, "total_steps": 0, "state_dir": None}

    try:
        data = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return {"incomplete": False, "current_step": 0, "total_steps": 0, "state_dir": None}

    status = data.get("status", "")
    incomplete = status in ("running", "failed", "cancelled")
    state_dir = _find_latest_state_dir(perm_dir, lora_name) if incomplete else None

    return {
        "incomplete": incomplete,
        "current_step": data.get("current_step", 0),
        "total_steps": data.get("total_steps", 0),
        "state_dir": state_dir,
    }


# ── In-process training ──────────────────────────────────────────────────
class _TqdmProgressWrapper(tqdm):
    """Wrapper around tqdm that tracks progress and checks for cancel signals."""
    _callbacks: list = []
    _cancel_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0

    @classmethod
    def set_callbacks(cls, callbacks, cancel_path=None):
        cls._callbacks = callbacks
        cls._cancel_path = cancel_path

    def update(self, n=1):
        self._step_count += n
        for cb in self._callbacks:
            try:
                cb(self._step_count, self)
            except Exception:
                pass
        if self._cancel_path and self._cancel_path.exists():
            logger.info("Cancel signal detected")
            raise KeyboardInterrupt("Training cancelled")
        return super().update(n)


# ── Custom save points (monkey-patch kohya-ss) ──────────────────────────
_save_points: set[int] = set()
_original_save_sd_model = None


def _calculate_save_steps(max_steps: int) -> list[int]:
    """Calculate checkpoint save steps at specific percentages.

    Saves at 50%, 66%, 75%, 82.5% of training (final 100% is handled by kohya-ss).
    """
    percentages = [0.50, 2/3, 0.75, 0.825]
    return [max(1, int(round(max_steps * p))) for p in percentages]


def _cleanup_extra_checkpoints(output_dir: Path, lora_name: str, max_steps: int) -> None:
    """Remove checkpoint files that aren't at configured save points.

    Keeps:
      - Checkpoints at save point steps (e.g., {name}-step000250.safetensors)
      - Final model ({name}.safetensors)
      - State directories (for resume)
    """
    keep_steps = set(_calculate_save_steps(max_steps)) | {max_steps}
    step_pattern = f"{lora_name}-step"

    for entry in output_dir.iterdir():
        name = entry.name
        # Skip state directories, final model, non-model files
        if name.endswith("-state") or name == f"{lora_name}.safetensors":
            continue
        if not name.startswith(step_pattern) or not name.endswith(".safetensors"):
            continue
        # Extract step number from {name}-step{NNNNNNNN}.safetensors
        try:
            step_str = name[len(step_pattern) : -len(".safetensors")]
            step = int(step_str)
            if step not in keep_steps:
                entry.unlink()
                logger.info("Removed extra checkpoint: %s", name)
        except (ValueError, OSError):
            pass


def _setup_save_filter():
    """Monkey-patch kohya-ss save_sd_model to only save at specific steps.

    Kohya-ss only supports uniform save_every_n_steps intervals.
    We patch the save function to only write checkpoints at our custom percentages.
    Must be called after kohya-ss modules are imported.
    """
    global _original_save_sd_model

    try:
        import library.train_util as train_util
    except ImportError:
        return False

    if not hasattr(train_util, "save_sd_model"):
        return False

    _original_save_sd_model = train_util.save_sd_model

    def _filtered_save(*args, **kwargs):
        if _current_step_global in _save_points:
            return _original_save_sd_model(*args, **kwargs)
        # Skip — not a configured save point

    train_util.save_sd_model = _filtered_save
    return True


def _restore_save():
    """Restore original kohya-ss save function."""
    global _original_save_sd_model
    if _original_save_sd_model is not None:
        try:
            import library.train_util as train_util
            train_util.save_sd_model = _original_save_sd_model
        except ImportError:
            pass
        _original_save_sd_model = None


# Shared step tracker (synced by _on_step callback)
_current_step_global = 0


def _build_kohya_args(params, kohya_parser):
    """Build a full kohya-ss Namespace using parser defaults + overrides."""
    p = params
    args = kohya_parser.parse_args([])
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
    # Disabled — we use monkey-patched save_sd_model for custom save points
    args.save_every_n_steps = p["max_steps"]
    args.gradient_checkpointing = p.get("gradient_checkpointing", True)
    args.cache_latents = p.get("cache_latents", True)
    args.cache_text_encoder_outputs = p.get("cache_text_encoder", False)
    args.network_train_unet_only = p.get("cache_text_encoder", False)
    args.vae_chunk_size = 64
    args.vae_disable_cache = True
    # Save accelerator state for resume support
    args.save_state = True
    # Resume from existing state if provided
    resume_path = p.get("resume")
    if resume_path:
        args.resume = str(resume_path)
    return args


def _reset_kohya_global_state() -> None:
    """Reset kohya-ss global singleton state between in-process training runs.

    kohya-ss uses class-level _strategy singletons in strategy_base that are
    set once and refuse to be overwritten. After each training run these must
    be cleared so the next run can set fresh strategies.
    """
    try:
        from library import strategy_base
        strategy_base.TokenizeStrategy._strategy = None
        strategy_base.LatentsCachingStrategy._strategy = None
        strategy_base.TextEncodingStrategy._strategy = None
        strategy_base.TextEncoderOutputsCachingStrategy._strategy = None
    except ImportError:
        # First run — modules not yet imported
        pass


def run_single_training(params: dict, output_dir: Path, job_id: str, resume: Path | None = None) -> dict:
    """Run a single training job in-process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "job_manifest.json"

    manifest = {
        "jobId": job_id,
        "status": "running",
        "params": {k: v for k, v in params.items() if k != "output_dir"},
        "output_dir": str(output_dir),
        "current_step": 0,
        "total_steps": params.get("max_steps", 800),
        "avg_loss": None,
    }
    _write_json(manifest_path, manifest)

    # Discover subsets and calculate repeats
    subsets = discover_subsets(params["training_images"])
    if not subsets:
        logger.error(f"No images found in {params['training_images']}")
        return {"status": "failed", "error": "No images found", "output_dir": str(output_dir)}

    total_images = sum(s["num_images"] for s in subsets)
    batch_size = params.get("batch_size", 4)

    # Auto-calculate max_steps from batch_size (unless user explicitly set a different value)
    auto_max = calculate_max_steps(batch_size)
    if params.get("max_steps", auto_max) == DEFAULTS["max_steps"]:
        params["max_steps"] = auto_max

    manual_repeats = params.get("repeats")

    # Auto-calculate repeats: target ~12 steps_per_epoch (10-15 range)
    base_repeats = manual_repeats if manual_repeats is not None else calculate_repeats(total_images, batch_size)

    subset_configs = [{"image_dir": s["image_dir"], "num_repeats": base_repeats} for s in subsets]

    # Bucket rebalance
    rebalance_subsets = maybe_build_bucket_rebalance_subset(
        training_images=params["training_images"],
        output_dir=output_dir,
        num_repeats=base_repeats,
        enabled=params.get("rebalance_buckets", False),
        dominance_threshold=params.get("bucket_dominance_threshold", 0.20),
        max_augmented_images=params.get("bucket_rebalance_max_aug", 64),
        seed=params.get("bucket_rebalance_seed", 42),
        resolution=params.get("resolution", 1024),
    )
    if rebalance_subsets is not None:
        subset_configs = rebalance_subsets

    num_repeats = base_repeats

    # Generate dataset TOML
    dataset_toml_path = output_dir / "dataset.toml"
    generate_dataset_toml(
        batch_size=params["batch_size"],
        num_images=total_images,
        epochs=4,  # unused by toml generator, kept for API compat
        num_repeats=num_repeats,
        output_path=str(dataset_toml_path),
        resolution=params.get("resolution", 1024),
        cache_text_encoder_outputs=params.get("cache_text_encoder", False),
        caption_tag_dropout_rate=params.get("caption_tag_dropout_rate", 0.1),
        keep_tokens=params.get("keep_tokens", 1),
        subsets=subset_configs,
    )


    # Zip training data backup
    try:
        zip_training_data(params["training_images"], str(output_dir))
    except Exception:
        pass

    # ── Setup in-process Kohya-ss training ─────────────────────────────
    sd_scripts_dir = str(_project_root / "sd-scripts")
    if sd_scripts_dir not in sys.path:
        sys.path.insert(0, sd_scripts_dir)
    _reset_kohya_global_state()
    params["dataset_config"] = str(dataset_toml_path)
    params["resume"] = str(resume) if resume else None

    current_step = 0
    total_steps = params.get("max_steps", 800)
    avg_loss = None
    error_message = None
    cancel_path = _project_root / "jobs" / f"{job_id}.cancel"

    def _on_step(step, bar):
        nonlocal current_step, total_steps, avg_loss
        global _current_step_global
        current_step = step
        _current_step_global = step
        total_steps = bar.total if bar.total else total_steps
        if hasattr(bar, "format_dict") and "avr_loss" in bar.format_dict:
            avg_loss = round(float(bar.format_dict["avr_loss"]), 6)
        if step % 20 == 0 and step > 0:
            manifest["current_step"] = current_step
            manifest["total_steps"] = total_steps
            manifest["avg_loss"] = avg_loss
            _write_json(manifest_path, manifest)

    _TqdmProgressWrapper.set_callbacks([_on_step], cancel_path)
    import tqdm as tqdm_module
    _real_tqdm = tqdm_module.tqdm
    tqdm_module.tqdm = _TqdmProgressWrapper

    try:
        import torch
        from anima_train_network import AnimaNetworkTrainer, setup_parser as setup_anima_parser
        import library.train_util as train_util

        args = _build_kohya_args(params, setup_anima_parser())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["PYTHONIOENCODING"] = "utf-8"


        train_util.verify_command_line_training_args(args)

        # Setup custom save points (50%, 66%, 75%, 82.5%)
        _save_points.clear()
        _save_points.update(_calculate_save_steps(params["max_steps"]))
        _setup_save_filter()

        if hasattr(args, "attn_mode") and args.attn_mode == "sdpa":
            args.attn_mode = "torch"

        trainer = AnimaNetworkTrainer()
        trainer.train(args)
        exit_code = 0
    except KeyboardInterrupt:

        exit_code = -1
    except Exception as e:
        error_message = str(e)
        logger.error("Training failed: %s", error_message)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        exit_code = 1
    finally:
        # Restore original kohya-ss save function
        _restore_save()
        _save_points.clear()

        # Clean up extra checkpoint files (keep only save points + final model)
        try:
            _cleanup_extra_checkpoints(output_dir, params["lora_name"], params["max_steps"])
        except Exception:
            pass

        manifest["current_step"] = current_step
        manifest["total_steps"] = total_steps
        manifest["avg_loss"] = avg_loss
        manifest["exit_code"] = exit_code
        manifest["status"] = (
            "completed" if exit_code == 0
            else ("cancelled" if cancel_path.exists() else "failed")
        )
        if error_message:
            manifest["error"] = error_message
        _write_json(manifest_path, manifest)
        # Free GPU memory for subsequent matrix jobs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        tqdm_module.tqdm = _real_tqdm
        if cancel_path.exists():
            cancel_path.unlink(missing_ok=True)
        return {"status": manifest["status"], "output_dir": str(output_dir), "exit_code": exit_code, "error": error_message}


# ── CLI main ─────────────────────────────────────────────────────────────
def _run_single(args, dataset_path, lora_name):
    """Execute a single training run."""
    output_base = build_output_dir(args.dataset, args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    # Use a .work subdirectory for intermediate files, final model goes to output_base
    work_dir = output_base / ".work"

    params = parse_params(args)
    params.update({
        "lora_name": lora_name,
        "training_images": str(dataset_path),
        "output_dir": str(work_dir),
        "job_id": lora_name,
    })

    (_project_root / "jobs").mkdir(exist_ok=True)

    # Auto-resume: check for incomplete run
    resume_path = None
    resume_info = _get_resume_info(work_dir, lora_name)
    if resume_info["incomplete"]:
        if resume_info["state_dir"]:
            resume_path = resume_info["state_dir"]
            logger.info(
                "Resuming from step %d/%d (state: %s)",
                resume_info["current_step"], resume_info["total_steps"],
                resume_path.name,
            )
        else:
            logger.warning(
                "Incomplete run (step %d/%d) but no state dir found — restarting",
                resume_info["current_step"], resume_info["total_steps"],
            )

    logger.info("Starting training: %s", lora_name)

    result = run_single_training(params, work_dir, lora_name, resume=resume_path)
    if result["status"] == "completed":
        _copy_final_model(work_dir, output_base, lora_name, {})
        print(f"\n\u2713 Training completed -> {output_base / (lora_name + '.safetensors')}")
    else:
        print(f"\n\u2717 Training {result['status']}: exit code {result.get('exit_code')}", file=sys.stderr)
        sys.exit(1)


def _run_matrix(args, dataset_path, lora_name):
    """Execute a matrix training run (all permutations)."""
    output_base = build_output_dir(args.dataset, args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    all_param_ranges = parse_param_ranges(args, include_single=True)
    param_ranges = {k: v for k, v in all_param_ranges.items() if len(v) > 1}
    if not param_ranges:
        print("ERROR: Matrix mode requires at least one parameter with multiple values (comma-separated).", file=sys.stderr)
        sys.exit(1)

    permutations = generate_permutations(param_ranges)


    base_params = {
        "network_dim": int(all_param_ranges["network_dim"][0]),
        "network_alpha": float(all_param_ranges["network_alpha"][0]),
        "learning_rate": float(all_param_ranges["learning_rate"][0]),
        "batch_size": int(all_param_ranges["batch_size"][0]),
        "max_steps": int(all_param_ranges["max_steps"][0]),
        "optimizer": str(all_param_ranges["optimizer"][0]),
        "scheduler": str(all_param_ranges["scheduler"][0]),
        "resolution": int(all_param_ranges["resolution"][0]),
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
    base_params.update({
        "lora_name": lora_name,
        "training_images": str(dataset_path),
        "job_id": "matrix",
    })

    manifest_path = output_base / "manifest.json"

    # Load existing manifest if resuming or if one exists (auto-resume)
    existing_manifest = None
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if existing_manifest is not None:
        manifest = existing_manifest
        logger.info("Loaded existing manifest: %d completed, %d failed, %d total",
                     manifest.get("completed", 0), manifest.get("failed", 0), manifest.get("total", 0))
    else:
        manifest = {
            "mode": "matrix",
            "param_ranges": {k: list(v) for k, v in param_ranges.items()},
            "total": len(permutations),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "permutations": [
                {"params": perm, "status": "pending", "output_dir": None, "error": None}
                for perm in permutations
            ],
        }

    (_project_root / "jobs").mkdir(exist_ok=True)

    # Count already-completed from existing manifest
    completed = manifest.get("completed", 0)
    failed = manifest.get("failed", 0)
    cancelled = manifest.get("cancelled", 0)

    # Build a set of completed perm names for quick lookup
    completed_perms = set()
    for entry in manifest.get("permutations", []):
        if entry.get("status") == "completed":
            perm_key = "x".join(f"{k}-{v}" for k, v in entry["params"].items())
            completed_perms.add(perm_key)

    # Check for incomplete runs and auto-resume
    has_incomplete = any(
        _is_incomplete(output_base / "x".join(f"{k}-{v}" for k, v in perm.items()))
        for perm in permutations
    )
    if has_incomplete:
        logger.info("Detected incomplete runs — will auto-resume")

    run_idx = 0
    total_to_run = sum(
        1 for perm in permutations
        if "x".join(f"{k}-{v}" for k, v in perm.items()) not in completed_perms
    )

    for idx, perm in enumerate(permutations):
        perm_parts = [f"{k}-{v}" for k, v in perm.items()]
        perm_name = "x".join(perm_parts)
        perm_dir = output_base / perm_name

        # Skip already-completed permutations
        if perm_name in completed_perms:
            manifest["permutations"][idx]["status"] = "completed"
            continue

        # Check for incomplete run and find resume path
        resume_info = _get_resume_info(perm_dir, lora_name)
        resume_path = None
        if resume_info["incomplete"]:
            if resume_info["state_dir"]:
                resume_path = resume_info["state_dir"]
                logger.info(
                    "Resuming %s from step %d/%d (state: %s)",
                    perm_name, resume_info["current_step"],
                    resume_info["total_steps"], resume_path.name,
                )
            else:
                logger.warning(
                    "Incomplete run %s (step %d/%d) but no state dir found — restarting",
                    perm_name, resume_info["current_step"],
                    resume_info["total_steps"],
                )

        run_idx += 1
        print(f"\n[{run_idx}/{total_to_run}] {perm_name}")
        if resume_path:
            print(f"  \u21bb Resuming from {resume_path.name}")

        run_params = {**base_params, **perm}
        run_params["output_dir"] = str(perm_dir)

        manifest["permutations"][idx]["status"] = "running"
        manifest["permutations"][idx]["output_dir"] = str(perm_dir)
        _write_json(manifest_path, manifest)

        try:
            result = run_single_training(run_params, perm_dir, f"matrix-{idx}", resume=resume_path)
            if result["status"] == "completed":
                manifest["permutations"][idx]["status"] = "completed"
                manifest["completed"] += 1
                completed += 1
                _copy_final_model(perm_dir, output_base, lora_name, perm)
                print(f"  \u2713 {perm_name}")
            else:
                err_detail = result.get("error") or f"exit code {result.get('exit_code', 'unknown')}"
                error = f"exit code {result.get('exit_code', 'unknown')}"
                if err_detail != error:
                    error += f" — {err_detail}"
                manifest["permutations"][idx]["status"] = "failed"
                manifest["permutations"][idx]["error"] = error
                manifest["failed"] += 1
                failed += 1
                print(f"  \u2717 {perm_name} — {error}", file=sys.stderr)
        except Exception as e:
            manifest["permutations"][idx]["status"] = "failed"
            manifest["permutations"][idx]["error"] = str(e)
            manifest["failed"] += 1
            failed += 1
            print(f"  \u2717 {perm_name} — {e}", file=sys.stderr)

        _write_json(manifest_path, manifest)

    print(f"\nMatrix finished: {completed} completed, {failed} failed")
    print(f"Output: {output_base}")

    if failed > 0 and completed == 0:
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate mode
    if args.validate:
        dataset = ensure_dataset(args)
        # Handle comma-separated matrix params — use first value for validation
        _first_int = lambda v: int(str(v).split(",")[0])
        valid, warnings = validate_dataset(
            dataset,
            batch_size=_first_int(args.batch_size),
        )
        if warnings:
            print(f"\n  ({len(warnings)} warning(s) — training is still allowed)")
        sys.exit(0 if valid else 1)

    dataset = ensure_dataset(args)

    # Training gate: check validation marker
    if not check_validation(dataset):
        print("ERROR: Dataset has not been validated.", file=sys.stderr)
        print(f"  Run: uv run python scripts/train.py --validate --dataset {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Resolve lora name
    dataset_path = Path(dataset).resolve()
    if args.name:
        lora_name = args.name
    else:
        name = dataset_path.name
        if name == "img":
            name = dataset_path.parent.name
        lora_name = name

    if args.mode == "single":
        _run_single(args, dataset_path, lora_name)
    else:
        _run_matrix(args, dataset_path, lora_name)


if __name__ == "__main__":
    main()
