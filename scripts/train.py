"""LoRA training CLI — character and style modes.

Entry point that orchestrates validation, dataset config generation,
and in-process kohya-ss training.

Usage:
    uv run python scripts/train.py --type character --dataset datasets/emiru/ --validate
    uv run python scripts/train.py --type character --dataset datasets/emiru/ --name Emiru
    uv run python scripts/train.py --type style --dataset datasets/blobcg/ --name BlobCG-Style
"""

import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
)
from scripts.constants import (
    CHARACTER_DEFAULTS,
    IMAGE_EXTENSIONS,
    MODEL_PATHS,
    PROJECT_ROOT,
    STYLE_DEFAULTS,
)
from scripts.dataset_toml import (
    discover_subsets,
    generate_dataset_toml,
)
from scripts.validation import (
    calculate_max_steps,
    calculate_repeats,
    check_validation,
    get_dataset_img_dir,
    get_dataset_out_dir,
    validate_dataset,
)
from scripts.evaluation import (
    load_eval_config,
    pick_caption,
    run_evaluation,
)
from scripts.training_chart import (
    generate_png_chart,
    parse_tensorboard_events,
    print_training_summary,
)
from scripts.swap_metadata import swap_metadata_on_all
from scripts.zip_training_data import zip_training_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
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


def build_output_dir(dataset_dir: str, output_base: str | None) -> Path:
    """Build the output directory path.

    Default: <dataset>/out/
    Override with --output flag.
    """
    if output_base:
        return Path(output_base)
    return get_dataset_out_dir(dataset_dir)


def _build_cli_command() -> str:
    """Reconstruct the exact CLI command used to start this run."""
    script = sys.argv[0] if sys.argv else "scripts/train.py"
    return " ".join(f'"{arg}"' if " " in arg else arg for arg in sys.argv)


def _build_config_from_params(params: dict[str, Any], training_type: str) -> dict[str, Any]:
    """Build the training config dict from effective params."""
    batch_size = params.get("batch_size", 4)
    auto_max = calculate_max_steps(batch_size, training_type)
    effective_steps = params.get("max_steps") or auto_max

    return {
        "training_type": training_type,
        "network_dim": params.get("network_dim"),
        "network_alpha": params.get("network_alpha"),
        "learning_rate": params.get("learning_rate"),
        "batch_size": params.get("batch_size"),
        "max_steps": effective_steps,
        "max_steps_auto": params.get("max_steps") is None,
        "optimizer": params.get("optimizer"),
        "scheduler": params.get("scheduler"),
        "resolution": params.get("resolution"),
        "mixed_precision": params.get("mixed_precision"),
        "timestep_sampling": params.get("timestep_sampling"),
        "gradient_checkpointing": params.get("gradient_checkpointing", True),
        "cache_latents": params.get("cache_latents", True),
        "cache_text_encoder": params.get("cache_text_encoder", False),
        "caption_tag_dropout_rate": params.get("caption_tag_dropout_rate"),
        "keep_tokens": params.get("keep_tokens"),
        "repeats": params.get("repeats"),
        "rebalance_buckets": params.get("rebalance_buckets", True),
        "bucket_dominance_threshold": params.get("bucket_dominance_threshold"),
        "bucket_rebalance_max_aug": params.get("bucket_rebalance_max_aug"),
        "bucket_rebalance_seed": params.get("bucket_rebalance_seed"),
    }


def _write_training_config(
    output_dir: Path,
    lora_name: str,
    params: dict[str, Any],
    training_type: str,
) -> None:
    """Write a reproducibility config artifact (training_config.json)."""
    config: dict[str, Any] = {
        "lora_name": lora_name,
        "training_type": training_type,
        "training_images": params.get("training_images"),
        "output_dir": str(output_dir),
        "cli_command": _build_cli_command(),
        "params": _build_config_from_params(params, training_type),
        "training_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = output_dir / "training_config.json"
    _write_json(config_path, config)
    logger.info("Wrote training config: %s", config_path)


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
    """Find the latest accelerator state directory for resume."""
    if not perm_dir.exists():
        return None

    pattern_prefix = f"{lora_name}-step"
    suffix = "-state"
    state_dirs = [
        d for d in perm_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(pattern_prefix)
        and d.name.endswith(suffix)
    ]
    if state_dirs:
        state_dirs.sort(key=lambda d: d.name)
        return state_dirs[-1]

    final_state = perm_dir / f"{lora_name}-state"
    if final_state.is_dir():
        return final_state

    return None


def _is_incomplete(perm_dir: Path) -> bool:
    """Check if a run is incomplete (has no final model)."""
    manifest = perm_dir / "job_manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
        return data.get("status", "") in ("running", "failed", "cancelled")
    except (json.JSONDecodeError, OSError):
        return False


def _get_resume_info(perm_dir: Path, lora_name: str) -> dict[str, Any]:
    """Get resume info for an incomplete run."""
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
    _callbacks: list[Any] = []
    _cancel_path: Path | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_count = 0

    @classmethod
    def set_callbacks(cls, callbacks: list[Any], cancel_path: Path | None = None) -> None:
        cls._callbacks = callbacks
        cls._cancel_path = cancel_path

    def update(self, n: int = 1) -> Any:
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
_50p_step_global: int = 0
_original_save_fn: Any = None


def _calculate_save_steps(max_steps: int) -> list[int]:
    """Calculate checkpoint save steps at 50% and 75%."""
    percentages = [0.50, 0.75]
    return [max(1, int(round(max_steps * p))) for p in percentages]


def _compute_save_interval(max_steps: int) -> int:
    """Compute a save interval that hits all configured save points."""
    from math import gcd
    save_points = _calculate_save_steps(max_steps)
    interval = max_steps
    for sp in save_points:
        interval = gcd(interval, sp)
    return max(1, interval)


def _cleanup_extra_checkpoints(output_dir: Path, lora_name: str, max_steps: int) -> None:
    """Remove checkpoint files that aren't at configured save points."""
    keep_steps = set(_calculate_save_steps(max_steps)) | {max_steps}
    step_pattern = f"{lora_name}-step"

    for entry in output_dir.iterdir():
        name = entry.name
        if name.endswith("-state") or name == f"{lora_name}.safetensors":
            continue
        if not name.startswith(step_pattern) or not name.endswith(".safetensors"):
            continue
        try:
            step_str = name[len(step_pattern) : -len(".safetensors")]
            step = int(step_str)
            if step not in keep_steps:
                entry.unlink()
                logger.info("Removed extra checkpoint: %s", name)
        except (ValueError, OSError):
            pass


def _setup_save_filter() -> bool:
    """Monkey-patch kohya-ss model save to only save at configured steps."""
    global _original_save_fn

    try:
        from library import anima_train_utils
    except ImportError:
        return False

    if not hasattr(anima_train_utils, "save_anima_model_on_epoch_end_or_stepwise"):
        return False

    _original_save_fn = anima_train_utils.save_anima_model_on_epoch_end_or_stepwise

    def _filtered_save(*args: Any, **kwargs: Any) -> None:
        global _50p_step_global
        global_step = kwargs.get("global_step", args[6] if len(args) > 6 else 0)
        accelerator = kwargs.get("accelerator", args[2] if len(args) > 2 else None)

        if global_step in _save_points:
            _original_save_fn(*args, **kwargs)  # type: ignore[misc]
            if (accelerator is not None
                    and abs(global_step - _50p_step_global) <= 2):
                logger.info("Saving accelerator state at step %d (50%%)", global_step)
                accelerator.save_state()

    anima_train_utils.save_anima_model_on_epoch_end_or_stepwise = _filtered_save
    return True


def _restore_save() -> None:
    """Restore original kohya-ss save function."""
    global _original_save_fn
    if _original_save_fn is not None:
        try:
            from library import anima_train_utils
            anima_train_utils.save_anima_model_on_epoch_end_or_stepwise = _original_save_fn
        except ImportError:
            pass
        _original_save_fn = None


_current_step_global = 0


def _build_kohya_args(params: dict[str, Any], kohya_parser: Any, tb_logs_dir: Path) -> Any:
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
    args.save_every_n_steps = _compute_save_interval(p["max_steps"])
    args.save_state = False
    args.gradient_checkpointing = p.get("gradient_checkpointing", True)
    args.cache_latents = p.get("cache_latents", True)
    args.cache_text_encoder_outputs = p.get("cache_text_encoder", False)
    args.network_train_unet_only = p.get("cache_text_encoder", False)
    args.vae_chunk_size = 64
    args.vae_disable_cache = True
    tb_logs_dir.mkdir(parents=True, exist_ok=True)
    args.logging_dir = str(tb_logs_dir)
    args.log_with = "tensorboard"
    resume_path = p.get("resume")
    if resume_path:
        args.resume = str(resume_path)
    return args


def _reset_kohya_global_state() -> None:
    """Reset kohya-ss global singleton state between in-process training runs."""
    try:
        from library import strategy_base
        strategy_base.TokenizeStrategy._strategy = None
        strategy_base.LatentsCachingStrategy._strategy = None
        strategy_base.TextEncodingStrategy._strategy = None
        strategy_base.TextEncoderOutputsCachingStrategy._strategy = None
    except ImportError:
        pass


def run_single_training(
    params: dict[str, Any],
    output_dir: Path,
    job_id: str,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Run a single training job in-process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "job_manifest.json"

    manifest: dict[str, Any] = {
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

    # Auto-calculate max_steps from batch_size and type
    training_type = params.get("training_type", "character")
    auto_max = calculate_max_steps(batch_size, training_type)
    if params.get("max_steps") is None:
        params["max_steps"] = auto_max

    manual_repeats = params.get("repeats")
    base_repeats = manual_repeats if manual_repeats is not None else calculate_repeats(total_images, batch_size)

    subset_configs: list[dict[str, Any]] = [
        {"image_dir": s["image_dir"], "num_repeats": base_repeats} for s in subsets
    ]

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
        logger.info("Bucket rebalance: using %d rebalanced subsets", len(rebalance_subsets))
        for i, sub in enumerate(rebalance_subsets):
            img_count = _count_images_in_dir(sub["image_dir"])
            logger.info("  subset %d: %s (%d images, repeats=%d)", i, Path(sub["image_dir"]).name, img_count, sub["num_repeats"])
        subset_configs = rebalance_subsets
    else:
        logger.info("Bucket rebalance: skipped \u2014 using original subsets")

    # Generate dataset TOML
    dataset_toml_path = output_dir / "dataset.toml"
    generate_dataset_toml(
        batch_size=params["batch_size"],
        num_images=total_images,
        epochs=4,
        num_repeats=base_repeats,
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
    avg_loss: float | None = None
    error_message: str | None = None
    cancel_path = _project_root / "jobs" / f"{job_id}.cancel"
    tb_logs_dir = output_dir / ".tb_logs"

    def _on_step(step: int, bar: Any) -> None:
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

        args = _build_kohya_args(params, setup_anima_parser(), tb_logs_dir)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["PYTHONIOENCODING"] = "utf-8"

        train_util.verify_command_line_training_args(args)

        _save_points.clear()
        _save_points.update(_calculate_save_steps(params["max_steps"]))
        _50p_step_global = max(1, int(round(params["max_steps"] * 0.5)))
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
        _restore_save()
        _save_points.clear()

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

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        tqdm_module.tqdm = _real_tqdm
        if cancel_path.exists():
            cancel_path.unlink(missing_ok=True)

        m_steps, m_loss, m_lr = parse_tensorboard_events(tb_logs_dir)

        return {
            "status": manifest["status"],
            "output_dir": str(output_dir),
            "exit_code": exit_code,
            "error": error_message,
            "metrics_steps": m_steps,
            "metrics_loss": m_loss,
            "metrics_lr": m_lr,
        }


# ── CLI main ─────────────────────────────────────────────────────────────
def _run_training(args: Any, dataset_path: Path, lora_name: str, training_type: str) -> None:
    """Execute a single training run."""
    output_base = build_output_dir(str(dataset_path), args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Resolve image directory
    img_dir = get_dataset_img_dir(str(dataset_path))

    # Abort if the final model already exists at the output path
    existing_model = output_base / f"{lora_name}.safetensors"
    if existing_model.exists():
        print(f"ERROR: Output model already exists: {existing_model}", file=sys.stderr)
        print(f"  Remove it or use --name / --output to choose a different path.", file=sys.stderr)
        sys.exit(1)

    # Use a .work subdirectory for intermediate files
    work_dir = output_base / ".work"

    params = parse_params(args, training_type)
    params.update({
        "lora_name": lora_name,
        "training_images": str(img_dir),
        "output_dir": str(work_dir),
        "job_id": lora_name,
        "training_type": training_type,
    })

    (_project_root / "jobs").mkdir(exist_ok=True)

    # Auto-resume: check for incomplete run
    resume_path: Path | None = None
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
                "Incomplete run (step %d/%d) but no state dir found \u2014 restarting",
                resume_info["current_step"], resume_info["total_steps"],
            )

    # Print effective params
    effective_steps = params.get("max_steps") or calculate_max_steps(
        params.get("batch_size", 4), training_type
    )
    type_label = "style" if training_type == "style" else "character"
    logger.info(
        "Starting %s training: %s (dim=%d, lr=%.4f, steps=%d, bs=%d)",
        type_label, lora_name,
        params.get("network_dim", 8),
        params.get("learning_rate", 0.0002),
        effective_steps,
        params.get("batch_size", 4),
    )

    result = run_single_training(params, work_dir, lora_name, resume=resume_path)
    if result["status"] == "completed":
        # Swap tag-frequency metadata with clean .tags data (all .safetensors files)
        swap_metadata_on_all(str(work_dir), str(img_dir))

        # Copy final model from work dir to output base
        final_model = work_dir / f"{lora_name}.safetensors"
        if final_model.exists():
            dest = output_base / f"{lora_name}.safetensors"
            dest.write_bytes(final_model.read_bytes())
            logger.info("Copied final model: %s", dest.name)

        print(f"\n\u2713 Training completed -> {output_base / (lora_name + '.safetensors')}")

        # Write reproducibility config artifact
        _write_training_config(output_base, lora_name, params, training_type)

        # Training metrics chart
        _steps = result.get("metrics_steps", [])
        _loss = result.get("metrics_loss", [])
        _lr = result.get("metrics_lr", [])
        if _steps and _loss:
            print_training_summary(_steps, _loss, _lr if _lr else None)
            try:
                png_path = output_base / f"{lora_name}_training_chart.png"
                generate_png_chart(
                    _steps,
                    _loss,
                    _lr if _lr else None,
                    png_path,
                    title=f"Training: {lora_name}",
                )
                print(f"  Chart saved -> {png_path}")
            except ImportError:
                logger.warning(
                    "matplotlib not installed \u2014 skip PNG chart. Install with: uv add matplotlib"
                )

        # Post-training evaluation
        if args.evaluate:
            print(f"\n--- Running evaluation ---")
            eval_config = load_eval_config(args.eval_config)
            run_evaluation(eval_config, str(img_dir), str(work_dir))
    else:
        print(f"\n\u2717 Training {result['status']}: exit code {result.get('exit_code')}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate mode
    if args.validate:
        dataset = ensure_dataset(args)
        _first_bs = int(str(args.batch_size).split(",")[0])
        valid, warnings = validate_dataset(
            dataset,
            batch_size=_first_bs,
        )
        if not valid:
            for w in warnings:
                print(f"  ERROR: {w}", file=sys.stderr)
            sys.exit(1)
        if warnings:
            print(f"\n  ({len(warnings)} warning(s) \u2014 training is still allowed)")
        sys.exit(0)

    dataset = ensure_dataset(args)
    training_type = args.type

    # Training gate: check validation marker
    if not check_validation(dataset):
        print("ERROR: Dataset has not been validated.", file=sys.stderr)
        print(f"  Run: uv run python scripts/train.py --type {training_type} --dataset {args.dataset} --validate", file=sys.stderr)
        sys.exit(1)

    # Resolve lora name
    dataset_path = Path(dataset).resolve()
    if args.name:
        lora_name = args.name
    else:
        lora_name = dataset_path.name

    _run_training(args, dataset_path, lora_name, training_type)


if __name__ == "__main__":
    main()
