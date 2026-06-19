"""Post-training evaluation using sd.cpp (sd-cli).

Generates inference images at multiple resolutions for every LoRA file
produced during training (checkpoints + final model).
"""

import json
import logging
import random
import subprocess
from pathlib import Path

from scripts.constants import MODEL_PATHS, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Resolutions to evaluate at: (width, height)
EVAL_RESOLUTIONS = [
    (768, 1280),
    (1280, 1280),
    (1344, 768),
]

# Default config values (used when key missing or empty in config file)
_EVAL_DEFAULTS = {
    "negative_prompt": (
        "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, "
        "cropped, worst quality, low quality"
    ),
    "steps": 20,
    "cfg_scale": 7.0,
    "seed": 42,
    "sampler": "euler_a",
}

# ── Config loading ───────────────────────────────────────────────────────


def _write_default_config(path: Path) -> None:
    """Write a default eval config file and alert the user."""
    default_data = {
        "model": "",
        "encoder": "",
        "vae": "",
        "negative_prompt": _EVAL_DEFAULTS["negative_prompt"],
        "steps": _EVAL_DEFAULTS["steps"],
        "cfg_scale": _EVAL_DEFAULTS["cfg_scale"],
        "seed": _EVAL_DEFAULTS["seed"],
        "sampler": _EVAL_DEFAULTS["sampler"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(default_data, indent=2) + "\n")
    print(f"  [eval] Created default config: {path}")
    print(f"  [eval] Edit it to customize model paths, steps, CFG, seed, etc.")


def load_eval_config(config_path: str | None) -> dict:
    """Load eval config from JSON, auto-generating if missing.

    Args:
        config_path: Path to JSON config file. If None, uses project default.

    Returns:
        Dict with keys: model, encoder, vae, negative_prompt, steps, cfg_scale, seed, sampler
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "eval.config.json")

    path = Path(config_path)

    # Auto-generate if missing
    if not path.exists():
        _write_default_config(path)

    config = dict(_EVAL_DEFAULTS)

    try:
        user_config = json.loads(path.read_text())
        config.update(user_config)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load eval config %s: %s — using defaults", path, e)

    # Apply model path defaults for empty strings
    config["model"] = config.get("model", "") or MODEL_PATHS["diffusion_model"]
    config["encoder"] = config.get("encoder", "") or MODEL_PATHS["text_encoder"]
    config["vae"] = config.get("vae", "") or MODEL_PATHS["vae"]

    return config


# ── Caption selection ────────────────────────────────────────────────────


def pick_caption(dataset_dir: str) -> str:
    """Pick a random caption from the dataset.

    Scans all .txt files in the dataset directory (root + immediate subdirs),
    picks one at random, and returns its content as a comma-separated tag string.

    Args:
        dataset_dir: Path to the dataset img/ directory.

    Returns:
        Caption text (comma-separated tags).

    Raises:
        ValueError: If no caption files found.
    """
    root = Path(dataset_dir).resolve()
    captions: list[Path] = []

    # Collect from root
    for entry in root.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".txt":
            captions.append(entry)

    # Collect from immediate subdirectories
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            for sub_entry in entry.iterdir():
                if sub_entry.is_file() and sub_entry.suffix.lower() == ".txt":
                    captions.append(sub_entry)

    if not captions:
        raise ValueError(f"No caption (.txt) files found in {dataset_dir}")

    chosen = random.choice(captions)
    content = chosen.read_text().strip()
    logger.info("Selected caption from %s: %s", chosen.name, content[:80] + ("..." if len(content) > 80 else ""))
    return content


# ── LoRA discovery ───────────────────────────────────────────────────────


def discover_lora_files(output_dir: str) -> list[tuple[Path, str]]:
    """Find all .safetensors LoRA files in output directory.

    Returns list of (file_path, prompt_name) tuples. The prompt_name is the
    filename without the .safetensors extension (e.g. 'froot-step00000250').

    Skips files inside -state/ directories.

    Args:
        output_dir: Path to directory containing LoRA files.

    Returns:
        List of (Path, str) tuples, sorted by filename.
    """
    root = Path(output_dir).resolve()
    results: list[tuple[Path, str]] = []

    for entry in root.iterdir():
        # Skip state directories and non-files
        if entry.is_dir():
            continue
        if not entry.is_file() or entry.suffix.lower() != ".safetensors":
            continue
        # Skip files inside -state directories
        if "-state" in str(entry.parent):
            continue

        # Use filename without extension as the LoRA prompt name
        prompt_name = entry.stem

        results.append((entry, prompt_name))

    results.sort(key=lambda x: x[0].name)
    logger.info("Discovered %d LoRA files in %s", len(results), root)
    return results


# ── sd-cli inference ─────────────────────────────────────────────────────


def _build_sd_cli_command(
    config: dict,
    lora_dir: str,
    lora_name: str,
    caption: str,
    width: int,
    height: int,
    output_path: str,
) -> list[str]:
    """Build the sd-cli command argument list.

    Args:
        config: Eval config dict.
        lora_dir: Parent directory of the LoRA file (for --lora-model-dir).
        lora_name: Base name for <lora:Name:1> prompt tag.
        caption: Caption tags string.
        width: Output image width.
        height: Output image height.
        output_path: Full path for the output image.

    Returns:
        List of command-line arguments (excluding the binary name).
    """
    prompt = f"masterpiece, {caption}, <lora:{lora_name}:1>"

    cmd = [
        "sd-cli",
        "--diffusion-model", config["model"],
        "--llm", config["encoder"],
        "--vae", config["vae"],
        "--lora-model-dir", lora_dir,
        "--prompt", prompt,
    ]

    neg = config.get("negative_prompt", "")
    if neg:
        cmd.extend(["--negative-prompt", neg])

    cmd.extend([
        "--width", str(width),
        "--height", str(height),
        "--steps", str(config["steps"]),
        "--cfg-scale", str(config["cfg_scale"]),
        "--seed", str(config["seed"]),
        "--sampling-method", config["sampler"],
        "--output", output_path,
        "--vae-tiling",
        "--offload-to-cpu",
    ])

    return cmd


def run_inference(
    config: dict,
    lora_path: Path,
    lora_name: str,
    caption: str,
    width: int,
    height: int,
    output_dir: Path,
) -> bool:
    """Run a single sd-cli inference pass.

    Args:
        config: Eval config dict.
        lora_path: Full path to the LoRA .safetensors file.
        lora_name: Base name for the <lora:Name:1> prompt tag.
        caption: Caption tags string.
        width: Output image width.
        height: Output image height.
        output_dir: Directory to write the output image into.

    Returns:
        True if the inference succeeded (exit code 0).
    """
    output_path = output_dir / f"{lora_name}-{width}x{height}.png"
    lora_dir = str(lora_path.parent)

    cmd = _build_sd_cli_command(
        config, lora_dir, lora_name, caption, width, height, str(output_path)
    )

    logger.info(
        "Inference: %s @ %dx%d -> %s",
        lora_name, width, height, output_path.name,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per inference
        )
        if result.returncode == 0:
            logger.info("  Generated: %s", output_path)
            return True
        else:
            stderr_preview = (result.stderr or result.stdout or "(no output)")[:300]
            logger.error(
                "  sd-cli failed (exit %d): %s",
                result.returncode,
                stderr_preview,
            )
            return False
    except FileNotFoundError:
        logger.error("  sd-cli not found on PATH. Install sd.cpp and ensure sd-cli is accessible.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("  sd-cli timed out after 600s for %s", lora_name)
        return False
    except OSError as e:
        logger.error("  Failed to run sd-cli: %s", e)
        return False


# ── Main orchestrator ────────────────────────────────────────────────────


def run_evaluation(
    config: dict,
    dataset_dir: str,
    output_dir: str,
    caption: str | None = None,
) -> int:
    """Run full evaluation: pick caption, discover LoRAs, generate images.

    For each LoRA file in output_dir, generates images at all configured
    resolutions using sd-cli. Images are saved to output_dir/samples/.

    Args:
        config: Eval config dict (from load_eval_config).
        dataset_dir: Path to dataset img/ directory (for caption selection).
        output_dir: Path to directory containing LoRA .safetensors files.
        caption: Pre-selected caption string. If None, picks one randomly.

    Returns:
        Number of successful inferences.
    """
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Pick caption (use provided or pick randomly)
    if caption is None:
        try:
            caption = pick_caption(dataset_dir)
        except ValueError as e:
            logger.error("Cannot pick caption for evaluation: %s", e)
            return 0

    # Discover LoRA files
    lora_files = discover_lora_files(output_dir)
    if not lora_files:
        logger.warning("No LoRA files found in %s — skipping evaluation", output_dir)
        return 0

    total = len(lora_files) * len(EVAL_RESOLUTIONS)
    success_count = 0

    for lora_path, lora_name in lora_files:
        for width, height in EVAL_RESOLUTIONS:
            ok = run_inference(
                config,
                lora_path,
                lora_name,
                caption,
                width,
                height,
                samples_dir,
            )
            if ok:
                success_count += 1
            else:
                # Remove partial output if it exists
                (samples_dir / f"{lora_name}-{width}x{height}.png").unlink(missing_ok=True)

    logger.info(
        "Evaluation complete: %d/%d images generated in %s",
        success_count, total, samples_dir,
    )
    return success_count
