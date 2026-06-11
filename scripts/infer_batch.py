"""Ad-hoc batch inference script — scan a directory tree for LoRAs and run test prompts.

Usage examples:

  # Run inference on all .safetensors files under a folder:
  uv run python scripts/infer_batch.py --dir output/job-123 --prompt "masterpiece, 1girl, solo"

  # Auto-generate prompt from training data tags (detects trigger word automatically):
  uv run python scripts/infer_batch.py --dir output/ --training-images datasets/mari/img

  # Custom settings:
  uv run python scripts/infer_batch.py --dir output/ --prompt "masterpiece, 1girl" --seed 1234 --steps 40

For each LoRA, generates 3 preview images at different aspect ratios:
  - 768x1280 (portrait)
  - 1280x1280 (square)
  - 1280x768 (landscape)
"""

import argparse
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.prompt_generator import generate_test_prompt
from scripts.tag_extractor import extract_tags

# Default model paths for sd-cli inference
_DEFAULT_MODELS = {
    "diffusion_model": "models/diffusion_model/anima-base-v1.0.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
    "text_encoder": "models/text_encoder/qwen_3_06b_base.safetensors",
}

# Anima-optimized inference defaults
_NEGATIVE_PROMPT = "worst quality, low quality, blurry, bad anatomy, deformed hands"
_DEFAULT_CFG = "4.0"
_DEFAULT_STEPS = "30"
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"

# Preview resolutions: (width, height)
PREVIEW_RESOLUTIONS = [
    (768, 1280),   # portrait
    (1280, 1280),  # square
    (1280, 768),   # landscape
]

# Tags to exclude when detecting the dataset trigger word
_TRIGGER_EXCLUSIONS = {"1girl", "solo"}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def find_loras(directory: str) -> list[Path]:
    """Recursively find all .safetensors files in a directory."""
    return sorted(Path(directory).rglob("*.safetensors"))


def find_trigger_word(directory: str) -> str | None:
    """Find the tag that appears in every caption file (the dataset trigger word).

    Excludes common generic tags like '1girl' and 'solo'.
    Returns None if no trigger word can be determined.
    """
    txt_files = sorted(Path(directory).glob("*.txt"))
    if not txt_files:
        return None

    # Build per-file tag sets
    all_file_tags: list[set[str]] = []
    for f in txt_files:
        content = f.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        tags = {t.strip().lower() for t in content.split(",") if t.strip()}
        all_file_tags.append(tags)

    if len(all_file_tags) < 2:
        return None

    # Intersection of all caption tag sets
    common = all_file_tags[0]
    for tags in all_file_tags[1:]:
        common = common & tags

    # Filter out generic tags
    trigger = common - _TRIGGER_EXCLUSIONS

    if len(trigger) == 1:
        return trigger.pop()
    elif len(trigger) > 1:
        # Multiple candidates — pick the longest (most specific)
        return sorted(trigger, key=len, reverse=True)[0]

    return None


def run_inference(
    lora_path: Path,
    prompt: str,
    diffusion_model: str,
    vae_model: str,
    llm_model: str,
    seed: int,
    steps: int,
    cfg_scale: str,
    width: int,
    height: int,
) -> Path | None:
    """Run sd-cli inference for a single LoRA at a specific resolution.

    Returns the output file path on success, or None on failure.
    """
    lora_name = lora_path.stem
    lora_dir = lora_path.parent.resolve()
    sample_dir = lora_dir / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    output_path = sample_dir / f"{lora_name}_{width}x{height}.png"

    lora_prompt = f"<lora:{lora_name}:1> {prompt}"

    cmd = [
        "sd-cli",
        "--diffusion-model", diffusion_model,
        "--vae", vae_model,
        "--llm", llm_model,
        "--lora-model-dir", str(lora_dir),
        "--prompt", lora_prompt,
        "--negative-prompt", _NEGATIVE_PROMPT,
        "--cfg-scale", cfg_scale,
        "--sampling-method", _DEFAULT_SAMPLER,
        "--scheduler", _DEFAULT_SCHEDULER,
        "--steps", str(steps),
        "--diffusion-fa",
        "--offload-to-cpu",
        "-s", str(seed),
        "-W", str(width),
        "-H", str(height),
        "-o", str(output_path),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            cmd,
            cwd=_project_root,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            stderr_preview = (result.stderr or result.stdout or "no output")[:200]
            logger.warning(f"  Inference failed (exit {result.returncode}): {stderr_preview}")
            return None

        if output_path.exists():
            return output_path
        else:
            logger.warning(f"  sd-cli exited OK but output file not found: {output_path}")
            return None

    except FileNotFoundError:
        logger.error("sd-cli not found — install it first")
        return None
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run test-prompt inference on all LoRAs in a directory tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", "-d",
        required=True,
        help="Directory to scan recursively for .safetensors files",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Test prompt (comma-separated tags). If omitted, auto-generated from --training-images.",
    )
    parser.add_argument(
        "--training-images", "-t",
        default=None,
        help="Directory with training images + .txt captions (used to auto-generate prompt and detect trigger word)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for inference (default: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps (default: 30)",
    )
    parser.add_argument(
        "--cfg-scale",
        default=_DEFAULT_CFG,
        help=f"CFG scale (default: {_DEFAULT_CFG})",
    )
    parser.add_argument(
        "--diffusion-model",
        default=_DEFAULT_MODELS["diffusion_model"],
        help="Path to diffusion model",
    )
    parser.add_argument(
        "--vae",
        default=_DEFAULT_MODELS["vae"],
        help="Path to VAE model",
    )
    parser.add_argument(
        "--llm",
        default=_DEFAULT_MODELS["text_encoder"],
        help="Path to text encoder / LLM model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List LoRAs without running inference",
    )
    args = parser.parse_args()

    # Detect trigger word from training data
    trigger_word = None
    if args.training_images:
        trigger_word = find_trigger_word(args.training_images)
        if trigger_word:
            logger.info(f"Detected trigger word: {trigger_word}")

    # Resolve prompt
    prompt = args.prompt
    if not prompt and args.training_images:
        tags = extract_tags(args.training_images)
        prompt = generate_test_prompt(tags, num_tags=10, seed=args.seed)
        logger.info(f"Extracted {len(tags)} tags, generated prompt: {prompt}")
    elif not prompt:
        prompt = "masterpiece"
        logger.info("No prompt or training images — using default: masterpiece")

    # Build final prompt: masterpiece, <trigger>, <tags>
    if trigger_word:
        # Strip leading "masterpiece" if present, then reassemble
        tag_body = prompt.removeprefix("masterpiece, ").removeprefix("masterpiece").strip()
        if tag_body.startswith(", "):
            tag_body = tag_body[2:]
        prompt = f"masterpiece, {trigger_word}, {tag_body}"
        logger.info(f"Full prompt (with trigger): {prompt}")

    # Find LoRAs
    loras = find_loras(args.dir)
    if not loras:
        logger.error(f"No .safetensors files found in {args.dir}")
        sys.exit(1)

    logger.info(f"Found {len(loras)} LoRA(s) in {args.dir}")

    if args.dry_run:
        for lora in loras:
            print(lora)
        return

    # Run inference at all resolutions
    total = len(loras) * len(PREVIEW_RESOLUTIONS)
    success = 0
    failed = 0
    start_time = time.time()

    for lora_idx, lora in enumerate(loras, 1):
        logger.info(f"[{lora_idx}/{len(loras)}] {lora.name}")
        logger.info(f"  prompt: <lora:{lora.stem}:1> {prompt}")

        for res_idx, (width, height) in enumerate(PREVIEW_RESOLUTIONS, 1):
            logger.info(f"  [{res_idx}/{len(PREVIEW_RESOLUTIONS)}] {width}x{height}")
            output = run_inference(
                lora_path=lora,
                prompt=prompt,
                diffusion_model=args.diffusion_model,
                vae_model=args.vae,
                llm_model=args.llm,
                seed=args.seed,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                width=width,
                height=height,
            )
            if output:
                success += 1
                logger.info(f"    OK — {output.name}")
            else:
                failed += 1
                logger.warning(f"    FAILED")

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"Done: {success} succeeded, {failed} failed out of {total} ({elapsed}s)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
