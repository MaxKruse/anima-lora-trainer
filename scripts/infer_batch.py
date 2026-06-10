"""Ad-hoc batch inference script — scan a directory tree for LoRAs and run test prompts.

Usage examples:

  # Run inference on all .safetensors files under a folder:
  uv run python scripts/infer_batch.py --dir output/job-123 --prompt "masterpiece, 1girl, solo"

  # Auto-generate prompt from training data tags:
  uv run python scripts/infer_batch.py --dir output/ --training-images datasets/mari/img

  # Custom settings:
  uv run python scripts/infer_batch.py --dir output/ --prompt "masterpiece, 1girl" --seed 1234 --steps 40
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def find_loras(directory: str) -> list[Path]:
    """Recursively find all .safetensors files in a directory."""
    return sorted(Path(directory).rglob("*.safetensors"))


def run_inference_for_lora(
    lora_path: Path,
    prompt: str,
    diffusion_model: str,
    vae_model: str,
    llm_model: str,
    seed: int,
    steps: int,
    cfg_scale: str,
) -> bool:
    """Run sd-cli inference for a single LoRA file.

    Creates sample_images/ in the LoRA's parent directory.
    Returns True if inference succeeded and produced an image.
    """
    lora_name = lora_path.stem  # filename without extension
    lora_dir = lora_path.parent
    sample_dir = lora_dir / "sample_images"
    sample_dir.mkdir(exist_ok=True)

    lora_prompt = f"<lora:{lora_name}:1> {prompt}"

    cmd = [
        "sd-cli",
        "--model", diffusion_model,
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
        "-o", str(sample_dir),
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
            logger.warning(f"Inference failed (exit {result.returncode}): {stderr_preview}")
            return False

        # Check that at least one image was produced
        images = (
            list(sample_dir.glob("*.png"))
            + list(sample_dir.glob("*.jpg"))
            + list(sample_dir.glob("*.webp"))
        )
        if images:
            return True
        else:
            logger.warning("sd-cli exited OK but produced no images")
            return False

    except FileNotFoundError:
        logger.error("sd-cli not found — install it first")
        return False
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return False


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
        help="Directory with training images + .txt captions (used to auto-generate prompt if --prompt not given)",
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

    # Resolve prompt
    prompt = args.prompt
    if not prompt and args.training_images:
        tags = extract_tags(args.training_images)
        prompt = generate_test_prompt(tags, num_tags=10, seed=args.seed)
        logger.info(f"Extracted {len(tags)} tags, generated prompt: {prompt}")
    elif not prompt:
        prompt = "masterpiece"
        logger.info("No prompt or training images — using default: masterpiece")

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

    # Run inference
    total = len(loras)
    success = 0
    failed = 0
    start_time = time.time()

    for idx, lora in enumerate(loras, 1):
        logger.info(f"[{idx}/{total}] {lora.name}")
        ok = run_inference_for_lora(
            lora_path=lora,
            prompt=prompt,
            diffusion_model=args.diffusion_model,
            vae_model=args.vae,
            llm_model=args.llm,
            seed=args.seed,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
        )
        if ok:
            success += 1
            logger.info(f"  OK — sample_images/")
        else:
            failed += 1
            logger.warning(f"  FAILED")

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"Done: {success} succeeded, {failed} failed out of {total} ({elapsed}s)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
