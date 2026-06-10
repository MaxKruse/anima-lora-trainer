"""Single training script — generates dataset TOML, builds command, launches training.

Streams training output to a log file and parses progress (step, epoch, loss)
from kohya-ss tqdm output, writing updates to the job manifest.
"""

import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so `import scripts.X` works
# when run as `python scripts/train_single.py` (not `python -m scripts.train_single`)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Global flag for cancellation
_cancel_requested = False

def check_cancel_signal(job_id: str) -> bool:
    """Check if a cancel signal file exists for this job."""
    signal_path = Path(_project_root) / "jobs" / f"{job_id}.cancel"
    return signal_path.exists()

from scripts.command_builder import build_training_command
from scripts.dataset_toml import generate_dataset_toml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# CamelCase -> snake_case converter (TS sends camelCase, Python expects snake_case)
def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _normalize_params(params: dict) -> dict:
    return {_camel_to_snake(k): v for k, v in params.items()}


# Regex to parse tqdm progress line from kohya-ss output.
# Example: steps:  45%|████▌     | 450/1000 [02:15<02:45, 3.35it/s, avr_loss=0.123]
_TQDM_RE = re.compile(
    r"steps:\s+(\d+)%.*?\|\s+(\d+)/(\d+)\s+.*?avr_loss=([\d.]+)"
)
# Also match partial lines without avr_loss
_TQDM_RE_MINIMAL = re.compile(
    r"steps:\s+(\d+)%.*?\|\s+(\d+)/(\d+)"
)
# Epoch transition: "epoch 1/10"
_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)")


class TrainingProgress:
    """Thread-safe container for training progress state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.avg_loss = None
        self.status = "running"  # running | completed | failed
        self.error = None
        self.exit_code = None

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "avg_loss": self.avg_loss,
                "error": self.error,
                "exit_code": self.exit_code,
            }

    def update_from_tqdm(self, percent: int, current: int, total: int, loss: float | None = None):
        with self.lock:
            self.current_step = current
            self.total_steps = total
            if loss is not None:
                self.avg_loss = round(loss, 6)

    def update_epoch(self, current: int, total: int):
        with self.lock:
            self.current_epoch = current
            self.total_epochs = total

    def mark_completed(self, exit_code: int):
        with self.lock:
            self.status = "completed"
            self.exit_code = exit_code

    def mark_failed(self, exit_code: int | None, error: str | None):
        with self.lock:
            self.status = "failed"
            self.exit_code = exit_code
            self.error = error


def _count_images(image_dir: str) -> int:
    """Count image files in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    count = 0
    try:
        for entry in os.listdir(image_dir):
            if Path(entry).suffix.lower() in image_extensions:
                count += 1
    except FileNotFoundError:
        pass
    return max(count, 1)


def _write_manifest(path: Path, data: dict) -> None:
    """Write job manifest as JSON (atomic write via temp file)."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def _parse_and_write_progress(progress: TrainingProgress, manifest_path: Path):
    """Write current progress to manifest file."""
    _write_manifest(manifest_path, progress.to_dict())


# Default model paths for sd-cli inference (mirror command_builder.py MODEL_PATHS)
_INFERENCE_MODELS = {
    "diffusion_model": "models/diffusion_model/anima-base-v1.0.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
    "text_encoder": "models/text_encoder/qwen_3_06b_base.safetensors",
}


def run_inference(
    output_dir: Path,
    test_prompt: str,
    diffusion_model: str | None = None,
    vae_model: str | None = None,
    llm_model: str | None = None,
    seed: int = 42,
) -> bool:
    """Run sd-cli inference with the trained LoRA to generate sample images.

    Finds the latest .safetensors checkpoint in output_dir, runs sd-cli
    inference, and writes output to output_dir/sample_images/.

    Args:
        output_dir: Training output directory (contains .safetensors).
        test_prompt: Prompt text for inference.
        diffusion_model: Path to base diffusion model (default from config).
        vae_model: Path to VAE model (default from config).
        llm_model: Path to text encoder / LLM model (default from config).
        seed: Random seed for deterministic generation.

    Returns:
        True if inference succeeded and produced at least one image.
    """
    # Find the latest LoRA checkpoint
    safetensors_files = sorted(output_dir.glob("*.safetensors"), key=lambda p: p.name)
    if not safetensors_files:
        logger.warning("No .safetensors file found — skipping inference")
        return False

    lora_path = safetensors_files[-1]
    lora_name = lora_path.stem  # filename without extension

    # Create sample_images output directory
    sample_dir = output_dir / "sample_images"
    sample_dir.mkdir(exist_ok=True)

    # Build lora prompt: <lora:filename:1> test_prompt
    lora_prompt = f"<lora:{lora_name}:1> {test_prompt}"

    # Anima-specific inference settings (per Civitai developer docs + kohya-ss defaults):
    #   cfg-scale 3-5 (sweet spot 4), steps 25-35 (default 30), euler sampler, simple schedule
    NEGATIVE_PROMPT = "worst quality, low quality, blurry, bad anatomy, deformed hands"

    cmd = [
        "sd-cli",
        "--model", diffusion_model or _INFERENCE_MODELS["diffusion_model"],
        "--vae", vae_model or _INFERENCE_MODELS["vae"],
        "--llm", llm_model or _INFERENCE_MODELS["text_encoder"],
        "--lora-model-dir", str(output_dir),
        "--prompt", lora_prompt,
        "--negative-prompt", NEGATIVE_PROMPT,
        "--cfg-scale", "4.0",
        "--sampling-method", "euler",
        "--scheduler", "simple",
        "--steps", "30",
        "--diffusion-fa",
        "--offload-to-cpu",
        "-s", str(seed),
        "-o", str(sample_dir),
    ]

    logger.info(f"Running inference: sd-cli ... (prompt: {lora_prompt[:80]}...)")

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
            logger.warning(f"sd-cli inference failed (exit {result.returncode}): {stderr_preview}")
            return False

        # Check that at least one image was produced
        images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.webp"))
        if images:
            logger.info(f"Inference complete: {len(images)} image(s) in sample_images/")
            return True
        else:
            logger.warning("sd-cli exited successfully but produced no images")
            return False

    except FileNotFoundError:
        logger.warning("sd-cli not found — skipping inference (install with: pip install sd-cli or similar)")
        return False
    except Exception as e:
        logger.warning(f"Inference failed with exception: {e}")
        return False


def run_training(params: dict) -> dict:
    """Run a single training job with streaming output and progress tracking.

    Args:
        params: Training parameters dict (snake_case).

    Returns:
        dict with status and output_dir.
    """
    output_dir = Path(params["output_dir"])
    job_id = params.get("job_id", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "job_manifest.json"
    log_path = output_dir / "training.log"

    progress = TrainingProgress()

    # Write initial manifest
    _write_manifest(manifest_path, {
        "status": "running",
        "params": {k: v for k, v in params.items() if k != "output_dir"},
        "output_dir": str(output_dir),
        "log_file": str(log_path),
    })

    try:
        # Step 1: Generate dataset TOML
        dataset_toml_path = output_dir / "dataset.toml"
        num_images = _count_images(params["training_images"])

        user_repeats = params.get("repeats")
        if user_repeats is not None:
            num_repeats = user_repeats
        else:
            steps_per_epoch = max(100, num_images)
            num_repeats = max(1, -(-steps_per_epoch // num_images))

        generate_dataset_toml(
            image_dir=params["training_images"],
            batch_size=params["batch_size"],
            num_images=num_images,
            epochs=params["epochs"],
            num_repeats=num_repeats,
            output_path=str(dataset_toml_path),
            resolution=params.get("resolution", 1024),
            cache_text_encoder_outputs=params.get("cache_text_encoder", False),
            caption_tag_dropout_rate=params.get("caption_tag_dropout_rate", 0.05),
            keep_tokens=params.get("keep_tokens", 1),
        )
        logger.info(f"Dataset TOML written to {dataset_toml_path}")

        # Step 2: Build training command
        params["dataset_config"] = str(dataset_toml_path)
        cmd = build_training_command(params)

        # Step 3: Launch training via `uv run` + accelerate
        full_cmd = ["uv", "run"] + cmd
        logger.info(f"Launching training: {' '.join(full_cmd[:8])}...")

        # Use PIPE for stdout so we can stream to file + parse progress
        # Set PYTHONIOENCODING=utf-8 so kohya-ss Japanese log messages don't crash on Windows cp1252
        # Set PYTHONUNBUFFERED=1 so tqdm progress lines flush immediately (not buffered by Python)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            full_cmd,
            cwd=_project_root,  # Must be project root so sd-scripts/ is found
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env,
        )

        def _stream_and_parse():
            """Read from process stdout, write to log file, parse progress."""
            last_flush = time.time()
            try:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    for raw_line in iter(proc.stdout.readline, b""):
                        # Check for cancel signal
                        if job_id and check_cancel_signal(job_id):
                            logger.info("Cancel signal received — terminating training")
                            proc.terminate()
                            break

                        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                        log_file.write(line + "\n")
                        log_file.flush()

                        if not line.strip():
                            continue

                        # Parse tqdm progress
                        m = _TQDM_RE.search(line)
                        if m:
                            pct, cur, tot, loss = m.groups()
                            progress.update_from_tqdm(
                                int(pct), int(cur), int(tot), float(loss)
                            )
                        else:
                            m = _TQDM_RE_MINIMAL.search(line)
                            if m:
                                pct, cur, tot = m.groups()
                                progress.update_from_tqdm(
                                    int(pct), int(cur), int(tot)
                                )

                        # Parse epoch transitions
                        m = _EPOCH_RE.search(line)
                        if m:
                            cur_ep, tot_ep = m.groups()
                            progress.update_epoch(int(cur_ep), int(tot_ep))

                        # Write manifest every 5 seconds
                        now = time.time()
                        if now - last_flush >= 5:
                            _parse_and_write_progress(progress, manifest_path)
                            last_flush = now
            finally:
                # Final manifest write
                _parse_and_write_progress(progress, manifest_path)

        stream_thread = threading.Thread(target=_stream_and_parse, daemon=True)
        stream_thread.start()

        # Wait for process to complete, checking for cancel signal periodically
        cancelled = False
        while proc.poll() is None:
            if job_id and check_cancel_signal(job_id):
                logger.info("Cancel signal detected — terminating training")
                proc.terminate()
                cancelled = True
                break
            time.sleep(1)

        exit_code = proc.wait() if not cancelled else proc.returncode
        stream_thread.join(timeout=5)

        if cancelled:
            progress.mark_failed(exit_code, "Cancelled by user")
        elif exit_code == 0:
            progress.mark_completed(exit_code)
        else:
            progress.mark_failed(exit_code, f"Training exited with code {exit_code}")

        # Final manifest write
        _parse_and_write_progress(progress, manifest_path)

        # Run inference with test prompt (if training succeeded and prompt provided)
        test_prompt = params.get("test_prompt")
        if progress.status == "completed" and test_prompt:
            run_inference(
                output_dir=output_dir,
                test_prompt=test_prompt,
                seed=params.get("test_seed", 42),
            )

        return {
            "status": progress.status,
            "output_dir": str(output_dir),
            "exit_code": progress.exit_code,
        }

    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        progress.mark_failed(None, str(e))
        _parse_and_write_progress(progress, manifest_path)
        return {
            "status": "failed",
            "error": str(e),
        }


def main():
    """CLI entry point: python train_single.py --params-json-file <path>"""
    import argparse

    parser = argparse.ArgumentParser(description="Run single LoRA training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--params-json", help="JSON string of training params")
    group.add_argument("--params-json-file", help="Path to JSON file of training params")
    args = parser.parse_args()

    if args.params_json_file:
        with open(args.params_json_file, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = json.loads(args.params_json)

    # Normalize camelCase (from TS) -> snake_case (expected by Python)
    params = _normalize_params(params)

    result = run_training(params)

    if result["status"] == "completed":
        print(f"Training completed: {result['output_dir']}")
    else:
        print(f"Training failed: {result.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
