"""Post-training evaluation using sd.cpp (sd-server HTTP API).

Generates inference images at multiple resolutions for every LoRA file
produced during training (checkpoints + final model).

Uses the native sdcpp async API (POST /sdcpp/v1/img_gen + poll) for
minimal overhead: the server keeps the model loaded in VRAM and we
just submit JSON requests.
"""

import base64
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from scripts.constants import MODEL_PATHS, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Resolutions to evaluate at: (width, height)
# Includes Anima-native portrait (832x1216) from Forge Neo preset.
EVAL_RESOLUTIONS = [
    (832, 1216),   # Anima native portrait
    (1024, 1024),  # Square (training resolution)
    (1280, 768),   # Landscape
]

# Default config values (used when key missing or empty in config file)
# Sampler/scheduler/steps/CFG defaults match Forge Neo Anima preset.
_EVAL_DEFAULTS = {
    "sd_server_path": "C:/tools/stable-diffusion/sd-server.exe",
    "server_url": "http://127.0.0.1:1234",
    "listen_port": 1234,
    "negative_prompt": (
        "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, "
        "cropped, worst quality, low quality"
    ),
    "steps": 32,
    "cfg_scale": 4.0,
    "seed": 42,
    "sampler": "er_sde",
    "scheduler": "simple",
    # Hires fix (post-ready upscaling)
    "hires_enabled": True,
    "hires_scale": 1.5,
    "hires_denoising": 0.5,
    "hires_upscaler": "Latent (bicubic antialiased)",
    "poll_interval": 2,           # seconds between job status polls
    "poll_timeout": 600,          # max seconds to wait for a single job
    "server_startup_timeout": 120, # max seconds to wait for server to be ready
}

# ── Config loading ───────────────────────────────────────────────────────


def _write_default_config(path: Path) -> None:
    """Write a default eval config file and alert the user."""
    default_data = {
        "sd_server_path": _EVAL_DEFAULTS["sd_server_path"],
        "server_url": _EVAL_DEFAULTS["server_url"],
        "listen_port": _EVAL_DEFAULTS["listen_port"],
        "diffusion_model": MODEL_PATHS["diffusion_model"],
        "vae": MODEL_PATHS["vae"],
        "encoder": MODEL_PATHS["text_encoder"],
        "negative_prompt": _EVAL_DEFAULTS["negative_prompt"],
        "steps": _EVAL_DEFAULTS["steps"],
        "cfg_scale": _EVAL_DEFAULTS["cfg_scale"],
        "seed": _EVAL_DEFAULTS["seed"],
        "sampler": _EVAL_DEFAULTS["sampler"],
        "scheduler": _EVAL_DEFAULTS["scheduler"],
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
        Dict with keys: sd_server_path, server_url, listen_port, diffusion_model,
        vae, encoder, negative_prompt, steps, cfg_scale, seed, sampler
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

    # Fill in model path defaults for empty/missing strings
    config["diffusion_model"] = config.get("diffusion_model", "") or MODEL_PATHS["diffusion_model"]
    config["vae"] = config.get("vae", "") or MODEL_PATHS["vae"]
    config["encoder"] = config.get("encoder", "") or MODEL_PATHS["text_encoder"]

    # Ensure server_url is set
    if not config.get("server_url"):
        config["server_url"] = _EVAL_DEFAULTS["server_url"]

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


# ── sd-server lifecycle ──────────────────────────────────────────────────

_server_process: subprocess.Popen | None = None


def _build_server_command(config: dict, lora_dirs: list[str]) -> list[str]:
    """Build the sd-server command-line argument list.

    Args:
        config: Eval config dict with model paths.
        lora_dirs: Directories containing LoRA .safetensors files.

    Returns:
        List of command arguments (including the binary path).
    """
    cmd = [
        config["sd_server_path"],
        "--diffusion-model", config["diffusion_model"],
        "--vae", config["vae"],
        "--llm", config["encoder"],
        "--listen-port", str(config.get("listen_port", 1234)),
    ]

    # Add all unique LoRA directories
    for lora_dir in lora_dirs:
        cmd.extend(["--lora-model-dir", lora_dir])

    return cmd


def _collect_lora_dirs(output_dir: Path) -> list[str]:
    """Collect all directories that contain LoRA .safetensors files.

    Scans .work/, permutation subdirs, and the top-level output_dir.

    Args:
        output_dir: Root training output directory.

    Returns:
        List of unique directory paths containing .safetensors files.
    """
    dirs: set[str] = set()

    # .work/ subdirectory
    work_dir = output_dir / ".work"
    if work_dir.is_dir() and any(work_dir.glob("*.safetensors")):
        dirs.add(str(work_dir.resolve()))

    # Permutation subdirectories
    for entry in output_dir.iterdir():
        if entry.is_dir() and entry.name not in (".work", "samples"):
            if any(entry.glob("*.safetensors")):
                dirs.add(str(entry.resolve()))

    # Top-level .safetensors
    if any(output_dir.glob("*.safetensors")):
        dirs.add(str(output_dir.resolve()))

    return sorted(dirs)


def _start_server(config: dict, output_dir: Path) -> subprocess.Popen:
    """Start sd-server as a subprocess.

    Args:
        config: Eval config dict with model paths and sd_server_path.
        output_dir: Training output directory (for LoRA dir discovery).

    Returns:
        Popen handle for the server process.

    Raises:
        RuntimeError: If the server binary is not found.
    """
    lora_dirs = _collect_lora_dirs(output_dir)
    cmd = _build_server_command(config, lora_dirs)

    logger.info("Starting sd-server: %s", cmd[0])
    logger.debug("Full command: %s", " ".join(cmd))

    server_path = Path(config["sd_server_path"])
    if not server_path.exists():
        raise RuntimeError(
            f"sd-server binary not found: {server_path}\n"
            f"Set 'sd_server_path' in eval.config.json or pass --sd-server-path."
        )

    # Start with stdout/stderr logged at DEBUG level
    # Using subprocess.PIPE + reader thread to avoid blocking on full pipe buffer
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    logger.info("sd-server started (pid %d)", proc.pid)

    # Read server output in background thread to avoid pipe buffer blocking
    # and log any error messages
    def _read_server_output():
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.debug("  [sd-server] %s", line)
                    # Also log WARN/ERROR lines at appropriate level
                    lower = line.lower()
                    if any(kw in lower for kw in ("error", "fail", "oom", "out of memory", "cuda error")):
                        logger.warning("  [sd-server ERROR] %s", line)
        except Exception:
            pass

    reader = threading.Thread(target=_read_server_output, daemon=True)
    reader.start()

    return proc


def _wait_for_server_ready(server_url: str, timeout: float) -> bool:
    """Wait for sd-server to respond on /sdcpp/v1/capabilities.

    Args:
        server_url: Base URL of the server.
        timeout: Max seconds to wait.

    Returns:
        True if the server became reachable, False on timeout.
    """
    url = f"{server_url.rstrip('/')}/sdcpp/v1/capabilities"
    deadline = time.time() + timeout
    interval = 2  # poll every 2 seconds

    logger.info("Waiting for sd-server to be ready at %s ...", server_url)

    while time.time() < deadline:
        try:
            _http_get(url)
            logger.info("sd-server is ready")
            return True
        except (urllib_error.URLError, OSError):
            pass
        except Exception:
            pass
        time.sleep(interval)

    logger.error("sd-server did not become ready within %.0fs", timeout)
    return False


def _stop_server(proc: subprocess.Popen | None) -> None:
    """Stop the sd-server subprocess.

    Tries graceful termination first, then force-kill if needed.

    Args:
        proc: Popen handle from _start_server(), or None.
    """
    if proc is None:
        return

    if proc.poll() is not None:
        logger.info("sd-server (pid %d) already exited (code %d)", proc.pid, proc.returncode)
        return

    logger.info("Stopping sd-server (pid %d) ...", proc.pid)
    proc.terminate()

    try:
        proc.wait(timeout=10)
        logger.info("sd-server stopped")
    except subprocess.TimeoutExpired:
        logger.warning("sd-server did not stop gracefully, force-killing")
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("sd-server (pid %d) could not be killed", proc.pid)


# ── sd-server sdcpp API ──────────────────────────────────────────────────


def _http_post(url: str, body: dict) -> bytes:
    """Send a JSON POST request and return raw response bytes."""
    data = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _http_get(url: str) -> bytes:
    """Send a GET request and return raw response bytes."""
    req = urllib_request.Request(url)
    with urllib_request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _check_server(server_url: str) -> bool:
    """Check if the sd-server is reachable and loaded.

    Returns True if the server responds to /sdcpp/v1/capabilities.
    """
    try:
        url = f"{server_url.rstrip('/')}/sdcpp/v1/capabilities"
        _http_get(url)
        logger.debug("sd-server is reachable at %s", server_url)
        return True
    except (urllib_error.URLError, OSError, Exception) as e:
        logger.error("Cannot reach sd-server at %s: %s", server_url, e)
        return False


def _submit_job(server_url: str, payload: dict) -> dict | None:
    """Submit an image generation job via /sdcpp/v1/img_gen.

    Returns the job response dict (with 'id' and 'status'), or None on failure.
    """
    try:
        url = f"{server_url.rstrip('/')}/sdcpp/v1/img_gen"
        raw = _http_post(url, payload)
        response = json.loads(raw)
        logger.debug("  Job submission response: status=%s, id=%s", response.get("status"), response.get("id"))
        return response
    except urllib_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        logger.error("  HTTP %d submitting job: %s", e.code, body)
        return None
    except (urllib_error.URLError, OSError, json.JSONDecodeError) as e:
        logger.error("  Failed to submit job: %s", e)
        return None
    except Exception as e:
        logger.error("  Unexpected error submitting job: %s", e)
        return None


def _poll_job(server_url: str, job_id: str, timeout: float, interval: float) -> dict | None:
    """Poll a job until completion, failure, or timeout.

    Returns the full job dict on completion, or None on failure/timeout.
    """
    url = f"{server_url.rstrip('/')}/sdcpp/v1/jobs/{job_id}"
    deadline = time.time() + timeout
    last_status = None
    status_interval = 15  # log status every N seconds
    next_status_log = time.time() + status_interval

    while time.time() < deadline:
        try:
            raw = _http_get(url)
            job = json.loads(raw)
            status = job.get("status")
            queue_pos = job.get("queue_position")

            if status == "completed":
                return job
            elif status == "failed":
                error = job.get("error", {})
                logger.error(
                    "  Job %s failed: %s",
                    job_id,
                    error.get("message", "unknown error"),
                )
                return None
            elif status == "cancelled":
                logger.warning("  Job %s was cancelled", job_id)
                return None
            else:
                # queued / generating — log progress periodically
                elapsed = time.time() - (deadline - timeout)
                remaining = max(0, round(deadline - time.time()))
                if time.time() >= next_status_log:
                    pos_str = f", position {queue_pos}" if queue_pos else ""
                    logger.info(
                        "  Job %s: status=%s%s (elapsed %.0fs, remaining %.0fs)",
                        job_id, status, pos_str, elapsed, remaining,
                    )
                    next_status_log = time.time() + status_interval
                # Detect stuck queue (status unchanged for too long)
                if status == "queued" and last_status == "queued":
                    pass  # continue logging via periodic status above
                last_status = status
        except (urllib_error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning("  Poll error for job %s: %s", job_id, e)

        time.sleep(interval)

    logger.error("  Job %s timed out after %.0fs (final status: %s)", job_id, timeout, last_status)
    return None


def _build_hires_config(config: dict, width: int, height: int) -> dict:
    """Build the hires (highres fix) section of the sdcpp payload.

    Produces post-ready images by upscaling the base generation.
    Target dimensions are rounded to multiples of 64.

    Args:
        config: Eval config dict with hires_* keys.
        width: Base generation width.
        height: Base generation height.

    Returns:
        Dict matching the hires schema of the sdcpp API.
    """
    if not config.get("hires_enabled", True):
        return {"enabled": False}

    scale = config.get("hires_scale", 1.5)
    target_w = round(width * scale / 64) * 64
    target_h = round(height * scale / 64) * 64

    return {
        "enabled": True,
        "upscaler": config.get("hires_upscaler", "Latent (bicubic antialiased)"),
        "scale": scale,
        "target_width": target_w,
        "target_height": target_h,
        "steps": 0,          # 0 = reuse base steps
        "denoising_strength": config.get("hires_denoising", 0.5),
        "custom_sigmas": [],
        "upscale_tile_size": 128,
    }


def _get_server_loras(server_url: str) -> list[dict]:
    """Query the server's known LoRAs from /sdcpp/v1/capabilities.

    Returns list of LoRA dicts from the capabilities endpoint.
    """
    try:
        url = f"{server_url.rstrip('/')}/sdcpp/v1/capabilities"
        raw = _http_get(url)
        caps = json.loads(raw)
        return caps.get("loras", [])
    except Exception as e:
        logger.warning("  Failed to query server capabilities: %s", e)
        return []


def _resolve_lora_path(lora_path: Path, server_url: str) -> str | None:
    """Resolve the LoRA path to the format sd-server expects.

    Tries to match the local file against the server's known LoRAs
    (from /sdcpp/v1/capabilities). Falls back to the filename.

    Returns the path string to use in the payload, or None if not found.
    """
    server_loras = _get_server_loras(server_url)
    if not server_loras:
        logger.warning("  Server reported no LoRAs in capabilities")
        return lora_path.name

    # Try matching by filename (with and without extension)
    filename = lora_path.name
    stem = lora_path.stem

    for slora in server_loras:
        slora_path = slora.get("path", "")
        slora_name = slora.get("name", "")
        # Match against server's path or name
        if slora_path == filename or slora_path == stem:
            logger.debug("  LoRA path resolved via server 'path': %s", slora_path)
            return slora_path
        if slora_name == filename or slora_name == stem:
            logger.debug("  LoRA path resolved via server 'name': %s", slora_name)
            return slora_path if slora_path else slora_name

    # Fallback: try the filename as-is
    logger.warning(
        "  LoRA '%s' not found in server capabilities. "
        "Server knows: %s",
        filename,
        [l.get("path", l.get("name", "?")) for l in server_loras],
    )
    return filename


def _build_sdcpp_payload(
    config: dict,
    lora_path: Path,
    caption: str,
    width: int,
    height: int,
    server_url: str | None = None,
) -> dict:
    """Build the sdcpp img_gen request payload.

    Args:
        config: Eval config dict.
        lora_path: Full path to the LoRA .safetensors file.
        caption: Caption tags string.
        width: Output image width.
        height: Output image height.
        server_url: Server URL for LoRA path resolution (optional).

    Returns:
        Dict matching the sdcpp API request body schema.
    """
    prompt = f"masterpiece, {caption}"

    # Resolve LoRA path to match what the server expects
    if server_url:
        lora_path_str = _resolve_lora_path(lora_path, server_url)
    else:
        lora_path_str = lora_path.name

    payload = {
        "prompt": prompt,
        "negative_prompt": config.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "seed": config.get("seed", 42),
        "batch_count": 1,
        "sample_params": {
            "sample_method": config.get("sampler", "er_sde"),
            "scheduler": config.get("scheduler", "simple"),
            "sample_steps": config.get("steps", 32),
            "guidance": {
                "txt_cfg": config.get("cfg_scale", 4.0),
            },
        },
        "lora": [
            {
                # sd.cpp expects the LoRA filename relative to --lora-model-dir,
                # NOT an absolute path. The server validates against its known
                # LoRA list (from /sdcpp/v1/capabilities).
                "path": lora_path_str,
                "multiplier": 1.0,
            },
        ],
        "hires": _build_hires_config(config, width, height),
        "vae_tiling_params": {
            "enabled": True,
        },
        "output_format": "png",
    }

    return payload


def run_inference(
    config: dict,
    lora_path: Path,
    lora_name: str,
    caption: str,
    width: int,
    height: int,
    output_dir: Path,
) -> bool:
    """Run a single inference via sd-server sdcpp API.

    Submits an async job, polls for completion, and saves the image.

    Args:
        config: Eval config dict (must include server_url).
        lora_path: Full path to the LoRA .safetensors file.
        lora_name: Display name for logging.
        caption: Caption tags string.
        width: Output image width.
        height: Output image height.
        output_dir: Directory to write the output image into.

    Returns:
        True if the inference succeeded and image was saved.
    """
    server_url = config.get("server_url", "http://127.0.0.1:1234")
    output_path = output_dir / f"{lora_name}-{width}x{height}.png"
    poll_interval = config.get("poll_interval", 2)
    poll_timeout = config.get("poll_timeout", 600)

    logger.info(
        "Inference: %s @ %dx%d -> %s",
        lora_name, width, height, output_path.name,
    )

    # Build and submit job
    payload = _build_sdcpp_payload(config, lora_path, caption, width, height, server_url)
    job = _submit_job(server_url, payload)

    if job is None:
        return False

    job_id = job.get("id", "unknown")
    logger.debug("  Submitted job %s (status: %s)", job_id, job.get("status"))

    # Poll for result
    result_job = _poll_job(server_url, job_id, poll_timeout, poll_interval)
    if result_job is None:
        return False

    # Extract and save image
    result = result_job.get("result", {})
    images = result.get("images", [])
    if not images:
        logger.error("  Job completed but returned no images")
        return False

    try:
        image_data = base64.b64decode(images[0]["b64_json"])
        output_path.write_bytes(image_data)
        logger.info("  Generated: %s", output_path)
        return True
    except (KeyError, IndexError, Exception) as e:
        logger.error("  Failed to decode/save image: %s", e)
        return False


# ── Main orchestrator ────────────────────────────────────────────────────


def run_evaluation(
    config: dict,
    dataset_dir: str,
    output_dir: str,
    caption: str | None = None,
) -> int:
    """Run full evaluation: start server, generate images, stop server.

    Starts sd-server with the configured model and LoRA directories,
    waits for it to be ready, runs inference for each LoRA at all
    resolutions, then stops the server.

    For each LoRA file in output_dir, generates images at all configured
    resolutions via sd-server. Images are saved to output_dir/samples/.

    Args:
        config: Eval config dict (from load_eval_config).
        dataset_dir: Path to dataset img/ directory (for caption selection).
        output_dir: Path to directory containing LoRA .safetensors files.
        caption: Pre-selected caption string. If None, picks one randomly.

    Returns:
        Number of successful inferences.
    """
    global _server_process

    server_url = config.get("server_url", "http://127.0.0.1:1234")
    startup_timeout = config.get("server_startup_timeout", 120)
    out_path = Path(output_dir).resolve()
    proc: subprocess.Popen | None = None

    try:
        # Start sd-server
        proc = _start_server(config, out_path)
        _server_process = proc

        # Wait for server to be ready
        if not _wait_for_server_ready(server_url, startup_timeout):
            return 0

        # Pick caption
        if caption is None:
            try:
                caption = pick_caption(dataset_dir)
            except ValueError as e:
                logger.error("Cannot pick caption for evaluation: %s", e)
                return 0

        # Discover LoRA files (use the discover_all_loras from evaluate.py instead)
        # But we keep the simple version here for the library API
        lora_files = discover_lora_files(output_dir)
        if not lora_files:
            logger.warning("No LoRA files found in %s — skipping evaluation", output_dir)
            return 0

        samples_dir = out_path / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

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
                    (samples_dir / f"{lora_name}-{width}x{height}.png").unlink(missing_ok=True)

        logger.info(
            "Evaluation complete: %d/%d images generated in %s",
            success_count, total, samples_dir,
        )
        return success_count

    except RuntimeError as e:
        logger.error("Server setup failed: %s", e)
        return 0

    finally:
        _stop_server(proc)
        _server_process = None
