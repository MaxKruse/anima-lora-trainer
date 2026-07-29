"""Debug script to probe sd-server capabilities and test payload formats.

Run this AFTER starting sd-server manually (or let this script start it).
It queries /sdcpp/v1/capabilities to see available samplers, LoRAs, and limits,
then tries a minimal job submission to pinpoint the 400 error cause.

Usage:
    uv run python scripts/debug_sd_server.py
    uv run python scripts/debug_sd_server.py --no-start   # server already running
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.constants import MODEL_PATHS
from scripts.evaluation import _EVAL_DEFAULTS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

SERVER_URL = "http://127.0.0.1:1234"


def http_get(url: str) -> bytes:
    req = urllib_request.Request(url)
    with urllib_request.urlopen(req, timeout=30) as resp:
        return resp.read()


def http_post(url: str, body: dict) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib_error.HTTPError as e:
        return e.code, e.read()


def wait_for_server(timeout: float = 120) -> bool:
    url = f"{SERVER_URL}/sdcpp/v1/capabilities"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_get(url)
            return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(output_dir: Path) -> subprocess.Popen | None:
    """Start sd-server with the eval config model paths."""
    config_path = _project_root / "eval.config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = dict(_EVAL_DEFAULTS)

    # Fill model paths
    config.setdefault("diffusion_model", MODEL_PATHS["diffusion_model"])
    config.setdefault("vae", MODEL_PATHS["vae"])
    config.setdefault("encoder", MODEL_PATHS["text_encoder"])

    # Find LoRA dir
    work_dir = output_dir / ".work"
    lora_dir = str(work_dir.resolve()) if work_dir.is_dir() else None

    cmd = [
        config["sd_server_path"],
        "--diffusion-model", config["diffusion_model"],
        "--vae", config["vae"],
        "--llm", config["encoder"],
        "--listen-port", str(config.get("listen_port", 1234)),
    ]
    if lora_dir:
        cmd.extend(["--lora-model-dir", lora_dir])

    logger.info("Starting sd-server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_project_root),
    )
    logger.info("sd-server started (pid %d)", proc.pid)
    return proc


def main():
    parser = argparse.ArgumentParser(description="Debug sd-server API")
    parser.add_argument("--no-start", action="store_true", help="Assume server is already running")
    parser.add_argument("--output", "-o", type=str, default=str(_project_root / "datasets" / "calico_trevobutevil" / "out"), help="Output dir for LoRA discovery")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()

    # Start server if needed
    proc = None
    if not args.no_start:
        proc = start_server(output_dir)
        if not wait_for_server():
            logger.error("Server did not start in time")
            sys.exit(1)
    else:
        if not wait_for_server(10):
            logger.error("Server not reachable at %s", SERVER_URL)
            sys.exit(1)

    try:
        # 1. Query capabilities
        logger.info("=" * 60)
        logger.info("1. Querying /sdcpp/v1/capabilities")
        logger.info("=" * 60)
        raw = http_get(f"{SERVER_URL}/sdcpp/v1/capabilities")
        caps = json.loads(raw)

        print("\n--- Model ---")
        print(json.dumps(caps.get("model", {}), indent=2))

        print("\n--- Current Mode ---")
        print(caps.get("current_mode"))

        print("\n--- Supported Modes ---")
        print(json.dumps(caps.get("supported_modes", []), indent=2))

        print("\n--- Defaults (img_gen) ---")
        defaults_by_mode = caps.get("defaults_by_mode", {})
        img_gen_defaults = defaults_by_mode.get("img_gen", {})
        print(json.dumps(img_gen_defaults, indent=2))

        print("\n--- Samplers ---")
        print(json.dumps(caps.get("samplers", []), indent=2))

        print("\n--- Schedulers ---")
        print(json.dumps(caps.get("schedulers", []), indent=2))

        print("\n--- LoRAs ---")
        print(json.dumps(caps.get("loras", []), indent=2))

        print("\n--- Limits ---")
        print(json.dumps(caps.get("limits", {}), indent=2))

        print("\n--- Features (img_gen) ---")
        features_by_mode = caps.get("features_by_mode", {})
        print(json.dumps(features_by_mode.get("img_gen", {}), indent=2))

        # 2. Test minimal payload (no LoRA)
        logger.info("=" * 60)
        logger.info("2. Testing minimal payload (no LoRA)")
        logger.info("=" * 60)

        # Use defaults from capabilities (fallback to Anima preset: er_sde + simple)
        sample_method = img_gen_defaults.get("sample_params", {}).get("sample_method", "er_sde")
        scheduler = img_gen_defaults.get("sample_params", {}).get("scheduler", "simple")
        steps = img_gen_defaults.get("sample_params", {}).get("sample_steps", 32)
        cfg = img_gen_defaults.get("sample_params", {}).get("guidance", {}).get("txt_cfg", 4.0)
        default_width = img_gen_defaults.get("width", 1024)
        default_height = img_gen_defaults.get("height", 1024)

        minimal_payload = {
            "prompt": "test",
            "negative_prompt": "",
            "width": default_width,
            "height": default_height,
            "seed": 42,
            "batch_count": 1,
            "sample_params": {
                "sample_method": sample_method,
                "scheduler": scheduler,
                "sample_steps": steps,
                "guidance": {
                    "txt_cfg": cfg,
                },
            },
            "output_format": "png",
        }

        status, raw = http_post(f"{SERVER_URL}/sdcpp/v1/img_gen", minimal_payload)
        print(f"\nStatus: {status}")
        print(f"Response: {raw.decode('utf-8', errors='replace')[:500]}")

        if status == 202:
            logger.info("Minimal payload accepted (202) - no LoRA works!")

        # 3. Test with LoRA using relative path (file stem)
        logger.info("=" * 60)
        logger.info("3. Testing with LoRA (relative path = file stem)")
        logger.info("=" * 60)

        available_loras = caps.get("loras", [])
        if available_loras:
            lora_entry = available_loras[0]
            lora_path = lora_entry.get("path", "")
            lora_name = lora_entry.get("name", "")

            print(f"\nTrying LoRA: {lora_name}")
            print(f"Path from capabilities: {lora_path}")

            payload_with_lora_relative = dict(minimal_payload)
            payload_with_lora_relative["lora"] = [
                {
                    "path": lora_path,
                    "multiplier": 1.0,
                }
            ]

            status, raw = http_post(f"{SERVER_URL}/sdcpp/v1/img_gen", payload_with_lora_relative)
            print(f"\nStatus: {status}")
            print(f"Response: {raw.decode('utf-8', errors='replace')[:500]}")

            if status == 202:
                logger.info("LoRA with relative path accepted (202)!")
            else:
                logger.error("LoRA with relative path FAILED (status %d)", status)

        # 4. Test with LoRA using absolute path (current broken behavior)
        logger.info("=" * 60)
        logger.info("4. Testing with LoRA (absolute path - current broken behavior)")
        logger.info("=" * 60)

        if available_loras:
            # Find the actual file
            work_dir = output_dir / ".work"
            lora_files = list(work_dir.glob("*.safetensors")) if work_dir.is_dir() else []
            if lora_files:
                abs_path = str(lora_files[0].resolve())
                print(f"\nTrying absolute path: {abs_path}")

                payload_with_lora_absolute = dict(minimal_payload)
                payload_with_lora_absolute["lora"] = [
                    {
                        "path": abs_path,
                        "multiplier": 1.0,
                    }
                ]

                status, raw = http_post(f"{SERVER_URL}/sdcpp/v1/img_gen", payload_with_lora_absolute)
                print(f"\nStatus: {status}")
                print(f"Response: {raw.decode('utf-8', errors='replace')[:500]}")

                if status == 202:
                    logger.info("LoRA with absolute path accepted (202)!")
                else:
                    logger.error("LoRA with absolute path FAILED (status %d) - confirms the bug", status)

        # 5. Test with LoRA using just the filename (stem + .safetensors)
        logger.info("=" * 60)
        logger.info("5. Testing with LoRA (filename only)")
        logger.info("=" * 60)

        if lora_files:
            filename = lora_files[0].name
            print(f"\nTrying filename: {filename}")

            payload_with_lora_filename = dict(minimal_payload)
            payload_with_lora_filename["lora"] = [
                {
                    "path": filename,
                    "multiplier": 1.0,
                }
            ]

            status, raw = http_post(f"{SERVER_URL}/sdcpp/v1/img_gen", payload_with_lora_filename)
            print(f"\nStatus: {status}")
            print(f"Response: {raw.decode('utf-8', errors='replace')[:500]}")

            if status == 202:
                logger.info("LoRA with filename accepted (202)!")

    finally:
        if proc is not None:
            logger.info("Stopping sd-server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
