"""Single training script — generates dataset TOML, builds command, launches training."""

import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import scripts.X` works
# when run as `python scripts/train_single.py` (not `python -m scripts.train_single`)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.command_builder import build_training_command
from scripts.dataset_toml import generate_dataset_toml


def run_training(params: dict) -> dict:
    """Run a single training job.

    Args:
        params: Training parameters dict.

    Returns:
        dict with status and output_dir.
    """
    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "job_manifest.json"

    # Write initial manifest
    manifest = {
        "status": "running",
        "params": {k: v for k, v in params.items() if k != "output_dir"},
        "output_dir": str(output_dir),
    }
    _write_manifest(manifest_path, manifest)

    try:
        # Step 1: Generate dataset TOML
        dataset_toml_path = output_dir / "dataset.toml"
        num_images = _count_images(params["training_images"])

        # Determine repeats: use explicit value, or derive from steps_per_epoch
        user_repeats = params.get("repeats")
        if user_repeats is not None:
            num_repeats = user_repeats
        else:
            steps_per_epoch = max(100, num_images)  # Default: at least 100 steps
            num_repeats = max(1, -(-steps_per_epoch // num_images))  # ceil division

        generate_dataset_toml(
            image_dir=params["training_images"],
            batch_size=params["batch_size"],
            num_images=num_images,
            epochs=params["epochs"],
            num_repeats=num_repeats,
            output_path=str(dataset_toml_path),
            resolution=params.get("resolution", 1024),
        )

        # Step 2: Build training command
        params["dataset_config"] = str(dataset_toml_path)
        cmd = build_training_command(params)

        # Step 3: Launch training via uv run
        full_cmd = ["uv", "run"] + cmd
        result = subprocess.run(full_cmd, cwd=str(output_dir.parent), shell=False)

        # Step 4: Update manifest
        status = "completed" if result.returncode == 0 else "failed"
        manifest["status"] = status
        manifest["exit_code"] = result.returncode
        _write_manifest(manifest_path, manifest)

        return {
            "status": status,
            "output_dir": str(output_dir),
            "exit_code": result.returncode,
        }

    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        _write_manifest(manifest_path, manifest)

        return {
            "status": "failed",
            "error": str(e),
        }


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
    return max(count, 1)  # At least 1 to avoid division by zero


def _write_manifest(path: Path, data: dict) -> None:
    """Write job manifest as JSON."""
    path.write_text(json.dumps(data, indent=2))


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
    result = run_training(params)

    if result["status"] == "completed":
        print(f"Training completed: {result['output_dir']}")
    else:
        print(f"Training failed: {result.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
