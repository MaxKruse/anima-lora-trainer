"""Matrix trainer script — parse args, generate permutations, iterate and train each."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.command_builder import build_training_command
from scripts.dataset_toml import generate_dataset_toml
from scripts.manifest_writer import ManifestWriter
from scripts.permutation_generator import generate_permutations
from scripts.permutation_namer import generate_folder_name


def run_matrix_training(
    permutations: list[dict],
    output_dir: str,
    resume: bool = False,
    base_params: dict | None = None,
) -> dict:
    """Run matrix training across all permutations.

    Args:
        permutations: List of permutation parameter dicts.
        output_dir: Base output directory.
        resume: If True, skip already-completed permutations.
        base_params: Base training parameters (merged with each permutation).

    Returns:
        dict with overall status and summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "manifest.json"
    writer = ManifestWriter(manifest_path, permutations)

    cancel_file = output_path / "cancel"

    # Determine which permutations to process
    if resume:
        pending_indices = writer.get_pending_indices()
    else:
        pending_indices = list(range(len(permutations)))

    print(f"Starting matrix training: {len(pending_indices)} permutations to process")

    for idx in pending_indices:
        # Check for cancel signal
        if cancel_file.exists():
            print("Cancel signal detected. Stopping training.")
            break

        perm = permutations[idx]
        perm_name = generate_folder_name(perm)
        perm_dir = output_path / perm_name

        print(f"\n[{idx + 1}/{len(permutations)}] Training {perm_name}...")
        writer.update_status(idx, "running")

        try:
            result = _train_single(perm, perm_dir, output_dir, base_params)
            status = result["status"]

            if status == "completed":
                output_files = result.get("output_files", [])
                writer.update_status(idx, "completed", output_files=output_files)
                print(f"  ✓ Completed: {perm_name}")
            else:
                error = result.get("error", "Unknown error")
                writer.update_status(idx, "failed", error=error)
                print(f"  ✗ Failed: {perm_name} — {error}")

        except Exception as e:
            writer.update_status(idx, "failed", error=str(e))
            print(f"  ✗ Exception: {perm_name} — {e}")

    # Final summary
    data = writer.read()
    completed = data["completed"]
    failed = data["failed"]
    total = data["total"]
    skipped = total - completed - failed

    print(f"\nMatrix training finished: {completed} completed, {failed} failed, {skipped} skipped")

    return {
        "status": "completed" if failed == 0 and skipped == 0 else "partial",
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "output_dir": str(output_path),
    }


def _train_single(
    perm_params: dict,
    perm_dir: Path,
    output_dir: str,
    base_params: dict | None = None,
) -> dict:
    """Train a single permutation.

    Args:
        perm_params: Permutation-specific parameters.
        perm_dir: Output directory for this permutation.
        output_dir: Base output directory.
        base_params: Base training parameters to merge with perm_params.

    Returns:
        dict with status and optional output_files/error.
    """
    perm_dir.mkdir(parents=True, exist_ok=True)

    # Merge base params with permutation params
    params = {**(base_params or {}), **perm_params}

    # Generate dataset TOML
    dataset_toml_path = perm_dir / "dataset.toml"
    if "training_images" in params:
        num_images = _count_images(params["training_images"])
        steps_per_epoch = max(100, num_images)

        generate_dataset_toml(
            image_dir=params["training_images"],
            batch_size=params.get("batch_size", 1),
            num_images=num_images,
            epochs=params.get("epochs", 10),
            steps_per_epoch=steps_per_epoch,
            output_path=str(dataset_toml_path),
        )
        params["dataset_config"] = str(dataset_toml_path)

    # Build and launch training command
    params["output_dir"] = str(perm_dir)
    params["output_name"] = perm_dir.name

    try:
        cmd = build_training_command(params)
        full_cmd = ["uv", "run"] + cmd
        result = subprocess.run(full_cmd, cwd=str(perm_dir.parent), shell=False)

        if result.returncode == 0:
            # Find output files
            output_files = [
                f.name for f in perm_dir.glob("*.safetensors") if f.is_file()
            ]
            return {"status": "completed", "output_files": output_files}
        else:
            return {"status": "failed", "error": f"Exit code {result.returncode}"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _count_images(image_dir: str) -> int:
    """Count image files in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    count = 0
    try:
        for entry in Path(image_dir).iterdir():
            if entry.suffix.lower() in image_extensions:
                count += 1
    except FileNotFoundError:
        pass
    return max(count, 1)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run matrix LoRA training")
    parser.add_argument("--param-ranges", required=True, help="JSON object of param ranges")
    parser.add_argument("--base-params", default="{}", help="JSON object of base params")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from manifest")
    args = parser.parse_args()

    param_ranges = json.loads(args.param_ranges)
    base_params = json.loads(args.base_params)

    # Generate permutations
    permutations = generate_permutations(param_ranges)
    print(f"Generated {len(permutations)} permutations")

    # Run matrix training
    result = run_matrix_training(
        permutations,
        args.output_dir,
        resume=args.resume,
        base_params=base_params,
    )

    if result["status"] != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
