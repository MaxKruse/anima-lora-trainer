"""Matrix trainer script — parse args, generate permutations, iterate and train each.

Delegates each permutation to train_single.py for the actual training.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import scripts.X` works
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set UTF-8 encoding for stdout so Unicode chars (✓, ✗) work on Windows cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# CamelCase -> snake_case converter (TS sends camelCase, Python expects snake_case)
def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _normalize_params(params: dict) -> dict:
    return {_camel_to_snake(k): v for k, v in params.items()}


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
    """Train a single permutation by delegating to train_single.py.

    Args:
        perm_params: Permutation-specific parameters.
        perm_dir: Output directory for this permutation.
        output_dir: Base output directory (unused, kept for compat).
        base_params: Base training parameters to merge with perm_params.

    Returns:
        dict with status and optional output_files/error.
    """
    perm_dir.mkdir(parents=True, exist_ok=True)

    # Merge base params with permutation params
    params = {**(base_params or {}), **perm_params}
    params["output_dir"] = str(perm_dir)

    # Write params to a temp file to avoid shell escaping issues
    params_file = perm_dir / "params.json"
    params_file.write_text(json.dumps(params))

    # Delegate to train_single.py which handles dataset TOML, command building,
    # progress tracking, and the actual training launch
    train_script = _project_root / "scripts" / "train_single.py"
    full_cmd = [
        "uv", "run", "python",
        str(train_script),
        "--params-json-file", str(params_file),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            full_cmd,
            cwd=_project_root,
            shell=False,
            env=env,
        )

        if result.returncode == 0:
            output_files = [
                f.name for f in perm_dir.glob("*.safetensors") if f.is_file()
            ]
            return {"status": "completed", "output_files": output_files}
        else:
            return {"status": "failed", "error": f"Exit code {result.returncode}"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run matrix LoRA training")
    parser.add_argument("--param-ranges", default=None, help="JSON string of param ranges (legacy)")
    parser.add_argument("--param-ranges-file", default=None, help="Path to JSON file of param ranges")
    parser.add_argument("--base-params", default="{}", help="JSON string of base params (legacy)")
    parser.add_argument("--base-params-file", default=None, help="Path to JSON file of base params")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from manifest")
    args = parser.parse_args()

    # Support both file paths and inline JSON strings
    if args.param_ranges_file:
        param_ranges = json.loads(Path(args.param_ranges_file).read_text())
    elif args.param_ranges:
        param_ranges = json.loads(args.param_ranges)
    else:
        parser.error("Either --param-ranges or --param-ranges-file is required")

    if args.base_params_file:
        base_params = json.loads(Path(args.base_params_file).read_text())
    else:
        base_params = json.loads(args.base_params)

    # Normalize camelCase (from TS) -> snake_case (expected by Python)
    param_ranges = _normalize_params(param_ranges)
    base_params = _normalize_params(base_params)

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
