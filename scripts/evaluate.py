"""Standalone LoRA evaluation script.

Discovers all .safetensors files in a training output directory and uses
sd-server to generate comparison images at multiple resolutions.

Handles both single-run output (``.work/`` subdirectory) and matrix-run
output (permutation subdirectories) automatically.

Requires sd-server to be running. Start it with:
    sd-server.exe --diffusion-model <model> --vae <vae> --llm <encoder> --lora-model-dir <dir>

Usage:
    # Evaluate all LoRAs in an output folder
    uv run python scripts/evaluate.py --dataset datasets/froot/img --output datasets/froot/out

    # Custom caption instead of random pick
    uv run python scripts/evaluate.py --dataset datasets/froot/img --output datasets/froot/out --caption "1girl, red hair"

    # Dry run (list LoRAs without generating images)
    uv run python scripts/evaluate.py --dataset datasets/froot/img --output datasets/froot/out --dry-run

    # Custom config, seed, and server URL
    uv run python scripts/evaluate.py --dataset datasets/froot/img --output datasets/froot/out --eval-config eval.config.json --seed 1234 --server-url http://127.0.0.1:1234
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Project setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation import (
    EVAL_RESOLUTIONS,
    discover_lora_files,
    load_eval_config,
    pick_caption,
    run_inference,
    _start_server,
    _stop_server,
    _wait_for_server_ready,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def discover_all_loras(output_dir: Path) -> list[tuple[Path, str]]:
    """Discover all .safetensors LoRA files in *output_dir*.

    Handles three layouts:
      1. Single run:  output_dir/.work/<name>.safetensors
      2. Matrix run:  output_dir/<perm-name>/<name>.safetensors
      3. Flat:        output_dir/<name>.safetensors (top-level files)

    Returns list of (file_path, display_label) tuples.
    The display_label includes the parent folder name for matrix runs.
    """
    results: list[tuple[Path, str]] = []

    if not output_dir.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return results

    # Check for single-run .work/ subdirectory
    work_dir = output_dir / ".work"
    if work_dir.is_dir():
        for lora_path, lora_name in discover_lora_files(str(work_dir)):
            results.append((lora_path, lora_name))

    # Scan immediate subdirectories (matrix permutation folders)
    perm_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name != ".work" and d.name != "samples"
    )
    for perm_dir in perm_dirs:
        for lora_path, lora_name in discover_lora_files(str(perm_dir)):
            # Prefix with permutation folder name for clarity
            label = f"{perm_dir.name}/{lora_name}"
            results.append((lora_path, label))

    # Also check top-level .safetensors (copied final models).
    # Skip files that are already found inside .work/ or perm dirs
    # (the top-level copy is redundant for evaluation).
    existing_stems = {r[0].stem for r in results}
    for entry in output_dir.iterdir():
        if (entry.is_file()
                and entry.suffix.lower() == ".safetensors"
                and "-state" not in str(entry.parent)):
            if entry.stem not in existing_stems:
                results.append((entry, entry.stem))

    results.sort(key=lambda x: x[1])
    return results


def run_evaluate(
    dataset_dir: Path,
    output_dir: Path,
    eval_config_path: str | None,
    caption: str | None,
    seed: int | None,
    server_url: str | None,
    sd_server_path: str | None,
    dry_run: bool,
) -> int:
    """Run the full evaluation pipeline.

    Returns the number of successful inferences (0 on failure).
    """
    # Load config
    config = load_eval_config(eval_config_path)
    if seed is not None:
        config["seed"] = seed
    if server_url is not None:
        config["server_url"] = server_url
    if sd_server_path is not None:
        config["sd_server_path"] = sd_server_path

    # Resolve caption
    if caption is None:
        try:
            caption = pick_caption(str(dataset_dir))
        except ValueError as e:
            logger.error("Cannot pick caption: %s", e)
            return 0
        logger.info("Auto-selected caption: %s", caption[:100])
    else:
        logger.info("Using provided caption: %s", caption[:100])

    # Discover LoRAs
    loras = discover_all_loras(output_dir)
    if not loras:
        logger.error("No LoRA (.safetensors) files found in %s", output_dir)
        return 0

    logger.info("Discovered %d LoRA(s):", len(loras))
    for _, label in loras:
        logger.info("  - %s", label)

    if dry_run:
        logger.info(
            "Dry run — would generate %d images (%d LoRAs x %d resolutions)",
            len(loras) * len(EVAL_RESOLUTIONS),
            len(loras),
            len(EVAL_RESOLUTIONS),
        )
        return 0

    # Start sd-server
    startup_timeout = config.get("server_startup_timeout", 120)
    server_url = config.get("server_url", "http://127.0.0.1:1234")
    proc = _start_server(config, output_dir)

    try:
        if not _wait_for_server_ready(server_url, startup_timeout):
            return 0

        # Create samples directory at output_dir root
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        total = len(loras) * len(EVAL_RESOLUTIONS)
        success = 0
        failed = 0

        for lora_idx, (lora_path, label) in enumerate(loras, 1):
            logger.info(
                "[%d/%d] Evaluating: %s",
                lora_idx, len(loras), label,
            )
            for res_idx, (width, height) in enumerate(EVAL_RESOLUTIONS, 1):
                logger.info(
                    "  [%d/%d] %dx%d",
                    res_idx, len(EVAL_RESOLUTIONS), width, height,
                )
                ok = run_inference(
                    config=config,
                    lora_path=lora_path,
                    lora_name=lora_path.stem,
                    caption=caption,
                    width=width,
                    height=height,
                    output_dir=samples_dir,
                )
                if ok:
                    success += 1
                else:
                    failed += 1
                    # Clean up partial output
                    (samples_dir / f"{lora_path.stem}-{width}x{height}.png").unlink(
                        missing_ok=True,
                    )

        logger.info(
            "Evaluation complete: %d/%d images generated in %s",
            success, total, samples_dir,
        )
        return success

    finally:
        _stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained LoRAs by generating comparison images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all LoRAs in an output folder
  %(prog)s --dataset datasets/froot/img --output datasets/froot/out

  # Custom caption
  %(prog)s --dataset datasets/froot/img --output datasets/froot/out --caption "1girl, red hair, standing"

  # Dry run (list LoRAs without generating images)
  %(prog)s --dataset datasets/froot/img --output datasets/froot/out --dry-run

  # Custom seed and config
  %(prog)s --dataset datasets/froot/img --output datasets/froot/out --seed 1234 --eval-config eval.config.json
""",
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to the dataset img/ directory (for caption selection)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the training output directory (contains .safetensors files)",
    )
    parser.add_argument(
        "--caption", "-c",
        default=None,
        help="Caption to use for all images (default: pick random from dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for inference (default: from config or 42)",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Path to eval config JSON (default: eval.config.json or auto-generated)",
    )
    parser.add_argument(
        "--sd-server-path",
        type=str,
        default=None,
        help="Path to sd-server.exe binary (default: from eval config)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="sd-server URL (default: from eval config or http://127.0.0.1:1234)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered LoRAs without running inference",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    if not dataset_path.is_dir():
        logger.error("Dataset directory does not exist: %s", dataset_path)
        sys.exit(1)

    if not output_path.is_dir():
        logger.error("Output directory does not exist: %s", output_path)
        sys.exit(1)

    success = run_evaluate(
        dataset_dir=dataset_path,
        output_dir=output_path,
        eval_config_path=args.eval_config,
        caption=args.caption,
        seed=args.seed,
        server_url=args.server_url,
        sd_server_path=args.sd_server_path,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        sys.exit(0)

    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
