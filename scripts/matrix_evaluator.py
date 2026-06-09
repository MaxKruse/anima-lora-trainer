"""Matrix evaluator — scan results, find LoRAs, run sd-cli, write evaluation.json."""

import json
import subprocess
import time
from pathlib import Path

from scripts.lora_finder import find_latest_lora, NoLoraFoundError
from scripts.sdcli_builder import build_sdcli_command


def run_evaluation(
    run_dir: str,
    diffusion_model: str,
    vae_model: str,
    llm_model: str,
    prompt: str,
    seed: int = 42,
) -> dict:
    """Evaluate all LoRA checkpoints in a matrix training run.

    Scans *run_dir* for permutation subdirectories, finds the latest LoRA
    checkpoint in each, runs ``sd-cli`` inference, and writes
    ``evaluation.json``.

    Args:
        run_dir: Path to the matrix training output directory.
        diffusion_model: Path to the base diffusion model.
        vae_model: Path to the VAE model.
        llm_model: Path to the text encoder / LLM model.
        prompt: Evaluation prompt text.
        seed: Random seed for deterministic generation.

    Returns:
        dict with summary of the evaluation run.
    """
    run_path = Path(run_dir)
    evaluation_path = run_path / "evaluation.json"

    # Discover permutation subdirectories
    perm_dirs = sorted(
        [d for d in run_path.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    results: list[dict] = []

    for perm_dir in perm_dirs:
        perm_name = perm_dir.name
        print(f"\nEvaluating {perm_name}...")

        try:
            lora_path = find_latest_lora(str(perm_dir))
        except NoLoraFoundError as e:
            print(f"  ✗ No LoRA found for {perm_name}: {e}")
            results.append(
                {
                    "perm_name": perm_name,
                    "lora_file": None,
                    "image_file": None,
                    "status": "failed",
                    "error": str(e),
                    "inference_time_ms": 0,
                }
            )
            continue

        # Build output image path
        image_filename = f"eval_{perm_name}.png"
        image_path = run_path / image_filename

        # Build and run sd-cli command
        cmd_str = build_sdcli_command(
            diffusion_model=diffusion_model,
            vae_model=vae_model,
            llm_model=llm_model,
            lora_file=str(lora_path),
            prompt=prompt,
            seed=seed,
            output_path=str(image_path),
        )

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            if result.returncode == 0:
                print(f"  ✓ Completed: {perm_name} ({elapsed_ms}ms)")
                results.append(
                    {
                        "perm_name": perm_name,
                        "lora_file": lora_path.name,
                        "image_file": image_filename,
                        "status": "completed",
                        "error": None,
                        "inference_time_ms": elapsed_ms,
                    }
                )
            else:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                print(f"  ✗ Failed: {perm_name} — {error_msg}")
                results.append(
                    {
                        "perm_name": perm_name,
                        "lora_file": lora_path.name,
                        "image_file": None,
                        "status": "failed",
                        "error": error_msg,
                        "inference_time_ms": elapsed_ms,
                    }
                )

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            print(f"  ✗ Exception: {perm_name} — {e}")
            results.append(
                {
                    "perm_name": perm_name,
                    "lora_file": lora_path.name,
                    "image_file": None,
                    "status": "failed",
                    "error": str(e),
                    "inference_time_ms": elapsed_ms,
                }
            )

    # Write evaluation.json
    evaluation_data = {
        "prompt": prompt,
        "seed": seed,
        "total": len(results),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }

    evaluation_path.write_text(json.dumps(evaluation_data, indent=2))
    print(f"\nWrote {evaluation_path}")

    return evaluation_data


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate matrix training LoRAs")
    parser.add_argument("--run-dir", required=True, help="Matrix training output directory")
    parser.add_argument("--diffusion-model", required=True, help="Diffusion model path")
    parser.add_argument("--vae-model", required=True, help="VAE model path")
    parser.add_argument("--llm-model", required=True, help="Text encoder / LLM path")
    parser.add_argument("--prompt", required=True, help="Evaluation prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = run_evaluation(
        run_dir=args.run_dir,
        diffusion_model=args.diffusion_model,
        vae_model=args.vae_model,
        llm_model=args.llm_model,
        prompt=args.prompt,
        seed=args.seed,
    )

    print(f"\nDone: {result['completed']} completed, {result['failed']} failed out of {result['total']}")


if __name__ == "__main__":
    main()
