"""CLI argument parsing for LoRA training.

Handles argument definition, parsing, and conversion to training params.
"""

import argparse
import sys
from scripts.constants import DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="LoRA Matrix Trainer — single and matrix training modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a dataset
  %(prog)s --validate --dataset datasets/froot/img

  # Single training run with defaults
  %(prog)s --mode single --dataset datasets/froot/img --name Froot-Anima

  # Custom parameters
  %(prog)s --mode single --dataset datasets/froot/img --name Froot --lr 0.0001 --bs 2 --network-dim 32

  # Matrix run (all permutations of given values)
  %(prog)s --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,32 --alpha 1,16 --lr 0.0001,0.0002

  # Matrix with output dir
  %(prog)s --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,20,32 --alpha 1,20 -o datasets/froot/out/matrix-run
""",
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        choices=["single", "matrix"],
        default="single",
        help="Training mode: single run or matrix (permutation) sweep [default: single]",
    )

    # Validate
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset (check image count, captions) and exit",
    )

    # Dataset (not required so --help works without it)
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Path to directory containing training images and .txt captions",
    )

    # Name
    parser.add_argument(
        "--name", "-n",
        help="LoRA output name (default: dataset folder name)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output directory base (default: datasets/<name>/out/<job-id>)",
    )

    # ── Training parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--network-dim",
        type=str,
        default=str(DEFAULTS["network_dim"]),
        help=f"LoRA dimension(s) [default: {DEFAULTS['network_dim']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=str,
        default=str(DEFAULTS["network_alpha"]),
        help=f"LoRA alpha(s) [default: {DEFAULTS['network_alpha']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=str,
        default=str(DEFAULTS["learning_rate"]),
        help=f"Learning rate(s) [default: {DEFAULTS['learning_rate']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--batch-size", "--bs",
        type=str,
        default=str(DEFAULTS["batch_size"]),
        help=f"Batch size(s) [default: {DEFAULTS['batch_size']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--max-steps", "--ss",
        type=str,
        default=str(DEFAULTS["max_steps"]),
        help=f"Max training step(s) [default: auto from batch_size, manual override] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULTS["optimizer"],
        help=f"Optimizer(s) [default: {DEFAULTS['optimizer']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--scheduler", "-s",
        type=str,
        default=DEFAULTS["scheduler"],
        help=f"LR scheduler(s) [default: {DEFAULTS['scheduler']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=str(DEFAULTS["resolution"]),
        help=f"Resolution(s) [default: {DEFAULTS['resolution']}] (comma-sep for matrix)",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=None,
        help="Override num_repeats (auto-calculated from image count if omitted)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--no-cache-latents",
        action="store_true",
        help="Disable latent caching",
    )
    parser.add_argument(
        "--cache-text-encoder",
        action="store_true",
        help="Enable text encoder output caching",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=DEFAULTS["mixed_precision"],
        help=f"Mixed precision [default: {DEFAULTS['mixed_precision']}]",
    )
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default=DEFAULTS["timestep_sampling"],
        help=f"Timestep sampling [default: {DEFAULTS['timestep_sampling']}]",
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=DEFAULTS["caption_tag_dropout_rate"],
        help=f"Caption tag dropout rate [default: {DEFAULTS['caption_tag_dropout_rate']}]",
    )
    parser.add_argument(
        "--keep-tokens",
        type=int,
        default=DEFAULTS["keep_tokens"],
        help=f"Keep first N tokens from shuffle [default: {DEFAULTS['keep_tokens']}]",
    )
    parser.add_argument(
        "--rebalance-buckets",
        action="store_true",
        help="Detect dominant bucket skew (>20%% default) and redistribute by cropping excess images to adjacent buckets",
    )
    parser.add_argument(
        "--bucket-dominance-threshold",
        type=float,
        default=0.20,
        help="Dominant bucket share threshold that triggers rebalancing [default: 0.20]",
    )
    parser.add_argument(
        "--bucket-rebalance-max-aug",
        type=int,
        default=64,
        help="Maximum random-crop augmented samples for bucket rebalance [default: 64]",
    )
    parser.add_argument(
        "--bucket-rebalance-seed",
        type=int,
        default=42,
        help="Random seed for bucket rebalance crop selection [default: 42]",
    )

    # Matrix resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="(Matrix mode) Resume from existing manifest",
    )

    return parser


def parse_list_value(value: str) -> list:
    """Parse a comma-separated value string into a list of mixed types."""
    parts = [p.strip() for p in value.split(",")]
    result = []
    for part in parts:
        if not part:
            continue
        try:
            int_val = int(part)
            if str(int_val) == part:
                result.append(int_val)
                continue
        except ValueError:
            pass
        try:
            result.append(float(part))
            continue
        except ValueError:
            pass
        result.append(part)
    return result


def parse_params(args) -> dict:
    """Parse CLI args into a training params dict (single value each)."""
    return {
        "network_dim": int(args.network_dim),
        "network_alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "max_steps": int(args.max_steps),
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "resolution": int(args.resolution),
        "mixed_precision": args.mixed_precision,
        "timestep_sampling": args.timestep_sampling,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "cache_latents": not args.no_cache_latents,
        "cache_text_encoder": args.cache_text_encoder,
        "caption_tag_dropout_rate": args.caption_dropout,
        "keep_tokens": args.keep_tokens,
        "repeats": args.repeats,
        "rebalance_buckets": args.rebalance_buckets,
        "bucket_dominance_threshold": args.bucket_dominance_threshold,
        "bucket_rebalance_max_aug": args.bucket_rebalance_max_aug,
        "bucket_rebalance_seed": args.bucket_rebalance_seed,
    }


def parse_param_ranges(args, include_single: bool = False) -> dict:
    """Parse CLI args into param ranges dict.

    If include_single is False (default), only parameters with multiple values
    are returned (used for permutation generation).
    If include_single is True, all range-capable parameters are returned.
    """
    range_keys = [
        ("network_dim", args.network_dim),
        ("network_alpha", args.alpha),
        ("learning_rate", args.learning_rate),
        ("batch_size", args.batch_size),
        ("max_steps", args.max_steps),
        ("optimizer", args.optimizer),
        ("scheduler", args.scheduler),
        ("resolution", args.resolution),
    ]
    ranges = {}
    for key, value in range_keys:
        parsed = parse_list_value(value)
        if include_single or len(parsed) > 1:
            ranges[key] = parsed
    return ranges


def ensure_dataset(args):
    """Validate that --dataset was provided for non-help operations.

    Returns the dataset path string.
    """
    if args.dataset is None:
        print("ERROR: --dataset is required.", file=sys.stderr)
        raise SystemExit(1)
    return args.dataset
