"""CLI argument parsing for LoRA training.

Handles argument definition, parsing, and conversion to training params.
"""

import argparse
import sys
from typing import Any

from scripts.constants import (
    CHARACTER_DEFAULTS,
    SHARED_DEFAULTS,
    STYLE_DEFAULTS,
)


def _type_defaults(type_: str) -> dict[str, Any]:
    """Get type-specific defaults."""
    return STYLE_DEFAULTS if type_ == "style" else CHARACTER_DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="LoRA Trainer — character and style LoRA training on Anima",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a dataset
  %(prog)s --type character --dataset datasets/emiru/ --validate

  # Train a character LoRA with defaults
  %(prog)s --type character --dataset datasets/emiru/ --name Emiru-Anima

  # Train a style LoRA (lower LR, more steps, higher dim)
  %(prog)s --type style --dataset datasets/blobcg/ --name BlobCG-Style

  # Custom parameters
  %(prog)s --type character --dataset datasets/emiru/ --name Emiru --lr 0.0001 --bs 2 --network-dim 32
"""
    )

    # Type (required)
    parser.add_argument(
        "--type", "-t",
        choices=["character", "style"],
        default="character",
        help="Training type: character (dim=8, lr=0.0002, steps=600) or style (dim=16, lr=0.0001, steps=1200) [default: character]",
    )

    # Validate
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset (check folder structure, image count, captions) and exit",
    )

    # Dataset (not required so --help works without it)
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Path to dataset directory (must contain img/ and out/ subdirectories)",
    )

    # Name
    parser.add_argument(
        "--name", "-n",
        help="LoRA output name (default: dataset folder name)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: <dataset>/out/)",
    )

    # ── Training parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--network-dim",
        type=str,
        default=None,
        help=f"LoRA dimension [default: {CHARACTER_DEFAULTS['network_dim']} (char) / {STYLE_DEFAULTS['network_dim']} (style)]",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=str,
        default=str(CHARACTER_DEFAULTS["network_alpha"]),
        help=f"LoRA alpha [default: {CHARACTER_DEFAULTS['network_alpha']}]",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=str,
        default=None,
        help=f"Learning rate [default: {CHARACTER_DEFAULTS['learning_rate']} (char) / {STYLE_DEFAULTS['learning_rate']} (style)]",
    )
    parser.add_argument(
        "--batch-size", "--bs",
        type=str,
        default=str(CHARACTER_DEFAULTS["batch_size"]),
        help=f"Batch size [default: {CHARACTER_DEFAULTS['batch_size']}]",
    )
    parser.add_argument(
        "--max-steps", "--ss",
        type=str,
        default=None,
        help=f"Max training steps [default: auto from batch_size and type (char bs4=600, style bs4=1200), manual override]",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=SHARED_DEFAULTS["optimizer"],
        help=f"Optimizer [default: {SHARED_DEFAULTS['optimizer']}]",
    )
    parser.add_argument(
        "--scheduler", "-s",
        type=str,
        default=SHARED_DEFAULTS["scheduler"],
        help=f"LR scheduler [default: {SHARED_DEFAULTS['scheduler']}]",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=str(SHARED_DEFAULTS["resolution"]),
        help=f"Resolution [default: {SHARED_DEFAULTS['resolution']}]",
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
        default=SHARED_DEFAULTS["mixed_precision"],
        help=f"Mixed precision [default: {SHARED_DEFAULTS['mixed_precision']}]",
    )
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default=SHARED_DEFAULTS["timestep_sampling"],
        help=f"Timestep sampling [default: {SHARED_DEFAULTS['timestep_sampling']}]",
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=SHARED_DEFAULTS["caption_tag_dropout_rate"],
        help=f"Caption tag dropout rate [default: {SHARED_DEFAULTS['caption_tag_dropout_rate']}]",
    )
    parser.add_argument(
        "--keep-tokens",
        type=int,
        default=SHARED_DEFAULTS["keep_tokens"],
        help=f"Keep first N tokens from shuffle [default: {SHARED_DEFAULTS['keep_tokens']}]",
    )
    parser.add_argument(
        "--rebalance-buckets",
        action="store_true",
        default=SHARED_DEFAULTS["rebalance_buckets"],
        help="Detect dominant bucket skew (>20%% default) and redistribute by cropping excess images to adjacent buckets [default: on]",
    )
    parser.add_argument(
        "--no-rebalance-buckets",
        action="store_false",
        dest="rebalance_buckets",
        help="Disable bucket rebalancing",
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

    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run sd.cpp inference on the trained LoRA after training completes",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Path to eval config JSON",
    )

    return parser


def parse_params(
    args: argparse.Namespace,
    training_type: str = "character",
) -> dict[str, Any]:
    """Parse CLI args into a training params dict.

    Uses type-specific defaults for network_dim, learning_rate, and max_steps.
    When a param is not explicitly set (None), the type-specific default is used.
    max_steps is always None so that run_single_training can auto-calculate from batch_size.
    """
    td = _type_defaults(training_type)

    return {
        "network_dim": int(args.network_dim) if args.network_dim is not None else td["network_dim"],
        "network_alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate) if args.learning_rate is not None else td["learning_rate"],
        "batch_size": int(args.batch_size),
        "max_steps": int(args.max_steps) if args.max_steps is not None else None,
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


def ensure_dataset(args: argparse.Namespace) -> str:
    """Validate that --dataset was provided for non-help operations.

    Returns the dataset path string.

    Raises:
        SystemExit: If --dataset is not provided.
    """
    if args.dataset is None:
        print("ERROR: --dataset is required.", file=sys.stderr)
        raise SystemExit(1)
    return args.dataset
