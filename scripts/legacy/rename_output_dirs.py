"""Batch-rename existing output directories to use compact names.

Parses old-style folder names like:
  anima_batch-size-4_epochs-10_learning-rate-1e-3_mixed-precision-bf16_...
and renames them to:
  anima_bs-4_ep-10_lr-1e-3_mp-bf16_...

Usage:
  uv run python scripts/rename_output_dirs.py output/job-123abc
  uv run python scripts/rename_output_dirs.py output/ --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


# Mapping from old-style param slugs to short codes
_OLD_PARAM_MAP = {
    "batch-size": "bs",
    "epochs": "ep",
    "learning-rate": "lr",
    "mixed-precision": "mp",
    "network-alpha": "na",
    "network-dim": "nd",
    "optimizer": "opt",
    "resolution": "res",
    "scheduler": "sch",
    "timestep-sampling": "ts",
    "noise-offset": "no",
    "gradient-accumulation-steps": "gas",
    "max-train-steps": "mts",
    "text-encoder-lr": "tlr",
    "unet-lr": "ulr",
    "lr-scheduler": "lrs",
    "lr-warmup-steps": "lrs_w",
    "rank-dropout": "rd",
}

# Mapping from old-style value slugs to short codes
_OLD_VALUE_MAP = {
    "mixed-precision-bf16": "bf16",
    "mixed-precision-fp16": "fp16",
    "mixed-precision-none": "np",
    "adamw8bit": "a8b",
    "adamw8bit-torch": "a8bt",
    "adamw": "aw",
    "prodigy": "pr",
    "lion": "lion",
    "constant": "c",
    "cosine": "cos",
    "cosine-with-restarts": "cosr",
    "linear": "l",
    "polynomial": "poly",
    "constant-with-warmup": "cw",
    "sigmoid": "sig",
    "karras": "kar",
    "exponential": "exp",
    "uniform": "u",
}


def parse_old_folder_name(name: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse an old-style folder name into (prefix, [(param, value), ...]).

    e.g. "anima_batch-size-4_epochs-10" -> ("anima", [("batch-size", "4"), ("epochs", "10")])
    """
    parts = name.split("_")
    if not parts:
        return name, []

    prefix = parts[0]
    params = []

    i = 1
    while i < len(parts):
        part = parts[i]
        # Match pattern: param-name-value or param-name-value-with-dashes
        # The param is everything up to the second hyphen-separated segment that looks like a value
        # Strategy: try to match known params first
        matched = False
        for old_param, short in _OLD_PARAM_MAP.items():
            pattern = f"{old_param}-"
            if part.startswith(pattern):
                value = part[len(pattern):]
                params.append((old_param, value))
                matched = True
                break
        if matched:
            i += 1
            continue

        # Try compound params like "timestep-sampling-sigmoid"
        # These span multiple underscore-separated parts
        remaining = "_".join(parts[i:])
        for old_param, short in _OLD_PARAM_MAP.items():
            pattern = f"{old_param}-"
            if remaining.startswith(pattern):
                # Value is everything after the param prefix
                value = remaining[len(pattern):]
                params.append((old_param, value))
                i = len(parts)  # consume all remaining
                break
        else:
            # Unknown param — try to split on first hyphen
            if "-" in part:
                dash_idx = part.index("-")
                param = part[:dash_idx]
                value = part[dash_idx + 1:]
                params.append((param, value))
            else:
                # No hyphen — might be a leftover or unknown format
                params.append((part, ""))
            i += 1

    return prefix, params


def shorten_value(value: str) -> str:
    """Apply value abbreviation."""
    if value in _OLD_VALUE_MAP:
        return _OLD_VALUE_MAP[value]
    # Check if it's a boolean
    if value == "true":
        return "t"
    if value == "false":
        return "f"
    return value


def shorten_param_name(param: str) -> str:
    """Apply param name abbreviation."""
    if param in _OLD_PARAM_MAP:
        return _OLD_PARAM_MAP[param]
    # Fallback: first 2 chars of each word
    words = param.replace("-", " ").split()
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[:2] for w in words)


def build_new_name(prefix: str, params: list[tuple[str, str]]) -> str:
    """Build a compact folder name from parsed components."""
    parts = [prefix]
    for param, value in params:
        short_param = shorten_param_name(param)
        short_value = shorten_value(value)
        parts.append(f"{short_param}-{short_value}")
    return "_".join(parts)


def rename_directory(old_path: Path, new_name: str, dry_run: bool = False) -> bool:
    """Rename a directory, printing what would happen."""
    parent = old_path.parent
    new_path = parent / new_name

    if old_path == new_path:
        print(f"  SKIP (same name): {old_path.name}")
        return True

    if new_path.exists():
        print(f"  SKIP (exists):    {new_name}")
        return False

    if dry_run:
        print(f"  WOULD RENAME: {old_path.name} ({len(old_path.name)} chars)")
        print(f"    -> {new_name} ({len(new_name)} chars)")
        return True

    try:
        shutil.move(str(old_path), str(new_path))
        print(f"  RENAMED: {old_path.name} ({len(old_path.name)} chars)")
        print(f"    -> {new_name} ({len(new_name)} chars)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch-rename output directories to compact names")
    parser.add_argument("directory", help="Job output directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    args = parser.parse_args()

    base_dir = Path(args.directory)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)

    # Find all subdirectories (the training run folders)
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        sys.exit(0)

    renamed = 0
    skipped = 0
    errors = 0

    for d in subdirs:
        name = d.name
        # Only process folders that look like training run names (start with model prefix)
        if not any(name.startswith(p) for p in ["anima", "flux", "sd3", "sdxl", "lumina", "hunyuan"]):
            continue

        prefix, params = parse_old_folder_name(name)
        if not params:
            print(f"  SKIP (no params): {name}")
            skipped += 1
            continue

        new_name = build_new_name(prefix, params)

        if rename_directory(d, new_name, args.dry_run):
            if name != new_name and not (args.dry_run):
                renamed += 1
            elif name == new_name:
                skipped += 1
            else:
                renamed += 1
        else:
            errors += 1

    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Done: {renamed} renamed, {skipped} skipped, {errors} errors out of {len(subdirs)}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
