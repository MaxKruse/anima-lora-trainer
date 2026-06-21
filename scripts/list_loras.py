"""Print top-level .safetensors files as LoRA prompt tags.

Outputs comma-separated and pipe-separated <lora:name:1> tags
for use in generation prompts or web UIs.

Usage:
    python scripts/list_loras.py datasets/character/out
    python scripts/list_loras.py --format pipe datasets/character/out
"""

import argparse
import sys
from pathlib import Path


def list_loras(folder: Path, separator: str = ",") -> str:
    """Return LoRA tags for all .safetensors files in *folder*.

    Args:
        folder: Directory to scan (non-recursive).
        separator: Character joining tags (default: comma).

    Returns:
        Comma-separated string of <lora:name:1> tags.

    Raises:
        FileNotFoundError: If *folder* does not exist.
        NotADirectoryError: If *folder* is not a directory.
        SystemExit: If no .safetensors files are found.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix == ".safetensors"
    )

    if not files:
        print(f"No .safetensors files found in {folder}", file=sys.stderr)
        raise SystemExit(1)

    return separator.join(f"<lora:{f.stem}:1>" for f in files)


def main():
    parser = argparse.ArgumentParser(
        description="Print .safetensors files as <lora:name:1> prompt tags",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "pipe"],
        default="csv",
        help="Output format: csv (comma-separated) or pipe (| separated) [default: csv]",
    )
    args = parser.parse_args()

    folder = Path(args.directory)
    separator = "|" if args.format == "pipe" else ","

    try:
        result = list_loras(folder, separator)
        print(result)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
