"""Print top-level .safetensors files as a comma-separated string.

Usage:
    python scripts/list_loras.py datasets/squchan/out
"""

import sys
from pathlib import Path


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix == ".safetensors"
    )

    if not files:
        print("No .safetensors files found.", file=sys.stderr)
        sys.exit(1)

    print(",".join(f"<lora:{f.stem}:1>" for f in files))

    print("\n\n")

    print("|".join(f"<lora:{f.stem}:1>" for f in files))


if __name__ == "__main__":
    main()
