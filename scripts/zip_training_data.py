"""Zip training images and captions into a single archive.

Preserves folder structure (subdirectories for outfits, variations, etc.)
so the archive mirrors the original dataset layout.

Usage:
    python zip_training_data.py <source_dir> <output_dir>

Creates <output_dir>/training-data.zip containing all image and
caption files from <source_dir>, preserving subdirectory structure.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

from scripts.constants import IMAGE_EXTENSIONS


def zip_training_data(source_dir: str, output_dir: str) -> str | None:
    """Zip all images and their caption files from source_dir.

    Preserves subdirectory structure. For example:
        source_dir/
          base/
            img1.jpg, img1.txt
          outfit1/
            img2.jpg, img2.txt
    becomes:
        training-data.zip/
          base/
            img1.jpg, img1.txt
          outfit1/
            img2.jpg, img2.txt

    Returns the path to the created zip file, or None if no files found.
    """
    src = Path(source_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Collect all image and caption files, preserving relative paths
    files_to_add: list[tuple[Path, str]] = []  # (absolute_path, archive_arcname)

    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        for filename in sorted(files):
            filepath = root_path / filename
            suffix = filepath.suffix.lower()

            # Include images and .txt caption files
            if suffix in IMAGE_EXTENSIONS or suffix == '.txt':
                # Relative path from source_dir becomes the archive path
                rel_path = filepath.relative_to(src)
                # Use forward slashes for cross-platform zip compatibility
                arcname = str(rel_path).replace(os.sep, '/')
                files_to_add.append((filepath, arcname))

    if not files_to_add:
        print("No image or caption files found in source directory", file=sys.stderr)
        return None

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'training-data.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filepath, arcname in files_to_add:
            zf.write(filepath, arcname)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Zip training data (preserves folder structure)')
    parser.add_argument('source_dir', help='Directory containing training images')
    parser.add_argument('output_dir', help='Directory for the output zip file')
    args = parser.parse_args()

    try:
        result = zip_training_data(args.source_dir, args.output_dir)
        if result is None:
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
