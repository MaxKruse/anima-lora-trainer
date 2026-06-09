"""Zip training images and captions into a single archive.

Usage:
    python zip_training_data.py <source_dir> <output_dir>

Creates <output_dir>/training-data.zip containing all image and
caption files from <source_dir>.
"""

import argparse
import os
import sys
import zipfile


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def zip_training_data(source_dir: str, output_dir: str) -> str | None:
    """Zip all images and their caption files from source_dir.

    Returns the path to the created zip file, or None if no files found.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Collect all files
    files = sorted(os.listdir(source_dir))
    image_files = [
        f for f in files
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("No image files found in source directory", file=sys.stderr)
        return None

    # Also include caption files (.txt) that match image names
    image_bases = {os.path.splitext(f)[0] for f in image_files}
    caption_files = [
        f for f in files
        if f.lower().endswith('.txt') and os.path.splitext(f)[0] in image_bases
    ]

    all_files = sorted(set(image_files) | set(caption_files))

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'training-data.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in all_files:
            filepath = os.path.join(source_dir, filename)
            if os.path.isfile(filepath):
                zf.write(filepath, filename)
                print(f"  added: {filename}")

    print(f"Created {zip_path} with {len(all_files)} files")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Zip training data')
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
