"""Swap tag-frequency metadata in safetensors LoRA files with clean .tags data.

After training, kohya-ss embeds ss_tag_frequency metadata derived from .txt
caption files. When captions contain natural language additions (For Anima
mode), the tag frequency includes full sentences as "tags".

This script reads .tags files (booru tags only) and replaces the contaminated
metadata fields with clean counts, leaving all other header keys untouched.

Runs on ALL .safetensors files in the work directory (checkpoints + final).
"""

import json
import logging
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Threshold: require at least this fraction of images to have .tags files
_TAGS_COVERAGE_THRESHOLD = 0.8


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], int]:
    """Read the safetensors header without loading model weights.

    Returns (header_dict, header_size_in_bytes).
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    return json.loads(header_json), header_size


def _write_safetensors_header(path: Path, new_header: dict[str, str]) -> None:
    """Rewrite the safetensors header in-place.

    Reads the original file, replaces only the header JSON prefix,
    and keeps all weight data bytes unchanged.
    """
    new_json = json.dumps(new_header)
    new_bytes = new_json.encode("utf-8")
    new_size = len(new_bytes)

    with open(path, "rb") as f:
        _, old_header_size = _read_safetensors_header(path)
        f.seek(8 + old_header_size)
        weights_data = f.read()

    tmp = path.with_suffix(".swap.tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", new_size))
        f.write(new_bytes)
        f.write(weights_data)
    tmp.replace(path)


def _count_tags_in_dir(dir_path: Path) -> int:
    """Count .tags files in a directory (non-recursive)."""
    return sum(
        1 for entry in dir_path.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".tags"
    )


def _count_images_in_dir(dir_path: Path) -> int:
    """Count image files in a directory (non-recursive)."""
    from scripts.constants import IMAGE_EXTENSIONS
    return sum(
        1 for entry in dir_path.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_tags_file(path: Path) -> list[str]:
    """Parse a .tags file into a list of cleaned tag strings."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def _discover_tag_files(img_dir: str) -> dict[str, list[str]]:
    """Discover all .tags files under img_dir (root + immediate subdirs).

    Returns {relative_subdir_name: [tag_strings]} for each directory
    that has .tags files. Uses "" (empty string) for root-level files.
    """
    root = Path(img_dir).resolve()
    result: dict[str, list[str]] = defaultdict(list)

    # Scan root
    root_tags = _count_tags_in_dir(root)
    if root_tags > 0:
        for tags_file in sorted(root.glob("*.tags")):
            tags = _parse_tags_file(tags_file)
            result[""].extend(tags)

    # Scan immediate subdirs
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            sub_tags = _count_tags_in_dir(entry)
            if sub_tags > 0:
                for tags_file in sorted(entry.glob("*.tags")):
                    tags = _parse_tags_file(tags_file)
                    result[entry.name].extend(tags)

    return dict(result)


def _build_tag_frequency(tags_by_dir: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Build a tag frequency dict grouped by source directory.

    Matches kohya-ss format: {"bucket_name": {"tag": count, ...}, ...}
    """
    freq: dict[str, dict[str, int]] = {}
    for dir_name, all_tags in tags_by_dir.items():
        counts: dict[str, int] = defaultdict(int)
        for tag in all_tags:
            counts[tag] += 1
        freq[dir_name] = dict(counts)
    return freq


def check_tags_coverage(img_dir: str) -> tuple[bool, int, int]:
    """Check if .tags files cover enough images to justify a swap.

    Counts .tags files vs image files across root + subdirs.

    Returns (covered, tags_count, images_count).
    covered is True if tags/images >= _TAGS_COVERAGE_THRESHOLD.
    """
    root = Path(img_dir).resolve()
    total_images = _count_images_in_dir(root)
    total_tags = _count_tags_in_dir(root)

    # Also count subdirs
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            total_images += _count_images_in_dir(entry)
            total_tags += _count_tags_in_dir(entry)

    if total_images == 0:
        return False, 0, 0
    return (total_tags / total_images) >= _TAGS_COVERAGE_THRESHOLD, total_tags, total_images


def _count_unique_tags(freq: dict[str, Any]) -> int:
    """Count total unique tags across all buckets in a frequency dict."""
    all_tags: set[str] = set()
    for bucket_tags in freq.values():
        if isinstance(bucket_tags, dict):
            all_tags.update(bucket_tags.keys())
    return len(all_tags)


def _swap_single_file(model_path: Path, clean_freq: dict[str, dict[str, int]]) -> int:
    """Swap metadata on a single safetensors file.

    Returns the new unique tag count (for reporting), or 0 if skipped.
    """
    header, _ = _read_safetensors_header(model_path)
    meta = header.get("__metadata__", {})
    if not meta:
        logger.warning("swap_metadata: no __metadata__ in %s — skipping", model_path.name)
        return 0

    old_freq_raw = meta.get("ss_tag_frequency", "{}")
    try:
        old_freq = json.loads(old_freq_raw) if old_freq_raw else {}
    except json.JSONDecodeError:
        old_freq = {}
    old_unique = _count_unique_tags(old_freq)

    # --- Swap ss_tag_frequency ---
    meta["ss_tag_frequency"] = json.dumps(clean_freq)

    # --- Swap ss_datasets[0].tag_frequency ---
    datasets_raw = meta.get("ss_datasets", "[]")
    try:
        datasets = json.loads(datasets_raw) if datasets_raw else []
    except json.JSONDecodeError:
        datasets = []

    if datasets and isinstance(datasets, list):
        for ds in datasets:
            if isinstance(ds, dict) and "tag_frequency" in ds:
                ds["tag_frequency"] = clean_freq
                break

    meta["ss_datasets"] = json.dumps(datasets)
    header["__metadata__"] = meta

    _write_safetensors_header(model_path, header)
    return old_unique


def swap_metadata_on_all(work_dir: str, img_dir: str) -> bool:
    """Swap tag-frequency metadata on ALL .safetensors files in work_dir.

    Covers checkpoints (50%, 75%) and the final model.
    Reads .tags files once, builds clean frequency, applies to every file.

    Args:
        work_dir: Path to the .work/ output directory containing .safetensors files.
        img_dir: Path to the dataset img/ directory containing .tags files.

    Returns:
        True if swap was performed on at least one file.
    """
    work = Path(work_dir).resolve()

    # Check coverage
    covered, tags_count, images_count = check_tags_coverage(img_dir)
    if not covered:
        logger.info(
            "swap_metadata: .tags coverage %d/%d (%.0f%%) < %.0f%% threshold — skipping",
            tags_count, images_count,
            (tags_count / images_count * 100) if images_count else 0,
            _TAGS_COVERAGE_THRESHOLD * 100,
        )
        return False

    logger.info(
        "swap_metadata: .tags coverage %d/%d (%.0f%%) — proceeding",
        tags_count, images_count,
        tags_count / images_count * 100,
    )

    # Build clean frequency (read .tags files once)
    tags_by_dir = _discover_tag_files(img_dir)
    if not tags_by_dir:
        logger.info("swap_metadata: no .tags files found — skipping")
        return False

    clean_freq = _build_tag_frequency(tags_by_dir)
    clean_unique = _count_unique_tags(clean_freq)
    logger.info(
        "swap_metadata: clean frequency — %d dirs, %d unique tags",
        len(tags_by_dir), clean_unique,
    )

    # Find all .safetensors files (checkpoints + final)
    safetensors_files = sorted(work.glob("*.safetensors"))
    if not safetensors_files:
        logger.info("swap_metadata: no .safetensors files in %s — skipping", work)
        return False

    # Swap each file
    total_old_unique = 0
    for sf in safetensors_files:
        old_unique = _swap_single_file(sf, clean_freq)
        total_old_unique = max(total_old_unique, old_unique)
        logger.info(
            "swap_metadata: %s — %d -> %d unique tags",
            sf.name, old_unique, clean_unique,
        )

    logger.info(
        "swap_metadata: swapped %d file(s), %d -> %d unique tags total",
        len(safetensors_files), total_old_unique, clean_unique,
    )
    return True
