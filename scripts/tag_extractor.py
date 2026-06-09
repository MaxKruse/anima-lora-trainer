"""Extract unique tags from .txt caption files in a directory."""

import os
from pathlib import Path


def extract_tags(directory: str) -> list[str]:
    """Read all .txt files in *directory* and return a sorted list of unique tags.

    Tags are split on commas or whitespace.  Empty captions are ignored.
    """
    tags: set[str] = set()

    for file_path in sorted(Path(directory).glob("*.txt")):
        content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue

        # Replace commas with spaces, then split on whitespace
        normalized = content.replace(",", " ")
        for tag in normalized.split():
            tag = tag.strip().lower()
            if tag:
                tags.add(tag)

    return sorted(tags)
