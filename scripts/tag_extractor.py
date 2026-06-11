"""Extract unique tags from .txt caption files in a directory."""

from pathlib import Path


def extract_tags(directory: str) -> list[str]:
    """Read all .txt files in *directory* and return a sorted list of unique tags.

    Tags are comma-separated (e.g. "1girl, red hair, large breasts").
    Empty captions are ignored.
    """
    tags: set[str] = set()

    for file_path in sorted(Path(directory).glob("*.txt")):
        content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue

        # Tags are comma-separated (e.g., "1girl, red hair, large breasts")
        for tag in content.split(","):
            tag = tag.strip().lower()
            if tag:
                tags.add(tag)

    return sorted(tags)
