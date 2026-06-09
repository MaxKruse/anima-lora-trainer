"""Tests for tag extraction from caption files."""

import pytest
from scripts.tag_extractor import extract_tags


class TestExtractTags:
    """Test tag extraction from .txt caption files."""

    def test_returns_all_unique_tags_from_txt_files(self, tmp_path):
        """Given directory with .txt files, returns list of all unique tags."""
        (tmp_path / "img1.txt").write_text("cat, dog")
        (tmp_path / "img2.txt").write_text("dog, bird")
        (tmp_path / "img3.txt").write_text("cat")

        tags = extract_tags(str(tmp_path))

        assert sorted(tags) == ["bird", "cat", "dog"]

    def test_splits_comma_separated_tags(self, tmp_path):
        """Multi-tag captions separated by commas are split into individual tags."""
        (tmp_path / "img1.txt").write_text("red, blue, green")

        tags = extract_tags(str(tmp_path))

        assert sorted(tags) == ["blue", "green", "red"]

    def test_splits_space_separated_tags(self, tmp_path):
        """Multi-tag captions separated by spaces are split into individual tags."""
        (tmp_path / "img1.txt").write_text("sunset mountain lake")

        tags = extract_tags(str(tmp_path))

        assert sorted(tags) == ["lake", "mountain", "sunset"]

    def test_ignores_empty_caption_files(self, tmp_path):
        """Empty caption files should not contribute any tags."""
        (tmp_path / "img1.txt").write_text("")
        (tmp_path / "img2.txt").write_text("  \n  ")
        (tmp_path / "img3.txt").write_text("valid_tag")

        tags = extract_tags(str(tmp_path))

        assert tags == ["valid_tag"]

    def test_handles_missing_captions_gracefully(self, tmp_path):
        """Directory with no .txt files should return empty list, not crash."""
        (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "img2.png").write_bytes(b"\x89PNG")

        tags = extract_tags(str(tmp_path))

        assert tags == []
