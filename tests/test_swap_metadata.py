"""Tests for swap_metadata - clean .tags metadata swap."""

import json
import struct
import tempfile
from pathlib import Path

from scripts.swap_metadata import (
    _build_tag_frequency,
    _parse_tags_file,
    check_tags_coverage,
    swap_metadata_on_all,
)


def _create_tags_file(dir_path: Path, name: str, content: str) -> None:
    """Helper: write a .tags file."""
    (dir_path / f"{name}.tags").write_text(content, encoding="utf-8")


def _create_image_file(dir_path: Path, name: str) -> None:
    """Helper: write a dummy .jpg file."""
    (dir_path / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xe0")


def _create_minimal_safetensors(
    path: Path,
    tag_frequency: dict,
    extra_meta: dict | None = None,
) -> None:
    """Create a minimal valid safetensors file with metadata for testing."""
    meta = {
        "__metadata__": {
            "ss_tag_frequency": json.dumps(tag_frequency),
            "ss_datasets": json.dumps([{
                "num_train_images": 5,
                "tag_frequency": tag_frequency,
            }]),
            "modelspec.title": "test-lora",
        }
    }
    if extra_meta:
        meta["__metadata__"].update(extra_meta)

    header_json = json.dumps(meta)
    header_bytes = header_json.encode("utf-8")

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # No weight data - minimal file


def _read_metadata(path: Path) -> dict:
    """Helper: read __metadata__ from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    header = json.loads(header_json)
    return header.get("__metadata__", {})


# ── _parse_tags_file ──────────────────────────────────────────────────────


class TestParseTagsFile:
    def test_basic_comma_separated(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tags"
        f.write_text("1girl, long hair, blue eyes", encoding="utf-8")
        assert _parse_tags_file(f) == ["1girl", "long hair", "blue eyes"]

    def test_whitespace_trimming(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tags"
        f.write_text("  1girl  ,  long hair ,blue eyes  ", encoding="utf-8")
        assert _parse_tags_file(f) == ["1girl", "long hair", "blue eyes"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tags"
        f.write_text("", encoding="utf-8")
        assert _parse_tags_file(f) == []

    def test_single_tag(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tags"
        f.write_text("solo", encoding="utf-8")
        assert _parse_tags_file(f) == ["solo"]

    def test_trailing_comma(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tags"
        f.write_text("1girl, solo,", encoding="utf-8")
        assert _parse_tags_file(f) == ["1girl", "solo"]


# ── _build_tag_frequency ──────────────────────────────────────────────────


class TestBuildTagFrequency:
    def test_single_dir(self) -> None:
        tags = {"": ["1girl", "solo", "1girl", "long hair"]}
        freq = _build_tag_frequency(tags)
        assert freq == {"": {"1girl": 2, "solo": 1, "long hair": 1}}

    def test_multiple_dirs(self) -> None:
        tags = {
            "": ["1girl", "solo"],
            "outfit1": ["1girl", "dress"],
        }
        freq = _build_tag_frequency(tags)
        assert freq[""]["1girl"] == 1
        assert freq[""]["solo"] == 1
        assert freq["outfit1"]["1girl"] == 1
        assert freq["outfit1"]["dress"] == 1

    def test_empty_input(self) -> None:
        freq = _build_tag_frequency({})
        assert freq == {}


# ── check_tags_coverage ───────────────────────────────────────────────────


class TestCheckTagsCoverage:
    def test_full_coverage(self, tmp_path: Path) -> None:
        for i in range(5):
            _create_image_file(tmp_path, f"img{i}")
            _create_tags_file(tmp_path, f"img{i}", "1girl")
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is True
        assert tags == 5
        assert imgs == 5

    def test_no_tags(self, tmp_path: Path) -> None:
        for i in range(5):
            _create_image_file(tmp_path, f"img{i}")
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is False
        assert tags == 0
        assert imgs == 5

    def test_below_threshold(self, tmp_path: Path) -> None:
        # 2 tags for 5 images = 40% < 80% threshold
        for i in range(5):
            _create_image_file(tmp_path, f"img{i}")
        _create_tags_file(tmp_path, "img0", "1girl")
        _create_tags_file(tmp_path, "img1", "solo")
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is False
        assert tags == 2
        assert imgs == 5

    def test_at_threshold(self, tmp_path: Path) -> None:
        # 4 tags for 5 images = 80% == threshold
        for i in range(5):
            _create_image_file(tmp_path, f"img{i}")
        for i in range(4):
            _create_tags_file(tmp_path, f"img{i}", "1girl")
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is True
        assert tags == 4
        assert imgs == 5

    def test_with_subdirs(self, tmp_path: Path) -> None:
        # 3 images + 3 tags in root, 2 images + 2 tags in subdir = 5/5
        for i in range(3):
            _create_image_file(tmp_path, f"img{i}")
            _create_tags_file(tmp_path, f"img{i}", "1girl")
        subdir = tmp_path / "outfit1"
        subdir.mkdir()
        for i in range(2):
            _create_image_file(subdir, f"img{i}")
            _create_tags_file(subdir, f"img{i}", "solo")
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is True
        assert tags == 5
        assert imgs == 5

    def test_empty_dir(self, tmp_path: Path) -> None:
        covered, tags, imgs = check_tags_coverage(str(tmp_path))
        assert covered is False
        assert tags == 0
        assert imgs == 0


# ── swap_metadata_on_all (integration) ────────────────────────────────────


class TestSwapMetadataOnAll:
    def test_basic_swap(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        # Create images + .tags files (full coverage)
        for i in range(3):
            _create_image_file(img_dir, f"img{i}")
            _create_tags_file(img_dir, f"img{i}", "1girl, solo, clean tag")

        # Create a safetensors file with dirty metadata
        dirty_freq = {
            "non_dominant": {
                "1girl": 3,
                "solo": 3,
                "a girl standing in a sunlit forest clearing": 1,
                "dappled light filtering through the canopy above her": 1,
            }
        }
        model = work_dir / "test-lora.safetensors"
        _create_minimal_safetensors(model, dirty_freq)

        # Run swap
        result = swap_metadata_on_all(str(work_dir), str(img_dir))
        assert result is True

        # Verify metadata was swapped
        meta = _read_metadata(model)
        new_freq = json.loads(meta["ss_tag_frequency"])
        assert "clean tag" in str(new_freq)
        assert "sunlit forest" not in str(new_freq)
        assert "dappled light" not in str(new_freq)

    def test_skips_when_no_tags(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        # Images but no .tags files
        for i in range(3):
            _create_image_file(img_dir, f"img{i}")

        dirty_freq = {"non_dominant": {"1girl": 3}}
        model = work_dir / "test-lora.safetensors"
        _create_minimal_safetensors(model, dirty_freq)

        result = swap_metadata_on_all(str(work_dir), str(img_dir))
        assert result is False

        # Metadata unchanged
        meta = _read_metadata(model)
        freq = json.loads(meta["ss_tag_frequency"])
        assert freq == dirty_freq

    def test_swaps_all_safetensors_files(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        # Full coverage
        for i in range(3):
            _create_image_file(img_dir, f"img{i}")
            _create_tags_file(img_dir, f"img{i}", "1girl, solo")

        # Create multiple safetensors files (checkpoint + final)
        dirty_freq = {"non_dominant": {"1girl": 3, "dirty nl sentence here": 1}}
        for name in ["model-step00000600.safetensors", "model.safetensors"]:
            _create_minimal_safetensors(work_dir / name, dirty_freq)

        result = swap_metadata_on_all(str(work_dir), str(img_dir))
        assert result is True

        # Both files should be cleaned
        for name in ["model-step00000600.safetensors", "model.safetensors"]:
            meta = _read_metadata(work_dir / name)
            freq = json.loads(meta["ss_tag_frequency"])
            assert "dirty nl sentence here" not in str(freq)

    def test_preserves_other_metadata(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        for i in range(3):
            _create_image_file(img_dir, f"img{i}")
            _create_tags_file(img_dir, f"img{i}", "1girl")

        dirty_freq = {"non_dominant": {"1girl": 3}}
        model = work_dir / "test.safetensors"
        _create_minimal_safetensors(model, dirty_freq, extra_meta={
            "ss_network_dim": "8",
            "ss_learning_rate": "0.0002",
            "ss_steps": "600",
            "modelspec.title": "Emiru-Anima",
        })

        swap_metadata_on_all(str(work_dir), str(img_dir))

        meta = _read_metadata(model)
        assert meta["ss_network_dim"] == "8"
        assert meta["ss_learning_rate"] == "0.0002"
        assert meta["ss_steps"] == "600"
        assert meta["modelspec.title"] == "Emiru-Anima"

    def test_also_swaps_ss_datasets_tag_frequency(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        for i in range(3):
            _create_image_file(img_dir, f"img{i}")
            _create_tags_file(img_dir, f"img{i}", "1girl, solo")

        dirty_freq = {"non_dominant": {"1girl": 3, "nl leakage sentence": 1}}
        model = work_dir / "test.safetensors"
        _create_minimal_safetensors(model, dirty_freq)

        swap_metadata_on_all(str(work_dir), str(img_dir))

        meta = _read_metadata(model)
        datasets = json.loads(meta["ss_datasets"])
        ds_freq = datasets[0]["tag_frequency"]
        assert "nl leakage sentence" not in str(ds_freq)

    def test_file_round_trip_integrity(self, tmp_path: Path) -> None:
        """Verify the safetensors file is still valid after swap."""
        img_dir = tmp_path / "img"
        work_dir = tmp_path / "work"
        img_dir.mkdir()
        work_dir.mkdir()

        for i in range(3):
            _create_image_file(img_dir, f"img{i}")
            _create_tags_file(img_dir, f"img{i}", "1girl")

        dirty_freq = {"non_dominant": {"1girl": 3}}
        model = work_dir / "test.safetensors"
        _create_minimal_safetensors(model, dirty_freq)

        # Read original file size (header + no weights)
        original_size = model.stat().st_size

        swap_metadata_on_all(str(work_dir), str(img_dir))

        # File should still be readable
        meta = _read_metadata(model)
        assert "ss_tag_frequency" in meta
        assert "ss_datasets" in meta
        assert "modelspec.title" in meta
