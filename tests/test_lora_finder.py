"""Tests for LoRA checkpoint file discovery."""

import pytest
from scripts.lora_finder import find_latest_lora, NoLoraFoundError


class TestFindLatestLora:
    """Test finding the highest-epoch LoRA checkpoint in a folder."""

    def test_returns_highest_epoch_checkpoint(self, tmp_path):
        """Given numbered checkpoints, return the one with the highest number."""
        for i in range(1, 11):
            (tmp_path / f"my_lora-{i:06d}.safetensors").write_bytes(b"\x00")

        result = find_latest_lora(str(tmp_path))

        assert result.name == "my_lora-000010.safetensors"

    def test_returns_single_checkpoint(self, tmp_path):
        """A folder with only one checkpoint should return it."""
        (tmp_path / "solo-000001.safetensors").write_bytes(b"\x00")

        result = find_latest_lora(str(tmp_path))

        assert result.name == "solo-000001.safetensors"

    def test_empty_folder_raises_error(self, tmp_path):
        """An empty folder should raise NoLoraFoundError."""
        with pytest.raises(NoLoraFoundError):
            find_latest_lora(str(tmp_path))

    def test_ignores_non_safetensors_files(self, tmp_path):
        """Non-.safetensors files should not be considered."""
        (tmp_path / "weights.safetensors").write_bytes(b"\x00")
        (tmp_path / "weights.pt").write_bytes(b"\x00")
        (tmp_path / "readme.txt").write_text("info")
        (tmp_path / "config.json").write_text("{}")

        result = find_latest_lora(str(tmp_path))

        assert result.name == "weights.safetensors"
