"""Tests for manifest writer."""

import json
import tempfile
from pathlib import Path

import pytest
from scripts.manifest_writer import ManifestWriter


class TestManifestWriter:
    """Test ManifestWriter class."""

    def test_initial_manifest_has_all_pending(self, tmp_path: Path):
        """Initial manifest has all permutations with status pending"""
        permutations = [
            {"network_dim": 1, "network_alpha": 1},
            {"network_dim": 1, "network_alpha": 4},
            {"network_dim": 2, "network_alpha": 1},
            {"network_dim": 2, "network_alpha": 4},
        ]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        data = writer.read()
        assert len(data["permutations"]) == 4
        for perm in data["permutations"]:
            assert perm["status"] == "pending"

    def test_update_to_running_persists(self, tmp_path: Path):
        """Updating one permutation to running persists correctly"""
        permutations = [
            {"network_dim": 1},
            {"network_dim": 2},
        ]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        writer.update_status(0, "running")

        data = writer.read()
        assert data["permutations"][0]["status"] == "running"
        assert data["permutations"][1]["status"] == "pending"

    def test_update_to_completed_stores_output_paths(self, tmp_path: Path):
        """Updating to completed stores output file paths"""
        permutations = [{"network_dim": 1}]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        writer.update_status(
            0,
            "completed",
            output_files=["lora-000001.safetensors", "lora-000002.safetensors"],
        )

        data = writer.read()
        assert data["permutations"][0]["status"] == "completed"
        assert len(data["permutations"][0]["output_files"]) == 2

    def test_update_to_failed_stores_error(self, tmp_path: Path):
        """Updating to failed stores error message"""
        permutations = [{"network_dim": 1}]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        writer.update_status(0, "failed", error="Out of memory")

        data = writer.read()
        assert data["permutations"][0]["status"] == "failed"
        assert data["permutations"][0]["error"] == "Out of memory"

    def test_manifest_survives_re_read(self, tmp_path: Path):
        """Manifest survives re-read (JSON round-trip)"""
        permutations = [
            {"network_dim": 1, "network_alpha": 1},
            {"network_dim": 2, "network_alpha": 4},
        ]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        # Update some statuses
        writer.update_status(0, "completed", output_files=["file1.safetensors"])
        writer.update_status(1, "failed", error="OOM")

        # Re-read by creating a new writer
        writer2 = ManifestWriter(manifest_path, permutations)
        data = writer2.read()

        assert data["permutations"][0]["status"] == "completed"
        assert data["permutations"][1]["status"] == "failed"
        assert data["permutations"][1]["error"] == "OOM"

    def test_manifest_written_to_disk(self, tmp_path: Path):
        """Manifest is actually written to disk as valid JSON"""
        permutations = [{"network_dim": 32}]

        manifest_path = tmp_path / "manifest.json"
        writer = ManifestWriter(manifest_path, permutations)

        assert manifest_path.exists()
        content = manifest_path.read_text()
        parsed = json.loads(content)
        assert "permutations" in parsed
