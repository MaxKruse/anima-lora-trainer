"""Tests for matrix trainer script."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.matrix_trainer import run_matrix_training


class TestMatrixTrainer:
    """Test run_matrix_training function."""

    def test_creates_output_dir_and_manifest(self, tmp_path: Path):
        """Creates output directory and manifest before training loop"""
        output_dir = tmp_path / "output"
        permutations = [
            {"network_dim": 1, "network_alpha": 1},
            {"network_dim": 2, "network_alpha": 4},
        ]

        def mock_train(perm, perm_dir, output_dir, base_params=None):
            return {"status": "completed"}

        with patch("scripts.matrix_trainer._train_single", side_effect=mock_train):
            run_matrix_training(permutations, str(output_dir))

        assert output_dir.exists()
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data["total"] == 2

    def test_processes_permutations_sequentially(self, tmp_path: Path):
        """Processes permutations sequentially (one at a time)"""
        output_dir = tmp_path / "output"
        permutations = [
            {"network_dim": 1},
            {"network_dim": 2},
            {"network_dim": 3},
        ]
        call_order = []

        def mock_train_sequential(perm, perm_dir, output_dir, base_params=None):
            call_order.append(perm["network_dim"])
            return {"status": "completed"}

        with patch("scripts.matrix_trainer._train_single", side_effect=mock_train_sequential):
            run_matrix_training(permutations, str(output_dir))

        assert call_order == [1, 2, 3]

    def test_updates_manifest_for_each_permutation(self, tmp_path: Path):
        """Updates manifest status for each permutation as it completes"""
        output_dir = tmp_path / "output"
        permutations = [
            {"network_dim": 1},
            {"network_dim": 2},
        ]

        def mock_train_mixed(perm, perm_dir, output_dir, base_params=None):
            if perm["network_dim"] == 1:
                return {"status": "completed"}
            return {"status": "failed", "error": "OOM"}

        with patch("scripts.matrix_trainer._train_single", side_effect=mock_train_mixed):
            run_matrix_training(permutations, str(output_dir))

        manifest_path = output_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())

        assert data["permutations"][0]["status"] == "completed"
        assert data["permutations"][1]["status"] == "failed"
        assert data["permutations"][1]["error"] == "OOM"

    def test_stops_on_cancel_signal(self, tmp_path: Path):
        """Stops on cancel signal file presence"""
        output_dir = tmp_path / "output"
        cancel_file = output_dir / "cancel"
        permutations = [
            {"network_dim": 1},
            {"network_dim": 2},
            {"network_dim": 3},
        ]
        call_order = []

        def mock_train_cancel(perm, perm_dir, output_dir, base_params=None):
            call_order.append(perm["network_dim"])
            # Create cancel file after first permutation
            if perm["network_dim"] == 1:
                cancel_file.touch()
            return {"status": "completed"}

        with patch("scripts.matrix_trainer._train_single", side_effect=mock_train_cancel):
            run_matrix_training(permutations, str(output_dir))

        # Should have stopped after first permutation
        assert call_order == [1]

    def test_resume_skips_completed(self, tmp_path: Path):
        """Supports --resume to skip already-completed permutations"""
        output_dir = tmp_path / "output"
        # Pre-create manifest with first permutation completed
        manifest_path = output_dir / "manifest.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "permutations": [
                {
                    "index": 0,
                    "params": {"network_dim": 1},
                    "status": "completed",
                    "output_files": [],
                },
                {
                    "index": 1,
                    "params": {"network_dim": 2},
                    "status": "pending",
                    "output_files": [],
                },
            ],
            "total": 2,
            "completed": 1,
            "failed": 0,
        }
        manifest_path.write_text(json.dumps(manifest_data))

        permutations = [
            {"network_dim": 1},
            {"network_dim": 2},
        ]
        call_order = []

        def mock_train_resume(perm, perm_dir, output_dir, base_params=None):
            call_order.append(perm["network_dim"])
            return {"status": "completed"}

        with patch("scripts.matrix_trainer._train_single", side_effect=mock_train_resume):
            run_matrix_training(permutations, str(output_dir), resume=True)

        # Should only train the pending permutation
        assert call_order == [2]
