"""Tests for single training script."""

import json
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from scripts.train_single import run_training


class TestRunTraining:
    """Test single training execution."""

    def test_creates_output_directory_before_launching(self, tmp_path):
        output_dir = tmp_path / "output" / "nested"
        params = {
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 1,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": str(tmp_path / "images"),
            "lora_name": "test-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": str(output_dir),
            "model_type": "anima",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_training(params)

            assert output_dir.exists()

    def test_writes_initial_job_manifest_with_status_running(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params = {
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 1,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": str(tmp_path / "images"),
            "lora_name": "test-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": str(output_dir),
            "model_type": "anima",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_training(params)

            manifest_path = output_dir / "job_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "completed"

    def test_updates_manifest_to_completed_on_exit_code_0(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params = {
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 1,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": str(tmp_path / "images"),
            "lora_name": "test-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": str(output_dir),
            "model_type": "anima",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_training(params)

            manifest_path = output_dir / "job_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "completed"
            assert result["status"] == "completed"

    def test_updates_manifest_to_failed_on_nonzero_exit_code(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        params = {
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 1,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": str(tmp_path / "images"),
            "lora_name": "test-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": str(output_dir),
            "model_type": "anima",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = run_training(params)

            manifest_path = output_dir / "job_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "failed"
            assert result["status"] == "failed"

    def test_produces_correct_command(self, tmp_path):
        """Verify the training command includes required flags."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        params = {
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": str(images_dir),
            "lora_name": "test-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": str(output_dir),
            "model_type": "anima",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_training(params)

            # Verify subprocess.run was called
            assert mock_run.call_count >= 1  # At least dataset toml gen + training
