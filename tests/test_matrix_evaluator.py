"""Tests for the matrix evaluator script."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from scripts.matrix_evaluator import run_evaluation


class TestRunEvaluation:
    """Test matrix evaluation across all permutation folders."""

    def _setup_run_dir(self, tmp_path):
        """Create a run directory with permutation subfolders and LoRA files."""
        run_dir = tmp_path / "run-001"
        run_dir.mkdir()

        # Create 3 permutation folders with LoRA checkpoints
        for name in ["perm-alpha", "perm-beta", "perm-gamma"]:
            perm_dir = run_dir / name
            perm_dir.mkdir()
            (perm_dir / f"{name}-000001.safetensors").write_bytes(b"\x00")
            (perm_dir / f"{name}-000002.safetensors").write_bytes(b"\x00")

        return run_dir

    def test_scans_results_folder_and_finds_all_permutation_subdirectories(self, tmp_path):
        """Scans the results folder and discovers all permutation subdirectories."""
        run_dir = self._setup_run_dir(tmp_path)

        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        assert result["total"] == 3

    def test_picks_highest_epoch_lora_for_each_permutation(self, tmp_path):
        """Selects the highest-epoch checkpoint for each permutation."""
        run_dir = self._setup_run_dir(tmp_path)

        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        # Each call to subprocess.run should use the -000002 (highest) checkpoint
        for call in mock_run.call_args_list:
            cmd = call[1].get("args", call[0][0] if call[0] else [])
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            assert "000002" in cmd_str, f"Expected highest epoch in: {cmd_str}"

    def test_runs_sdcli_for_each_lora_with_same_prompt_and_seed(self, tmp_path):
        """Runs sd-cli for each LoRA using the same prompt and seed."""
        run_dir = self._setup_run_dir(tmp_path)

        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        assert mock_run.call_count == 3
        for call in mock_run.call_args_list:
            cmd = call[1].get("args", call[0][0] if call[0] else [])
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            assert "cat dog bird" in cmd_str
            assert "-s" in cmd_str
            assert "42" in cmd_str

    def test_writes_evaluation_json_with_correct_structure(self, tmp_path):
        """Writes evaluation.json with the expected schema."""
        run_dir = self._setup_run_dir(tmp_path)

        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        eval_path = run_dir / "evaluation.json"
        assert eval_path.exists()

        data = json.loads(eval_path.read_text())
        assert "results" in data
        assert "prompt" in data
        assert "seed" in data
        assert data["prompt"] == "cat dog bird"
        assert data["seed"] == 42
        assert len(data["results"]) == 3

        for entry in data["results"]:
            assert "perm_name" in entry
            assert "lora_file" in entry
            assert "image_file" in entry
            assert "status" in entry
            assert "inference_time_ms" in entry

    def test_records_inference_time_ms_per_result(self, tmp_path):
        """Each result includes an inference_time_ms measurement."""
        run_dir = self._setup_run_dir(tmp_path)

        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        data = json.loads((run_dir / "evaluation.json").read_text())
        for entry in data["results"]:
            assert isinstance(entry["inference_time_ms"], (int, float))
            assert entry["inference_time_ms"] >= 0

    def test_records_failed_status_for_sdcli_errors_without_stopping(self, tmp_path):
        """sd-cli failures are recorded as failed but evaluation continues."""
        run_dir = self._setup_run_dir(tmp_path)

        # First call succeeds, second fails, third succeeds
        with patch("scripts.matrix_evaluator.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stderr=""),
                MagicMock(returncode=1, stderr="sd-cli: model not found"),
                MagicMock(returncode=0, stderr=""),
            ]
            run_evaluation(
                run_dir=str(run_dir),
                diffusion_model="models/anima/diffusion.safetensors",
                vae_model="models/anima/vae.safetensors",
                llm_model="models/anima/text_encoder/model.safetensors",
                prompt="cat dog bird",
                seed=42,
            )

        # All 3 permutations should still be processed
        assert mock_run.call_count == 3

        data = json.loads((run_dir / "evaluation.json").read_text())
        statuses = [r["status"] for r in data["results"]]
        assert statuses.count("completed") == 2
        assert statuses.count("failed") == 1
