"""Tests for permutation folder namer."""

import pytest
from scripts.permutation_namer import generate_folder_name


class TestGenerateFolderName:
    """Test generate_folder_name function."""

    def test_basic_params_produce_correct_name(self):
        """Params sorted alphabetically: lr < na < nd"""
        params = {
            "network_dim": 1,
            "network_alpha": 1,
            "learning_rate": 1e-4,
        }
        result = generate_folder_name(params)
        assert result == "anima_lr-1e-4_na-1_nd-1"

    def test_float_values_use_scientific_notation(self):
        """Float values use scientific notation consistently"""
        params = {
            "network_dim": 32,
            "learning_rate": 5e-4,
        }
        result = generate_folder_name(params)
        assert "lr-5e-4" in result

    def test_params_sorted_alphabetically(self):
        """Params sorted alphabetically for deterministic naming"""
        params = {
            "learning_rate": 1e-4,
            "network_dim": 32,
            "network_alpha": 16,
        }
        result = generate_folder_name(params)
        # Alphabetical: learning_rate < network_alpha < network_dim
        lr_pos = result.index("lr-")
        alpha_pos = result.index("na-")
        dim_pos = result.index("nd-")
        assert lr_pos < alpha_pos < dim_pos

    def test_string_params_included(self):
        """String param values like optimizer are abbreviated"""
        params = {
            "network_dim": 32,
            "optimizer": "AdamW8Bit",
        }
        result = generate_folder_name(params)
        assert "opt-a8b" in result

    def test_prefix_included(self):
        """Model prefix (anima) is included in folder name"""
        params = {"network_dim": 32}
        result = generate_folder_name(params)
        assert result.startswith("anima_")

    def test_custom_prefix(self):
        """Custom prefix can be specified"""
        params = {"network_dim": 32}
        result = generate_folder_name(params, prefix="flux")
        assert result.startswith("flux_")

    def test_same_params_always_produce_same_name(self):
        """Same params always produce the same folder name (deterministic)"""
        params = {
            "network_dim": 64,
            "network_alpha": 16,
            "learning_rate": 1e-4,
            "optimizer": "Prodigy",
        }
        name1 = generate_folder_name(params)
        name2 = generate_folder_name(params)
        assert name1 == name2

    def test_different_order_same_params_produce_same_name(self):
        """Dict key order doesn't matter (sorted alphabetically)"""
        params1 = {"network_dim": 32, "network_alpha": 16}
        params2 = {"network_alpha": 16, "network_dim": 32}
        name1 = generate_folder_name(params1)
        name2 = generate_folder_name(params2)
        assert name1 == name2

    def test_compact_names_stay_under_100_chars(self):
        """Even with many params, folder names should stay well under 100 chars."""
        params = {
            "batch_size": 4,
            "epochs": 20,
            "learning_rate": 1e-4,
            "mixed_precision": "bf16",
            "network_alpha": 8,
            "network_dim": 16,
            "optimizer": "adamw8bit",
            "resolution": 1024,
            "scheduler": "constant",
            "timestep_sampling": "sigmoid",
        }
        result = generate_folder_name(params)
        assert len(result) < 100

    def test_bool_values_abbreviated(self):
        """Boolean values use t/f shorthand."""
        params = {"network_dim": 32, "some_flag": True}
        result = generate_folder_name(params)
        assert "sofl-t" in result  # some_flag -> sofl (2 chars per word)

    def test_scheduler_values_abbreviated(self):
        """Common scheduler values are abbreviated."""
        params = {"scheduler": "cosine"}
        result = generate_folder_name(params)
        assert "sch-cos" in result

    def test_mixed_precision_abbreviated(self):
        """Mixed precision values are abbreviated."""
        params = {"mixed_precision": "bf16"}
        result = generate_folder_name(params)
        assert "mp-bf16" in result
