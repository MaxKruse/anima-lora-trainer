"""Tests for permutation folder namer."""

import pytest
from scripts.permutation_namer import generate_folder_name


class TestGenerateFolderName:
    """Test generate_folder_name function."""

    def test_basic_params_produce_correct_name(self):
        """Params sorted alphabetically: learning-rate < network-alpha < network-dim"""
        params = {
            "network_dim": 1,
            "network_alpha": 1,
            "learning_rate": 1e-4,
        }
        result = generate_folder_name(params)
        assert result == "anima_learning-rate-1e-4_network-alpha-1_network-dim-1"

    def test_float_values_use_scientific_notation(self):
        """Float values use scientific notation consistently"""
        params = {
            "network_dim": 32,
            "learning_rate": 5e-4,
        }
        result = generate_folder_name(params)
        assert "5e-4" in result

    def test_params_sorted_alphabetically(self):
        """Params sorted alphabetically for deterministic naming"""
        params = {
            "learning_rate": 1e-4,
            "network_dim": 32,
            "network_alpha": 16,
        }
        result = generate_folder_name(params)
        # Alphabetical: learning_rate < network_alpha < network_dim
        lr_pos = result.index("learning-rate")
        alpha_pos = result.index("network-alpha")
        dim_pos = result.index("network-dim")
        assert lr_pos < alpha_pos < dim_pos

    def test_string_params_included(self):
        """String param values like optimizer are included"""
        params = {
            "network_dim": 32,
            "optimizer": "AdamW8Bit",
        }
        result = generate_folder_name(params)
        assert "optimizer-adamw8bit" in result

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
