"""Tests for permutation generator."""

import pytest
from scripts.permutation_generator import generate_permutations


class TestGeneratePermutations:
    """Test generate_permutations function."""

    def test_two_params_produces_four_permutations(self):
        """Given {dim: [1,2], alpha: [1,4]}, produces 2×2 = 4 permutations"""
        params = {
            "network_dim": [1, 2],
            "network_alpha": [1, 4],
        }
        result = generate_permutations(params)
        assert len(result) == 4

        # Check all combinations exist
        combos = {(p["network_dim"], p["network_alpha"]) for p in result}
        assert combos == {(1, 1), (1, 4), (2, 1), (2, 4)}

    def test_three_params_produces_six_permutations(self):
        """Given {dim: [1,2,3], alpha: [1,4], lr: [1e-4]}, produces 3×2×1 = 6 permutations"""
        params = {
            "network_dim": [1, 2, 3],
            "network_alpha": [1, 4],
            "learning_rate": [1e-4],
        }
        result = generate_permutations(params)
        assert len(result) == 6

    def test_percent_alpha_resolves_to_dim_fraction(self):
        """25% alpha resolves to dim * 0.25 for each permutation's dim value"""
        params = {
            "network_dim": [4, 8, 16],
            "network_alpha": ["25%"],
        }
        result = generate_permutations(params)
        assert len(result) == 3

        # Alpha should be 25% of dim
        for perm in result:
            expected_alpha = perm["network_dim"] * 0.25
            assert perm["network_alpha"] == expected_alpha

    def test_each_permutation_is_flat_dict(self):
        """Each permutation is a flat dict of {param_name: resolved_value}"""
        params = {
            "network_dim": [32],
            "network_alpha": [16],
            "learning_rate": [1e-4],
        }
        result = generate_permutations(params)
        assert len(result) == 1

        perm = result[0]
        assert isinstance(perm, dict)
        assert perm["network_dim"] == 32
        assert perm["network_alpha"] == 16
        assert perm["learning_rate"] == 1e-4

    def test_large_input_produces_correct_count(self):
        """Large input (8×4×3×4×2×2×2) produces exactly 3,072 permutations"""
        params = {
            "network_dim": [1, 2, 4, 8, 16, 32, 64, 128],  # 8
            "network_alpha": [1, 2, 4, 8],  # 4
            "learning_rate": [1e-5, 5e-5, 1e-4],  # 3
            "optimizer": ["AdamW8Bit", "AdamW", "Prodigy", "Lion"],  # 4
            "scheduler": ["cosine", "constant"],  # 2
            "mixed_precision": ["bf16", "fp16"],  # 2
            "timestep_sampling": ["sigmoid", "uniform"],  # 2
        }
        result = generate_permutations(params)
        assert len(result) == 8 * 4 * 3 * 4 * 2 * 2 * 2  # 3,072

    def test_single_param_single_value(self):
        """Single param with single value produces one permutation"""
        params = {"network_dim": [32]}
        result = generate_permutations(params)
        assert len(result) == 1
        assert result[0] == {"network_dim": 32}

    def test_percent_with_non_numeric_base_raises_error(self):
        """Percent reference to non-numeric param raises ValueError"""
        params = {
            "optimizer": ["AdamW8Bit"],
            "network_alpha": ["25%"],  # 25% of optimizer? Invalid.
        }
        with pytest.raises(ValueError, match="reference"):
            generate_permutations(params)
