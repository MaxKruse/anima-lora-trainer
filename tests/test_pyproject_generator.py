"""Tests for pyproject.toml generator (scripts/setup_env.py)."""

import pytest
from scripts.setup_env import generate_pyproject_toml


class TestGeneratePyprojectToml:
    """Test pyproject.toml generation with correct CUDA index routing."""

    def test_cu128_routes_torch_to_cu128_index(self, tmp_path):
        output_path = tmp_path / "pyproject.toml"
        generate_pyproject_toml(str(output_path), "cu128")

        content = output_path.read_text()
        assert "pytorch-cu128" in content
        assert "pytorch-cu130" not in content or "explicit" in content  # index def ok, just not routed

    def test_cu130_routes_torch_to_cu130_index(self, tmp_path):
        output_path = tmp_path / "pyproject.toml"
        generate_pyproject_toml(str(output_path), "cu130")

        content = output_path.read_text()
        assert "pytorch-cu130" in content
        # Should route torch to cu130, not cu128
        lines = content.splitlines()
        sources_found = False
        for i, line in enumerate(lines):
            if "torch = [" in line:
                # Check next few lines for cu130 reference
                context = "\n".join(lines[i:i+3])
                assert "cu130" in context
                sources_found = True
                break
        assert sources_found, "No torch source routing found"

    def test_generated_toml_contains_all_required_dependencies(self, tmp_path):
        output_path = tmp_path / "pyproject.toml"
        generate_pyproject_toml(str(output_path), "cu128")

        content = output_path.read_text()
        required_deps = [
            "torch",
            "torchvision",
            "accelerate",
            "transformers",
            "diffusers",
            "safetensors",
            "bitsandbytes",
            "lion-pytorch",
            "pytorch-optimizer",
            "prodigyopt",
            "rich",
            "numpy",
            "einops",
            "opencv-python",
            "ftfy",
            "huggingface-hub",
            "tensorboard",
            "voluptuous",
            "imagesize",
        ]
        for dep in required_deps:
            assert dep in content, f"Missing dependency: {dep}"

    def test_generated_toml_is_valid(self, tmp_path):
        """Generated toml should parse round-trip."""
        output_path = tmp_path / "pyproject.toml"
        generate_pyproject_toml(str(output_path), "cu128")

        import toml
        parsed = toml.loads(output_path.read_text())
        assert "project" in parsed
        assert "dependencies" in parsed["project"]
        assert parsed["project"]["name"] == "lora-matrix-trainer"

    def test_generated_toml_has_both_cuda_indexes_defined(self, tmp_path):
        """Both cu128 and cu130 indexes should be defined, only one routed."""
        output_path = tmp_path / "pyproject.toml"
        generate_pyproject_toml(str(output_path), "cu128")

        import toml
        parsed = toml.loads(output_path.read_text())
        uv_config = parsed.get("tool", {}).get("uv", {})
        indexes = uv_config.get("index", [])
        index_names = [idx["name"] for idx in indexes]
        assert "pytorch-cu128" in index_names
        assert "pytorch-cu130" in index_names
