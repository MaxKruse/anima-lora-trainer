"""Tests for GPU detection module (scripts/setup_env.py)."""

import pytest
from scripts.setup_env import detect_gpu, NvidiaSmiError, UnsupportedGpuError


class TestDetectGpu:
    """Test GPU classification from nvidia-smi output."""

    def test_rtx_50_series_returns_blackwell_cu130(self):
        output = "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 5090"
        result = detect_gpu(output)
        assert result["cuda"] == "cu130"
        assert result["series"] == "blackwell"

    def test_rtx_40_series_returns_ada_cu128(self):
        output = "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 4090"
        result = detect_gpu(output)
        assert result["cuda"] == "cu128"
        assert result["series"] == "ada"

    def test_rtx_30_series_returns_ampere_cu128(self):
        output = "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 3080"
        result = detect_gpu(output)
        assert result["cuda"] == "cu128"
        assert result["series"] == "ampere"

    def test_unrecognized_gpu_raises_error(self):
        output = "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce GTX 1080"
        with pytest.raises(UnsupportedGpuError):
            detect_gpu(output)

    def test_empty_output_raises_nvidia_smi_error(self):
        with pytest.raises(NvidiaSmiError):
            detect_gpu("")

    def test_none_output_raises_nvidia_smi_error(self):
        with pytest.raises(NvidiaSmiError):
            detect_gpu(None)  # type: ignore
