"""GPU detection and pyproject.toml generation for LoRA Matrix Trainer."""

from typing import Optional


class NvidiaSmiError(Exception):
    """Raised when nvidia-smi output is empty or unavailable."""


class UnsupportedGpuError(Exception):
    """Raised when no recognized GPU is found in nvidia-smi output."""


def detect_gpu(nvidia_smi_output: Optional[str]) -> dict:
    """Parse nvidia-smi output and classify GPU into CUDA tier.

    Returns:
        dict with keys: cuda (str), series (str), gpu_name (str)

    Raises:
        NvidiaSmiError: If output is empty or None.
        UnsupportedGpuError: If no recognized GPU series found.
    """
    if not nvidia_smi_output or not nvidia_smi_output.strip():
        raise NvidiaSmiError("nvidia-smi returned no output")

    # Extract GPU name from output
    gpu_name = _extract_gpu_name(nvidia_smi_output)

    # Classify by series
    if "RTX 50" in gpu_name:
        return {"cuda": "cu130", "series": "blackwell", "gpu_name": gpu_name}
    elif "RTX 40" in gpu_name:
        return {"cuda": "cu128", "series": "ada", "gpu_name": gpu_name}
    elif "RTX 30" in gpu_name:
        return {"cuda": "cu128", "series": "ampere", "gpu_name": gpu_name}
    else:
        raise UnsupportedGpuError(
            f"GPU '{gpu_name}' is not supported. "
            "Requires RTX 30-series (Ampere) or newer."
        )


def _extract_gpu_name(nvidia_smi_output: str) -> str:
    """Extract the GPU name from nvidia-smi output.

    Searches for 'GPU Name:' line or falls back to 'GeForce RTX' pattern.
    """
    for line in nvidia_smi_output.splitlines():
        if "GPU Name" in line and ":" in line:
            return line.split(":", 1)[1].strip()

    # Fallback: search for GeForce pattern
    for line in nvidia_smi_output.splitlines():
        if "GeForce" in line:
            return line.strip()

    return nvidia_smi_output.strip()
