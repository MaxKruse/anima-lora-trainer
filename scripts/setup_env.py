"""GPU detection and pyproject.toml generation for LoRA Matrix Trainer."""

from pathlib import Path
from typing import Optional


class NvidiaSmiError(Exception):
    """Raised when nvidia-smi output is empty or unavailable."""


class UnsupportedGpuError(Exception):
    """Raised when no recognized GPU is found in nvidia-smi output."""


def detect_gpu(nvidia_smi_output: Optional[str]) -> dict:
    """Parse nvidia-smi output and classify GPU into CUDA tier.

    Returns:
        dict with keys: cuda (str), series (str), gpu_name (str), cuda_version (str|None)

    Raises:
        NvidiaSmiError: If output is empty or None.
        UnsupportedGpuError: If no recognized GPU series found.
    """
    if not nvidia_smi_output or not nvidia_smi_output.strip():
        raise NvidiaSmiError("nvidia-smi returned no output")

    # Extract GPU name from output
    gpu_name = _extract_gpu_name(nvidia_smi_output)

    # Extract CUDA toolkit version
    cuda_version = _extract_cuda_version(nvidia_smi_output)

    # Classify by series
    if "RTX 50" in gpu_name:
        return {"cuda": "cu130", "series": "blackwell", "gpu_name": gpu_name, "cuda_version": cuda_version}
    elif "RTX 40" in gpu_name:
        return {"cuda": "cu128", "series": "ada", "gpu_name": gpu_name, "cuda_version": cuda_version}
    elif "RTX 30" in gpu_name:
        return {"cuda": "cu128", "series": "ampere", "gpu_name": gpu_name, "cuda_version": cuda_version}
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


def _extract_cuda_version(nvidia_smi_output: str) -> Optional[str]:
    """Extract CUDA toolkit version from nvidia-smi output.

    Searches for 'CUDA Version: X.Y' pattern.
    Returns version string like '12.8' or None if not found.
    """
    import re

    for line in nvidia_smi_output.splitlines():
        match = re.search(r'CUDA Version[:\s]+([\d.]+)', line, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


# --- pyproject.toml generation ---

_REQUIRED_DEPS = [
    "torch",
    "torchvision",
    "accelerate",
    "transformers",
    "diffusers[torch]",
    "safetensors",
    "bitsandbytes",
    "lion-pytorch",
    "pytorch-optimizer",
    "prodigyopt",
    "prodigy-plus-schedule-free",
    "schedulefree",
    "sentencepiece",
    "toml",
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

_PYPROJECT_TEMPLATE = """\
[project]
name = "lora-matrix-trainer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
{deps}
]

[dependency-groups]
dev = [
    "pytest",
    "pytest-cov",
]

# --- CUDA 12.8 index (RTX 30/40 series) ---
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

# --- CUDA 13.0 index (RTX 50 series) ---
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

# --- Active CUDA index routing ---
[tool.uv.sources]
torch = [
    {{ index = "pytorch-{cuda}", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }},
]
torchvision = [
    {{ index = "pytorch-{cuda}", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }},
]
"""


def generate_pyproject_toml(output_path: str, cuda_version: str) -> None:
    """Generate a pyproject.toml with correct CUDA index routing.

    Args:
        output_path: File path to write pyproject.toml.
        cuda_version: Either "cu128" or "cu130".
    """
    if cuda_version not in ("cu128", "cu130"):
        raise ValueError(f"Invalid CUDA version: {cuda_version}. Must be cu128 or cu130.")

    deps_str = "\n".join(f'    "{dep}",\n' for dep in _REQUIRED_DEPS)
    content = _PYPROJECT_TEMPLATE.format(deps=deps_str, cuda=cuda_version)

    Path(output_path).write_text(content)


def run_setup(output_dir: Optional[str] = None) -> dict:
    """Run full setup: detect GPU, generate pyproject.toml.

    Args:
        output_dir: Directory to write pyproject.toml (default: current dir).

    Returns:
        dict with gpu detection results.
    """
    import subprocess

    # Detect GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise NvidiaSmiError(f"nvidia-smi failed: {result.stderr.strip()}")
    except FileNotFoundError:
        raise NvidiaSmiError("nvidia-smi not found. Ensure NVIDIA drivers are installed.")

    gpu_info = detect_gpu(result.stdout)

    # Generate pyproject.toml
    import os
    target_dir = output_dir or os.getcwd()
    toml_path = os.path.join(target_dir, "pyproject.toml")
    generate_pyproject_toml(toml_path, gpu_info["cuda"])

    return {
        **gpu_info,
        "pyproject_path": toml_path,
    }


# --- CLI entry point ---

def main():
    """CLI: python setup_env.py --generate <output_path> <cuda_version>"""
    import sys
    import json

    if len(sys.argv) >= 4 and sys.argv[1] == '--generate':
        output_path = sys.argv[2]
        cuda_version = sys.argv[3]
        generate_pyproject_toml(output_path, cuda_version)
        result = {
            "gpu_name": "detected",
            "series": "unknown",
            "cuda": cuda_version,
            "pyproject_path": output_path,
        }
        print(json.dumps(result))
        return

    # Default: run full setup
    try:
        result = run_setup()
        print(json.dumps(result))
    except (NvidiaSmiError, UnsupportedGpuError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
