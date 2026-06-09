"""Find LoRA checkpoint files in a permutation output folder."""

import re
from pathlib import Path


class NoLoraFoundError(FileNotFoundError):
    """Raised when no .safetensors checkpoint is found in the folder."""


def _epoch_key(name: str) -> int:
    """Extract the numeric epoch suffix from a checkpoint filename.

    E.g. ``my_lora-000010.safetensors`` → 10
    """
    match = re.search(r"-(\d+)\.safetensors$", name)
    if match:
        return int(match.group(1))
    return 0


def find_latest_lora(folder: str) -> Path:
    """Return the ``.safetensors`` file with the highest epoch number in *folder*.

    Raises :class:`NoLoraFoundError` if no safetensors files exist.
    """
    folder_path = Path(folder)
    safetensors_files = list(folder_path.glob("*.safetensors"))

    if not safetensors_files:
        raise NoLoraFoundError(
            f"No .safetensors files found in {folder_path}"
        )

    return max(safetensors_files, key=lambda p: _epoch_key(p.name))
