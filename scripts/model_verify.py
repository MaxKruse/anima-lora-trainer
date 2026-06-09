"""Safetensors model file verification."""

import json
import struct
from pathlib import Path


def verify_safetensors(file_path: str) -> bool:
    """Verify a .safetensors file has a valid header.

    The safetensors format is:
    - 8 bytes: uint64 little-endian header length
    - N bytes: JSON header describing tensors
    - Remaining bytes: tensor data

    Args:
        file_path: Path to the .safetensors file.

    Returns:
        True if the file has a valid safetensors header.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = path.stat().st_size

    # Need at least 8 bytes for header length
    if file_size < 8:
        return False

    # Read header length
    with open(path, 'rb') as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]

    # Header length must be reasonable
    if header_len == 0 or header_len > file_size - 8:
        return False

    # Read and parse JSON header
    with open(path, 'rb') as f:
        f.seek(8)
        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    # Must be a dict (metadata or tensor descriptions)
    if not isinstance(header, dict):
        return False

    return True
