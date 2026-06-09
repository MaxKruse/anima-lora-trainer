"""Tests for safetensors model verification."""

import struct
import pytest
from scripts.model_verify import verify_safetensors


class TestVerifySafetensors:
    """Test safetensors header validation."""

    def test_valid_safetensors_file_returns_true(self, tmp_path):
        """A properly formed safetensors file should return True."""
        file_path = tmp_path / "model.safetensors"
        # Create a minimal valid safetensors file:
        # 8 bytes (uint64) header length + JSON header
        header = b'{"__metadata__": {"format": "safetensors"}}'
        header_len = struct.pack('<Q', len(header))
        file_path.write_bytes(header_len + header)

        assert verify_safetensors(str(file_path)) is True

    def test_valid_safetensors_with_tensors_returns_true(self, tmp_path):
        """A safetensors file with tensor data should return True."""
        file_path = tmp_path / "model.safetensors"
        # Header with one tensor entry + dummy data
        header = b'{"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}'
        header_len = struct.pack('<Q', len(header))
        file_path.write_bytes(header_len + header + b'\x00\x00\x00\x00')

        assert verify_safetensors(str(file_path)) is True

    def test_truncated_file_returns_false(self, tmp_path):
        """A file cut short should return False."""
        file_path = tmp_path / "truncated.safetensors"
        # Write header length but not enough data
        header = b'{"weight": {"dtype": "F32"}}'
        header_len = struct.pack('<Q', len(header))
        file_path.write_bytes(header_len + header[:5])  # Truncated header

        assert verify_safetensors(str(file_path)) is False

    def test_empty_file_returns_false(self, tmp_path):
        """An empty file should return False."""
        file_path = tmp_path / "empty.safetensors"
        file_path.write_bytes(b'')

        assert verify_safetensors(str(file_path)) is False

    def test_nonexistent_file_raises_error(self):
        """A file that doesn't exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            verify_safetensors("/nonexistent/path/model.safetensors")

    def test_invalid_header_length_returns_false(self, tmp_path):
        """A file with header length exceeding file size should return False."""
        file_path = tmp_path / "bad_len.safetensors"
        # Claim 1MB header but only write 10 bytes
        header_len = struct.pack('<Q', 1_000_000)
        file_path.write_bytes(header_len + b'short')

        assert verify_safetensors(str(file_path)) is False

    def test_invalid_json_header_returns_false(self, tmp_path):
        """A file with non-JSON header should return False."""
        file_path = tmp_path / "bad_json.safetensors"
        header = b'not valid json at all {{{'
        header_len = struct.pack('<Q', len(header))
        file_path.write_bytes(header_len + header)

        assert verify_safetensors(str(file_path)) is False
