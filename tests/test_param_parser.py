"""Tests for parameter range parser."""

import pytest
from scripts.param_parser import parse_param_range


class TestParseParamRange:
    """Test parse_param_range function."""

    def test_parse_integers(self):
        """'1,2,3' → [1, 2, 3] (integers)"""
        result = parse_param_range("1,2,3")
        assert result == [1, 2, 3]
        assert all(isinstance(v, int) for v in result)

    def test_parse_floats(self):
        """'1e-4,5e-4,1e-3' → [1e-4, 5e-4, 1e-3] (floats)"""
        result = parse_param_range("1e-4,5e-4,1e-3")
        assert result == [1e-4, 5e-4, 1e-3]
        assert all(isinstance(v, float) for v in result)

    def test_parse_strings(self):
        """'AdamW8Bit,Prodigy' → ['AdamW8Bit', 'Prodigy'] (strings)"""
        result = parse_param_range("AdamW8Bit,Prodigy")
        assert result == ["AdamW8Bit", "Prodigy"]
        assert all(isinstance(v, str) for v in result)

    def test_preserves_percent_marker(self):
        """'1,4,8,25%' → [1, 4, 8, '25%'] (preserves % marker for later resolution)"""
        result = parse_param_range("1,4,8,25%")
        assert result == [1, 4, 8, "25%"]

    def test_empty_string_raises_value_error(self):
        """Empty string raises ValueError"""
        with pytest.raises(ValueError, match="empty"):
            parse_param_range("")

    def test_whitespace_only_raises_value_error(self):
        """Whitespace-only string raises ValueError"""
        with pytest.raises(ValueError, match="empty"):
            parse_param_range("   ")

    def test_single_value(self):
        """Single value returns list with one element"""
        result = parse_param_range("42")
        assert result == [42]

    def test_mixed_int_float(self):
        """Mixed int and float values"""
        result = parse_param_range("1,0.5,3")
        assert result == [1, 0.5, 3]

    def test_whitespace_around_values(self):
        """Whitespace around values is trimmed"""
        result = parse_param_range("1 , 2 , 3")
        assert result == [1, 2, 3]

    def test_negative_numbers(self):
        """Negative numbers are parsed correctly"""
        result = parse_param_range("-1,0,1")
        assert result == [-1, 0, 1]
