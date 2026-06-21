"""Tests for repeat-count math used by training setup."""

from scripts.validation import calculate_repeats


def test_calculate_repeats_basic():
    """12 images, bs=4, target 12 steps/epoch -> ceil(48/12) = 4."""
    assert calculate_repeats(12, batch_size=4) == 4


def test_calculate_repeats_few_images():
    """5 images, bs=4, target 12 steps/epoch -> ceil(48/5) = 10."""
    assert calculate_repeats(5, batch_size=4) == 10


def test_calculate_repeats_many_images():
    """50 images, bs=4, target 12 steps/epoch -> ceil(48/50) = 1."""
    assert calculate_repeats(50, batch_size=4) == 1


def test_calculate_repeats_respects_batch_size():
    """Same image count, different batch sizes produce different repeats."""
    r_bs1 = calculate_repeats(20, batch_size=1)
    r_bs4 = calculate_repeats(20, batch_size=4)
    assert r_bs4 > r_bs1


def test_calculate_repeats_custom_target():
    """Higher target steps/epoch requires more repeats."""
    r_10 = calculate_repeats(17, batch_size=4, target_steps_per_epoch=10)
    r_15 = calculate_repeats(17, batch_size=4, target_steps_per_epoch=15)
    # ceil(40/17)=3 vs ceil(60/17)=4
    assert r_15 > r_10


def test_calculate_repeats_invalid_inputs_fallback_to_one():
    assert calculate_repeats(0, batch_size=4) == 1
    assert calculate_repeats(10, batch_size=0) == 1
