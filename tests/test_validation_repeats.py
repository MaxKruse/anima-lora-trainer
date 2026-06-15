"""Tests for repeat-count math used by training setup."""

from scripts.validation import calculate_repeats


def test_calculate_repeats_uses_batch_size():
    # 70 images, 800 steps, batch size 4, target 4 epochs -> ceil(3200 / 280) = 12
    assert calculate_repeats(70, max_steps=800, batch_size=4, target_epochs=4) == 12


def test_calculate_repeats_respects_target_epochs():
    # Higher target epochs should require fewer repeats.
    repeats_4_epochs = calculate_repeats(70, max_steps=800, batch_size=4, target_epochs=4)
    repeats_10_epochs = calculate_repeats(70, max_steps=800, batch_size=4, target_epochs=10)
    
    repeat_10_images_2_epochs_bs_4 = calculate_repeats(10, max_steps=800, batch_size=4, target_epochs=2)
    # repeats = ceil((800 * 4) / (10 * 2)) = ceil(160) = 160
    assert repeat_10_images_2_epochs_bs_4 == 160

    assert repeats_10_epochs < repeats_4_epochs
    assert repeats_10_epochs == 5


def test_calculate_repeats_invalid_inputs_fallback_to_one():
    assert calculate_repeats(0, max_steps=800, batch_size=4, target_epochs=4) == 1
    assert calculate_repeats(10, max_steps=0, batch_size=4, target_epochs=4) == 1
    assert calculate_repeats(10, max_steps=800, batch_size=0, target_epochs=4) == 1
    assert calculate_repeats(10, max_steps=800, batch_size=4, target_epochs=0) == 1
