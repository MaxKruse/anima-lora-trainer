"""Training calculation math - the full chain from inputs to effective epochs.

This module tests every piece of the auto-calculation pipeline:

  1. calculate_max_steps(batch_size)
     Maps batch size to total training steps.

  2. calculate_repeats(num_images, batch_size)
     Maps image count + batch size to repeats per image.

  3. Effective steps_per_epoch
     Derived: (num_images * repeats) / batch_size

  4. Effective epochs
     Derived: max_steps / steps_per_epoch

The goal: no single epoch dominates training statistically.
Steps per epoch should stay in the 10-15 range (target ~12).
"""

import pytest

from scripts.validation import calculate_max_steps, calculate_repeats


# ---------------------------------------------------------------------------
# Helpers - make every test self-documenting
# ---------------------------------------------------------------------------

def _steps_per_epoch(num_images: int, repeats: int, batch_size: int) -> float:
    """How many gradient steps one full pass through the data takes."""
    return (num_images * repeats) / batch_size


def _effective_epochs(max_steps: int, steps_per_epoch: float) -> float:
    """How many full passes through the data fit into max_steps."""
    return max_steps / steps_per_epoch


def _total_samples_seen(max_steps: int, batch_size: int) -> int:
    """Total batch-sized samples the model sees (upper bound)."""
    return max_steps * batch_size


# ===========================================================================
# 1. calculate_max_steps(batch_size)
# ===========================================================================

class TestCalculateMaxSteps:
    """Batch size to total steps mapping.

    Higher batch sizes converge faster, so they need fewer total steps.
    Scaling is slightly less than linear to keep theoretical training
    stable across batch sizes.
    """

    def test_default_sweet_spot(self):
        """bs=4 is the default sweet spot -> 600 steps."""
        assert calculate_max_steps(4) == 600

    def test_batch_size_3(self):
        """bs=3 needs more steps than bs=4 -> 800 steps."""
        assert calculate_max_steps(3) == 800

    def test_batch_size_2(self):
        """bs=2 needs even more steps -> 1000 steps."""
        assert calculate_max_steps(2) == 1000

    def test_batch_size_1(self):
        """bs=1 needs the most steps -> 1600 steps."""
        assert calculate_max_steps(1) == 1600

    def test_unknown_batch_size_falls_back_to_default(self):
        """bs=5, bs=8, bs=16 etc. fall back to the bs=4 value."""
        assert calculate_max_steps(5) == 600
        assert calculate_max_steps(8) == 600
        assert calculate_max_steps(16) == 600

    def test_scaling_is_less_than_linear(self):
        """
        Linear scaling from bs=4 (600 steps) would give:
          bs=3 -> 800  (linear: 800)  ✓ exact
          bs=2 -> 1200 (linear: 1200) ✗ actual 1000 (less)
          bs=1 -> 2400 (linear: 2400) ✗ actual 1600 (less)

        Less than linear prevents excessive training at low batch sizes.
        """
        assert calculate_max_steps(2) < calculate_max_steps(4) * 2   # 1000 < 1200
        assert calculate_max_steps(1) < calculate_max_steps(4) * 4   # 1600 < 2400

    def test_total_samples_seen_decreases_slightly_with_smaller_bs(self):
        """
        Total samples = max_steps * batch_size.
        Should be roughly stable (not exactly, since scaling is sub-linear).
        """
        samples_bs4 = _total_samples_seen(calculate_max_steps(4), 4)   # 2400
        samples_bs3 = _total_samples_seen(calculate_max_steps(3), 3)   # 2400
        samples_bs2 = _total_samples_seen(calculate_max_steps(2), 2)   # 2000
        samples_bs1 = _total_samples_seen(calculate_max_steps(1), 1)   # 1600

        # bs=4 and bs=3 see the same total samples
        assert samples_bs4 == samples_bs3 == 2400
        # bs=2 sees fewer (sub-linear scaling)
        assert samples_bs2 < samples_bs4
        # bs=1 sees even fewer
        assert samples_bs1 < samples_bs2


# ===========================================================================
# 2. calculate_repeats(num_images, batch_size)
# ===========================================================================

class TestCalculateRepeats:
    """Repeats ensure each epoch has ~12 gradient steps (10-15 range).

    Formula: repeats = ceil(12 * batch_size / num_images)
    Result:  steps_per_epoch = (num_images * repeats) / batch_size ≈ 12
    """

    # --- Core formula correctness ---

    def test_exact_division(self):
        """
        12 images, bs=4: repeats = ceil(12*4/12) = ceil(4) = 4
        steps_per_epoch = (12*4)/4 = 12  (exact target)
        """
        repeats = calculate_repeats(12, batch_size=4)
        assert repeats == 4
        assert _steps_per_epoch(12, repeats, 4) == 12.0

    def test_needs_rounding_up(self):
        """
        10 images, bs=4: repeats = ceil(12*4/10) = ceil(4.8) = 5
        steps_per_epoch = (10*5)/4 = 12.5  (within 10-15 range)
        """
        repeats = calculate_repeats(10, batch_size=4)
        assert repeats == 5
        assert 10 <= _steps_per_epoch(10, repeats, 4) <= 15

    def test_few_images_high_repeats(self):
        """
        5 images, bs=4: repeats = ceil(12*4/5) = ceil(9.6) = 10
        steps_per_epoch = (5*10)/4 = 12.5
        """
        repeats = calculate_repeats(5, batch_size=4)
        assert repeats == 10
        assert _steps_per_epoch(5, repeats, 4) == 12.5

    def test_many_images_low_repeats(self):
        """
        20 images, bs=4: repeats = ceil(12*4/20) = ceil(2.4) = 3
        steps_per_epoch = (20*3)/4 = 15
        """
        repeats = calculate_repeats(20, batch_size=4)
        assert repeats == 3
        assert _steps_per_epoch(20, repeats, 4) == 15.0

    def test_very_many_images_minimum_one_repeat(self):
        """
        50 images, bs=4: repeats = ceil(12*4/50) = ceil(0.96) = 1
        steps_per_epoch = (50*1)/4 = 12.5
        """
        repeats = calculate_repeats(50, batch_size=4)
        assert repeats == 1
        assert _steps_per_epoch(50, repeats, 4) == 12.5

    # --- Batch size interaction ---

    def test_steps_per_epoch_stable_across_batch_sizes(self):
        """
        Same image count, different batch sizes.
        Steps per epoch should stay near 12 regardless of batch size.
        """
        num_images = 15

        for bs in [1, 2, 3, 4]:
            repeats = calculate_repeats(num_images, batch_size=bs)
            spe = _steps_per_epoch(num_images, repeats, bs)
            assert 10 <= spe <= 15, (
                f"bs={bs}: steps_per_epoch={spe} "
                f"(repeats={repeats}, images={num_images})"
            )

    def test_repeats_increase_with_batch_size(self):
        """
        Larger batch size processes more images per step,
        so we need more repeats to fill an epoch.
        """
        num_images = 12
        r_bs1 = calculate_repeats(num_images, batch_size=1)  # ceil(144/12)=1
        r_bs4 = calculate_repeats(num_images, batch_size=4)  # ceil(48/12)=4
        assert r_bs4 >= r_bs1

    # --- Edge cases ---

    def test_single_image(self):
        """1 image, bs=4: repeats = ceil(48/1) = 48."""
        repeats = calculate_repeats(1, batch_size=4)
        assert repeats == 48
        assert _steps_per_epoch(1, repeats, 4) == 12.0

    def test_zero_images_fallback(self):
        """0 images is invalid -> fallback to 1 repeat."""
        assert calculate_repeats(0, batch_size=4) == 1

    def test_zero_batch_size_fallback(self):
        """0 batch size is invalid -> fallback to 1 repeat."""
        assert calculate_repeats(10, batch_size=0) == 1

    def test_negative_images_fallback(self):
        """Negative images is invalid -> fallback to 1 repeat."""
        assert calculate_repeats(-5, batch_size=4) == 1

    def test_custom_target_steps(self):
        """
        Lower target = fewer repeats.
        target=10: ceil(10*4/17) = ceil(2.35) = 3
        target=15: ceil(15*4/17) = ceil(3.53) = 4
        """
        r_low = calculate_repeats(17, batch_size=4, target_steps_per_epoch=10)
        r_high = calculate_repeats(17, batch_size=4, target_steps_per_epoch=15)
        assert r_high > r_low


# ===========================================================================
# 3. Full pipeline - end-to-end scenarios
# ===========================================================================

class TestFullPipeline:
    """End-to-end: images + batch_size -> repeats -> steps_per_epoch -> epochs.

    These tests verify the complete chain produces sensible training configs.
    """

    @pytest.mark.parametrize(
        "num_images,batch_size",
        [
            (12, 4),   # typical character dataset
            (15, 4),   # slightly larger
            (20, 4),   # upper end
            (12, 2),   # typical with lower bs
            (15, 1),   # typical with bs=1
            (8, 4),    # small dataset
            (25, 4),   # large dataset
        ],
    )
    def test_steps_per_epoch_always_in_range(self, num_images, batch_size):
        """
        For any reasonable image count and batch size,
        steps_per_epoch should fall in the 10-15 range.
        """
        repeats = calculate_repeats(num_images, batch_size)
        spe = _steps_per_epoch(num_images, repeats, batch_size)

        assert 10 <= spe <= 15, (
            f"images={num_images}, bs={batch_size}: "
            f"repeats={repeats}, steps_per_epoch={spe}"
        )

    @pytest.mark.parametrize(
        "num_images,batch_size",
        [
            (12, 4),
            (15, 4),
            (20, 4),
            (12, 2),
            (15, 1),
        ],
    )
    def test_effective_epochs_are_reasonable(self, num_images, batch_size):
        """
        Effective epochs should be in a reasonable range (15-120).
        Too few = undertrained, too many = overtrained.
        bs=1 with many images pushes higher (1600 steps / ~12 spe = ~133 max).
        """
        max_steps = calculate_max_steps(batch_size)
        repeats = calculate_repeats(num_images, batch_size)
        spe = _steps_per_epoch(num_images, repeats, batch_size)
        epochs = _effective_epochs(max_steps, spe)

        assert 15 <= epochs <= 120, (
            f"images={num_images}, bs={batch_size}: "
            f"max_steps={max_steps}, repeats={repeats}, "
            f"steps_per_epoch={spe:.1f}, effective_epochs={epochs:.1f}"
        )

    def test_full_chain_12_images_bs4(self):
        """
        Concrete example: 12 images, batch_size=4 (most common case).

        max_steps    = 600  (from bs=4)
        repeats      = 4    (ceil(48/12) = 4)
        steps/epoch  = 12   ((12*4)/4 = 12)
        epochs       = 50   (600/12 = 50)
        """
        num_images = 12
        bs = 4

        max_steps = calculate_max_steps(bs)
        repeats = calculate_repeats(num_images, bs)
        spe = _steps_per_epoch(num_images, repeats, bs)
        epochs = _effective_epochs(max_steps, spe)

        assert max_steps == 600
        assert repeats == 4
        assert spe == 12.0
        assert epochs == 50.0

    def test_full_chain_15_images_bs2(self):
        """
        Concrete example: 15 images, batch_size=2.

        repeats      = ceil(12*2/15) = ceil(1.6) = 2
        max_steps    = 1000 (from bs=2)
        steps/epoch  = (15*2)/2 = 15
        epochs       = 1000/15 = 66.7
        """
        num_images = 15
        bs = 2

        max_steps = calculate_max_steps(bs)
        repeats = calculate_repeats(num_images, bs)
        spe = _steps_per_epoch(num_images, repeats, bs)
        epochs = _effective_epochs(max_steps, spe)

        assert max_steps == 1000
        assert repeats == 2
        assert spe == 15.0
        assert abs(epochs - 1000 / 15.0) < 0.01  # ~66.7

    def test_full_chain_20_images_bs4(self):
        """
        Concrete example: 20 images, batch_size=4.

        max_steps    = 600  (from bs=4)
        repeats      = 3    (ceil(48/20) = 3)
        steps/epoch  = 15   ((20*3)/4 = 15)
        epochs       = 40   (600/15 = 40)
        """
        num_images = 20
        bs = 4

        max_steps = calculate_max_steps(bs)
        repeats = calculate_repeats(num_images, bs)
        spe = _steps_per_epoch(num_images, repeats, bs)
        epochs = _effective_epochs(max_steps, spe)

        assert max_steps == 600
        assert repeats == 3
        assert spe == 15.0
        assert epochs == 40.0

    def test_batch_size_change_preserves_steps_per_epoch(self):
        """
        Changing batch size should NOT change steps_per_epoch.
        The repeats auto-adjust to keep it stable.
        """
        num_images = 15

        spe_bs4 = _steps_per_epoch(
            num_images,
            calculate_repeats(num_images, 4),
            4,
        )
        spe_bs2 = _steps_per_epoch(
            num_images,
            calculate_repeats(num_images, 2),
            2,
        )
        spe_bs1 = _steps_per_epoch(
            num_images,
            calculate_repeats(num_images, 1),
            1,
        )

        # All should be in the 10-15 range
        assert 10 <= spe_bs4 <= 15
        assert 10 <= spe_bs2 <= 15
        assert 10 <= spe_bs1 <= 15

    def test_batch_size_change_scales_total_epochs(self):
        """
        Smaller batch size -> more max_steps -> more total epochs.
        This is intentional: smaller batches need more passes to converge.
        """
        num_images = 15

        results = {}
        for bs in [1, 2, 3, 4]:
            max_steps = calculate_max_steps(bs)
            repeats = calculate_repeats(num_images, bs)
            spe = _steps_per_epoch(num_images, repeats, bs)
            epochs = _effective_epochs(max_steps, spe)
            results[bs] = epochs

        # Smaller batch size should give more (or equal) epochs
        assert results[1] >= results[2] >= results[3] >= results[4]


# ===========================================================================
# 4. Statistical balance - no single epoch dominates
# ===========================================================================

class TestStatisticalBalance:
    """Verify the core principle: no single epoch dominates training.

    Each epoch should contribute roughly the same fraction of total training.
    With steps_per_epoch in [10, 15] and max_steps=600, each epoch is
    about 2-6% of total training - which is statistically balanced.
    """

    def test_epoch_contribution_fraction(self):
        """
        Each epoch should be a small fraction of total training.
        (steps_per_epoch / max_steps * 100)

        With bs=1 and max_steps=1600, the fraction can be as low as ~0.75%.
        With bs=4 and max_steps=600, the fraction tops out at 2.5% (spe=15).
        """
        for num_images in [8, 10, 12, 15, 20, 25]:
            for bs in [1, 2, 3, 4]:
                max_steps = calculate_max_steps(bs)
                repeats = calculate_repeats(num_images, bs)
                spe = _steps_per_epoch(num_images, repeats, bs)
                fraction = spe / max_steps * 100

                assert 0.5 <= fraction <= 15, (
                    f"images={num_images}, bs={bs}: "
                    f"epoch is {fraction:.1f}% of training "
                    f"(spe={spe}, max_steps={max_steps})"
                )

    def test_minimum_epochs_prevents_single_epoch_dominance(self):
        """
        Even with the most extreme valid config,
        we should have at least 5 effective epochs.
        This prevents a single epoch from dominating.
        """
        for num_images in [5, 8, 10, 12, 15, 20, 25, 30]:
            for bs in [1, 2, 3, 4]:
                max_steps = calculate_max_steps(bs)
                repeats = calculate_repeats(num_images, bs)
                spe = _steps_per_epoch(num_images, repeats, bs)
                epochs = _effective_epochs(max_steps, spe)

                assert epochs >= 5, (
                    f"images={num_images}, bs={bs}: "
                    f"only {epochs:.1f} effective epochs "
                    f"(spe={spe}, max_steps={max_steps})"
                )
