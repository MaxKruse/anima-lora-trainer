"""Tests for scripts/training_chart.py — metrics collection and chart generation."""

import math
import tempfile
from pathlib import Path

import pytest

from scripts.training_chart import (
    TrainingMetricsCollector,
    generate_ascii_chart,
    generate_png_chart,
    print_training_summary,
    _smooth,
)


# ── _smooth ──────────────────────────────────────────────────────────────

class TestSmooth:
    def test_shorter_than_window_returns_copy(self):
        assert _smooth([1, 2, 3], window=5) == [1, 2, 3]

    def test_basic_smoothing(self):
        vals = [1.0, 1.0, 10.0, 1.0, 1.0]
        result = _smooth(vals, window=3)
        # Middle value (10.0) should be pulled toward neighbors
        assert result[2] < 10.0
        assert result[2] > 1.0


# ── generate_ascii_chart ─────────────────────────────────────────────────

class TestGenerateAsciiChart:
    def test_empty_data(self):
        output = generate_ascii_chart([], [], "Empty")
        assert "no data" in output

    def test_produces_ascii_only(self):
        steps = list(range(1, 101))
        values = [1.0 / i for i in steps]
        output = generate_ascii_chart(steps, values, "Test")
        # All chars should be ASCII (codepoint < 128)
        for ch in output:
            assert ord(ch) < 128, f"Non-ASCII char found: {ch!r}"

    def test_contains_title(self):
        output = generate_ascii_chart([1, 2, 3], [0.5, 0.3, 0.1], "MyTitle")
        assert "MyTitle" in output

    def test_contains_stats(self):
        output = generate_ascii_chart([1, 2, 3], [0.5, 0.3, 0.1], "T")
        assert "min=" in output
        assert "max=" in output
        assert "last=" in output
        assert "steps=" in output

    def test_chart_has_plot_characters(self):
        steps = list(range(1, 51))
        values = [1.0 - i / 50 for i in steps]
        output = generate_ascii_chart(steps, values, "T")
        assert "*" in output  # plot marker


# ── generate_png_chart ───────────────────────────────────────────────────

class TestGeneratePngChart:
    def test_creates_png_file(self):
        steps = list(range(1, 101))
        loss = [1.0 - i / 100 for i in steps]
        lr = [0.001 * (0.99 ** i) for i in steps]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chart.png"
            generate_png_chart(steps, loss, lr, out, title="Test")
            assert out.exists()
            assert out.stat().st_size > 0

    def test_without_lr_values(self):
        steps = list(range(1, 51))
        loss = [0.5 - i / 100 for i in steps]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chart_nolr.png"
            generate_png_chart(steps, loss, None, out, title="No LR")
            assert out.exists()

    def test_creates_parent_dirs(self):
        steps = [1, 2, 3]
        loss = [0.5, 0.3, 0.1]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "sub" / "chart.png"
            generate_png_chart(steps, loss, None, out)
            assert out.exists()


# ── TrainingMetricsCollector ─────────────────────────────────────────────

class TestTrainingMetricsCollector:
    def test_patch_and_unpatch_restores_original(self):
        """Verify that patch/unpatch restores the original method."""

        class FakeTrainer:
            @staticmethod
            def generate_step_logs(*args, **kwargs):
                return {"loss/average": 0.5}

        original = FakeTrainer.generate_step_logs

        collector = TrainingMetricsCollector()
        collector.patch(FakeTrainer, lambda: 1)
        assert FakeTrainer.generate_step_logs is not original

        collector.unpatch()
        assert FakeTrainer.generate_step_logs is original

    def test_captures_metrics_from_logs(self):
        """Verify that calling the patched method captures metrics."""

        class FakeTrainer:
            @staticmethod
            def generate_step_logs(
                self_ref, args, current_loss, avr_loss, lr_scheduler,
                lr_descriptions, optimizer=None, keys_scaled=None,
                mean_norm=None, maximum_norm=None,
                mean_grad_norm=None, mean_combined_norm=None,
            ):
                return {
                    "loss/current": float(current_loss),
                    "loss/average": float(avr_loss),
                    "lr/unet": 0.0002,
                }

        class StepHolder:
            value = 0

        collector = TrainingMetricsCollector()
        collector.patch(FakeTrainer, StepHolder)

        # Simulate 3 steps
        for i in range(1, 4):
            StepHolder.value = i
            FakeTrainer.generate_step_logs(
                None, None,  # self_ref, args
                0.5 - i * 0.1,  # current_loss
                0.4 - i * 0.08,  # avr_loss
                None, None,  # lr_scheduler, lr_descriptions
            )

        collector.unpatch()

        assert collector.steps == [1, 2, 3]
        assert len(collector.current_loss) == 3
        assert len(collector.avg_loss) == 3
        assert len(collector.lr_values) == 3
        assert collector.lr_values[0] == 0.0002

    def test_empty_collector(self):
        collector = TrainingMetricsCollector()
        assert collector.steps == []
        assert collector.current_loss == []
        assert collector.avg_loss == []
        assert collector.lr_values == []
