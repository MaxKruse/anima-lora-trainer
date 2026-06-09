"""Tests for dataset TOML generator."""

import os
import pytest
import toml
from scripts.dataset_toml import generate_dataset_toml


class TestGenerateDatasetToml:
    """Test dataset TOML config generation."""

    def test_produces_valid_toml_with_correct_structure(self, tmp_path):
        output_path = tmp_path / "dataset.toml"
        generate_dataset_toml(
            image_dir="/path/to/images",
            batch_size=4,
            num_images=100,
            epochs=10,
            steps_per_epoch=100,
            output_path=str(output_path),
        )

        assert output_path.exists()
        data = toml.load(str(output_path))

        # Check general section
        assert "general" in data
        assert data["general"]["shuffle_caption"] is True
        assert data["general"]["caption_extension"] == ".txt"
        assert data["general"]["keep_tokens"] == 1

        # Check datasets section
        assert "datasets" in data
        dataset = data["datasets"][0]
        assert dataset["resolution"] == 512
        assert dataset["batch_size"] == 4

        # Check subsets
        subset = dataset["subsets"][0]
        assert subset["image_dir"] == "/path/to/images"

    def test_num_repeats_calculated_correctly(self, tmp_path):
        output_path = tmp_path / "dataset.toml"
        # 10 images, 100 steps per epoch → ceil(100/10) = 10 repeats
        generate_dataset_toml(
            image_dir="/images",
            batch_size=1,
            num_images=10,
            epochs=5,
            steps_per_epoch=100,
            output_path=str(output_path),
        )

        data = toml.load(str(output_path))
        subset = data["datasets"][0]["subsets"][0]
        assert subset["num_repeats"] == 10

    def test_num_repeats_with_remainder(self, tmp_path):
        """When num_images doesn't divide evenly, ceil up."""
        output_path = tmp_path / "dataset.toml"
        # 7 images, 100 steps → ceil(100/7) = 15 repeats
        generate_dataset_toml(
            image_dir="/images",
            batch_size=1,
            num_images=7,
            epochs=1,
            steps_per_epoch=100,
            output_path=str(output_path),
        )

        data = toml.load(str(output_path))
        subset = data["datasets"][0]["subsets"][0]
        assert subset["num_repeats"] == 15  # ceil(100/7) = 14.286 → 15

    def test_sets_caption_extension_txt(self, tmp_path):
        output_path = tmp_path / "dataset.toml"
        generate_dataset_toml(
            image_dir="/images",
            batch_size=1,
            num_images=50,
            epochs=10,
            steps_per_epoch=50,
            output_path=str(output_path),
        )

        data = toml.load(str(output_path))
        assert data["general"]["caption_extension"] == ".txt"

    def test_sets_shuffle_caption_true(self, tmp_path):
        output_path = tmp_path / "dataset.toml"
        generate_dataset_toml(
            image_dir="/images",
            batch_size=1,
            num_images=50,
            epochs=10,
            steps_per_epoch=50,
            output_path=str(output_path),
        )

        data = toml.load(str(output_path))
        assert data["general"]["shuffle_caption"] is True

    def test_writes_to_specified_output_path(self, tmp_path):
        custom_path = tmp_path / "custom" / "dir" / "my_dataset.toml"
        generate_dataset_toml(
            image_dir="/images",
            batch_size=2,
            num_images=20,
            epochs=5,
            steps_per_epoch=20,
            output_path=str(custom_path),
        )

        assert custom_path.exists()

    def test_custom_resolution(self, tmp_path):
        output_path = tmp_path / "dataset.toml"
        generate_dataset_toml(
            image_dir="/images",
            batch_size=1,
            num_images=50,
            epochs=10,
            steps_per_epoch=50,
            resolution=1024,
            output_path=str(output_path),
        )

        data = toml.load(str(output_path))
        assert data["datasets"][0]["resolution"] == 1024
