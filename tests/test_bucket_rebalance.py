from pathlib import Path

import cv2
import numpy as np

from scripts.bucket_rebalance import (
    assign_bucket_resolution,
    maybe_build_bucket_rebalance_subset,
    plan_bucket_rebalance,
)


def _write_image(path: Path, width: int, height: int) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok


def _add_caption(path: Path, text: str = "tag1, tag2") -> None:
    path.with_suffix(".txt").write_text(text, encoding="utf-8")


def test_plan_bucket_rebalance_detects_skew():
    counts = {
        (848, 1184): 14,
        (1024, 1024): 2,
        (784, 1312): 2,
        (912, 1136): 2,
    }

    plan = plan_bucket_rebalance(counts, dominance_threshold=0.35, max_augmented_images=64)

    assert plan is not None
    assert plan["dominant_bucket"] == (848, 1184)
    assert plan["dominant_share"] > 0.35
    assert plan["augment_count"] > 0


def test_plan_bucket_rebalance_ensures_min_coverage():
    """When there are many small buckets, augment_count >= number of target buckets."""
    # 14 dominant + 10 buckets with 1 image each = 24 total
    # needed = ceil(14/0.35 - 24) = ceil(40 - 24) = 16
    # But there are 10 target buckets, so min coverage = 10
    # augment_count = max(16, 10) = 16 (needed wins)
    counts = {
        (848, 1184): 14,
    }
    for i in range(10):
        counts[(700 + i * 16, 1300 + (i % 3) * 16)] = 1

    plan = plan_bucket_rebalance(counts, dominance_threshold=0.35, max_augmented_images=64)

    assert plan is not None
    assert plan["augment_count"] >= len(plan["target_buckets"])


def test_plan_bucket_rebalance_min_coverage_when_needed_is_low():
    """When needed < target bucket count, use target bucket count."""
    # 10 dominant + 8 buckets with 1 image each = 18 total
    # needed = ceil(10/0.35 - 18) = ceil(28.57 - 18) = 11
    # target buckets = 8
    # augment_count = max(11, 8) = 11
    counts = {
        (848, 1184): 10,
    }
    for i in range(8):
        counts[(700 + i * 16, 1300 + (i % 2) * 16)] = 1

    plan = plan_bucket_rebalance(counts, dominance_threshold=0.35, max_augmented_images=64)

    assert plan is not None
    assert plan["augment_count"] >= len(plan["target_buckets"])


def test_plan_bucket_rebalance_skips_balanced_distribution():
    counts = {
        (848, 1184): 5,
        (1024, 1024): 4,
        (784, 1312): 4,
        (912, 1136): 3,
    }

    plan = plan_bucket_rebalance(counts, dominance_threshold=0.40, max_augmented_images=64)

    assert plan is None


def test_assign_bucket_resolution_matches_kohya_max_area_behavior():
    # Near-identical AR images above max area should collapse to one bucket.
    sizes = [
        (1536, 2144),
        (1600, 2232),
        (1792, 2500),
        (1216, 1696),
        (1408, 1968),
    ]

    buckets = [assign_bucket_resolution(w, h) for w, h in sizes]
    assert len(set(buckets)) == 1
    assert buckets[0] == (864, 1200)


def test_maybe_build_bucket_rebalance_subset_creates_augmented_data(tmp_path: Path):
    dataset_dir = tmp_path / "img"
    dataset_dir.mkdir(parents=True)

    # Dominant bucket (same quantized size)
    for i in range(8):
        path = dataset_dir / f"dom_{i:02d}.png"
        _write_image(path, 848, 1184)
        _add_caption(path)

    # Minority buckets
    minority = [
        (1024, 1024),
        (784, 1312),
        (912, 1136),
        (768, 1360),
    ]
    for i, (w, h) in enumerate(minority):
        path = dataset_dir / f"min_{i:02d}.png"
        _write_image(path, w, h)
        _add_caption(path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    subset = maybe_build_bucket_rebalance_subset(
        training_images=str(dataset_dir),
        output_dir=output_dir,
        num_repeats=4,
        enabled=True,
        dominance_threshold=0.35,
        max_augmented_images=24,
        seed=123,
    )

    assert subset is not None
    # subset is a list of 3 dicts: [non_dominant, dominant, crops]
    crops_subset = subset[2]
    assert crops_subset["num_repeats"] == 4

    aug_dir = Path(crops_subset["image_dir"])
    assert aug_dir.exists()

    generated_images = sorted(aug_dir.glob("*.png")) + sorted(aug_dir.glob("*.jpg"))
    assert generated_images, "Expected at least one generated augmented image"

    # Crop targets are adjacent resolutions (±16px) or minority buckets
    # that escape the dominant bucket — verify they're valid bucket sizes
    for image_path in generated_images:
        assert image_path.with_suffix(".txt").exists()
        image = cv2.imread(str(image_path))
        assert image is not None
        h, w = image.shape[:2]
        # Must be valid bucket resolution (≥256, multiple of 16)
        assert w >= 256 and h >= 256
        assert w % 16 == 0 and h % 16 == 0
        # Must NOT be the dominant bucket
        assert (w, h) != (848, 1184)


def test_maybe_build_bucket_rebalance_uses_resolution_param(tmp_path: Path):
    """The resolution parameter should be passed through to bucket collection."""
    dataset_dir = tmp_path / "img"
    dataset_dir.mkdir(parents=True)

    # Create images at a resolution that would be handled differently
    # by different max_area thresholds
    for i in range(8):
        path = dataset_dir / f"dom_{i:02d}.png"
        _write_image(path, 848, 1184)
        _add_caption(path)

    for i, (w, h) in enumerate([(1024, 1024), (784, 1312)]):
        path = dataset_dir / f"min_{i:02d}.png"
        _write_image(path, w, h)
        _add_caption(path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Should work with explicit resolution parameter
    subset = maybe_build_bucket_rebalance_subset(
        training_images=str(dataset_dir),
        output_dir=output_dir,
        num_repeats=4,
        enabled=True,
        dominance_threshold=0.35,
        max_augmented_images=24,
        seed=123,
        resolution=1024,
    )

    assert subset is not None
    # subset is a list of 3 dicts: [non_dominant, dominant, crops]
    assert Path(subset[0]["image_dir"]).exists()
