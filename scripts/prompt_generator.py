"""Generate evaluation and test prompts from a pool of tags."""

import random


def generate_test_prompt(tags: list[str], num_tags: int = 10, seed: int = 42) -> str:
    """Generate a test prompt from training data tags.

    Combines "masterpiece" with a random subset of tags from the training
    dataset's caption files. Uses a fixed seed for reproducibility across
    all permutations in a matrix run.

    Args:
        tags: Pool of unique tags extracted from training captions.
        num_tags: Number of random tags to include (default: 10).
        seed: Random seed for deterministic selection.

    Returns:
        Prompt string like "masterpiece tag1 tag2 ... tag10".
        Returns just "masterpiece" if no tags are available.
    """
    if not tags:
        return "masterpiece"

    rng = random.Random(seed)
    selected = rng.sample(tags, min(len(tags), num_tags))

    return "masterpiece, " + ", ".join(selected)


def generate_prompt(tags: list[str], seed: int = 0) -> str:
    """Combine a random subset of *tags* into a single prompt string.

    Selects 5-10 tags (or fewer if the pool is small) using a deterministic
    random seed.  Returns an empty string when no tags are available.
    """
    if not tags:
        return ""

    rng = random.Random(seed)

    num_tags = min(len(tags), rng.randint(5, 10))
    selected = rng.sample(tags, num_tags)

    return " ".join(selected)
