"""Generate evaluation prompts from a pool of tags."""

import random


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
