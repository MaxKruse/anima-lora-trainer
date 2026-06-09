"""Tests for evaluation prompt generation from tags."""

import pytest
from scripts.prompt_generator import generate_prompt


class TestGeneratePrompt:
    """Test random prompt generation from a tag pool."""

    def test_produces_prompt_with_random_subset_of_tags(self):
        """Given 20 tags, produces a prompt with 5-10 tags."""
        tags = [f"tag{i}" for i in range(20)]

        prompt = generate_prompt(tags, seed=42)

        words = prompt.split()
        assert 5 <= len(words) <= 10
        # All words must come from the original tag pool
        for word in words:
            assert word in tags

    def test_same_seed_produces_same_prompt(self):
        """Deterministic: same seed always yields the same prompt."""
        tags = [f"tag{i}" for i in range(20)]

        prompt_a = generate_prompt(tags, seed=123)
        prompt_b = generate_prompt(tags, seed=123)

        assert prompt_a == prompt_b

    def test_different_seeds_produce_different_prompts(self):
        """Different seeds should yield different prompts."""
        tags = [f"tag{i}" for i in range(20)]

        prompt_a = generate_prompt(tags, seed=1)
        prompt_b = generate_prompt(tags, seed=999)

        assert prompt_a != prompt_b

    def test_returns_empty_string_for_no_tags(self):
        """An empty tag list should produce an empty prompt."""
        assert generate_prompt([], seed=42) == ""
