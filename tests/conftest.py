"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def sample_row():
    """A single PokerBench-style row for unit tests."""
    return {
        "instruction": (
            "You are a specialist in playing 6-handed No Limit Texas Holdem. "
            "The following will be a game scenario and you need to make the optimal decision. "
            "In this hand, your position is BTN, and your holding is [Ace of Spade and King of Heart]. "
            "The pot is 10 chips. Now it is your turn to make a move."
        ),
        "output": "bet 8",
    }


class MockTokenizer:
    """Minimal tokenizer stub for tests — no model download required."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ARG002
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()
