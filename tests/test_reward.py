"""Tests for src/reward.py."""

import pytest
from src.reward import parse_action_type, parse_bet_amount, poker_reward


# ── parse_action_type ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("check", "check"),
    ("fold", "fold"),
    ("call", "call"),
    ("bet 10", "bet"),
    ("bet 10.5", "bet"),
    ("raise 25", "raise"),
    ("  Bet 8  ", "bet"),
    ("FOLD", "fold"),
    ("Call", "call"),
    ("raise 50 chips", "raise"),
    ("all-in", "raise"),
    ("all in", "raise"),
    ("allin", "raise"),
    ("go all-in", "raise"),
])
def test_parse_action_type_valid(text, expected):
    assert parse_action_type(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "bluff",
    "shove",
    "123",
    "???",
])
def test_parse_action_type_unknown_returns_none(text):
    assert parse_action_type(text) is None


# ── parse_bet_amount ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("bet 10", 10.0),
    ("raise 25", 25.0),
    ("bet 10.5", 10.5),
    ("raise 100 chips", 100.0),
    ("bet 0", 0.0),
])
def test_parse_bet_amount_with_sizing(text, expected):
    assert parse_bet_amount(text) == expected


@pytest.mark.parametrize("text", [
    "check",
    "fold",
    "call",
    "  check  ",
    "FOLD",
])
def test_parse_bet_amount_no_sizing_actions_return_none(text):
    assert parse_bet_amount(text) is None


def test_parse_bet_amount_bet_without_number_returns_none():
    assert parse_bet_amount("bet") is None


# ── poker_reward ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("predicted,correct", [
    ("check", "check"),
    ("fold", "fold"),
    ("call", "call"),
])
def test_reward_correct_no_sizing_actions(predicted, correct):
    assert poker_reward(predicted, correct) == 1.0


@pytest.mark.parametrize("predicted,correct", [
    ("fold", "check"),
    ("call", "fold"),
    ("bet 10", "check"),
    ("check", "bet 10"),
    ("raise 20", "call"),
    ("bet 10", "raise 10"),
])
def test_reward_wrong_action_type(predicted, correct):
    assert poker_reward(predicted, correct) == -1.0


@pytest.mark.parametrize("predicted,correct", [
    ("bet 10", "bet 10"),
    ("raise 20", "raise 20"),
    ("bet 10", "bet 9"),      # within 20%
    ("bet 10", "bet 11"),     # within 20%
    ("raise 10", "raise 12"), # within 20%
])
def test_reward_exact_or_near_sizing(predicted, correct):
    assert poker_reward(predicted, correct) == 1.0


@pytest.mark.parametrize("predicted,correct", [
    ("bet 6", "bet 10"),   # 60% — within 50%
    ("bet 14", "bet 10"),  # 140% — within 50%
    ("raise 8", "raise 15"),
])
def test_reward_reasonable_sizing(predicted, correct):
    assert poker_reward(predicted, correct) == 0.5


@pytest.mark.parametrize("predicted,correct", [
    ("bet 1", "bet 10"),   # 10% — too far off
    ("bet 20", "bet 10"),  # 200% — too far off
    ("raise 1", "raise 50"),
])
def test_reward_bad_sizing(predicted, correct):
    assert poker_reward(predicted, correct) == 0.0


def test_reward_correct_action_missing_amount():
    assert poker_reward("bet", "bet 10") == 0.0


def test_reward_whitespace_handling():
    assert poker_reward("  check  ", "check") == 1.0
    assert poker_reward("  bet 10  ", "bet 10") == 1.0


def test_reward_case_insensitive():
    assert poker_reward("CHECK", "check") == 1.0
    assert poker_reward("BET 10", "bet 10") == 1.0
