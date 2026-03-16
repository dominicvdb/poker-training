"""Reward function for GRPO training: parse and score predicted poker actions."""

import re

VALID_ACTIONS = ("check", "fold", "call", "bet", "raise")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PASSIVE = {"check", "call"}
_AGGRESSIVE = {"bet", "raise"}


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def parse_action_type(text: str) -> str | None:
    """Extract the action type from model output.

    Returns one of: check, fold, call, bet, raise — or None if unrecognisable.
    'all-in' variants are mapped to 'raise'.
    """
    text = _strip_thinking(text).lower()
    for action in VALID_ACTIONS:
        if text.startswith(action):
            return action
    if "all-in" in text or "allin" in text or "all in" in text:
        return "raise"
    return None


def parse_bet_amount(text: str) -> float | None:
    """Extract the numeric bet/raise size from model output.

    Returns None for check, fold, call (no sizing), and for any output
    where no number can be found.
    """
    text = _strip_thinking(text).lower()
    action = parse_action_type(text)
    if action in ("check", "fold", "call"):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def poker_reward(predicted: str, correct: str) -> float:
    """Score a predicted action against the correct GTO action.

    Tiered scoring gives GRPO more gradient signal than binary right/wrong:

        1.0  — correct action, sizing within 10% (or no sizing needed)
        0.7  — correct action, sizing within 20%
        0.4  — correct action, sizing within 50%
        0.1  — correct action, sizing way off or missing
       -0.3  — wrong action but in same category (bet↔raise or check↔call)
       -1.0  — completely wrong action (e.g. fold when should raise)
    """
    pred_action = parse_action_type(predicted)
    true_action = parse_action_type(correct)

    if pred_action != true_action:
        both = {pred_action, true_action}
        if both <= _AGGRESSIVE:
            return -0.3   # bet vs raise — direction right, category wrong
        if both <= _PASSIVE:
            return -0.3   # check vs call — passive but not catastrophic
        return -1.0       # completely wrong (fold vs raise, etc.)

    true_amount = parse_bet_amount(correct)
    if true_amount is None:
        return 1.0        # check / fold / call — no sizing needed

    if true_amount == 0:
        return 1.0

    pred_amount = parse_bet_amount(predicted)
    if pred_amount is None:
        return 0.1        # correct action type but no sizing given

    ratio = pred_amount / true_amount
    if 0.9 <= ratio <= 1.1:
        return 1.0
    elif 0.8 <= ratio <= 1.2:
        return 0.7
    elif 0.5 <= ratio <= 1.5:
        return 0.4
    else:
        return 0.1
