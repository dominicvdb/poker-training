"""Tests for src/cards.py — Card helpers and Deck class."""

import pytest
from treys import Card

from src.cards import card_to_int, parse_hand, Deck


# ── card_to_int ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "card_str",
    ["As", "Kh", "Td", "9c", "2s", "Jh", "Qd", "5c"],
)
def test_valid_card_strings(card_str: str) -> None:
    """Valid card strings should return the same int as treys.Card.new()."""
    result = card_to_int(card_str)
    expected = Card.new(card_str)
    assert result == expected


@pytest.mark.parametrize(
    "bad_str",
    [
        "Xs",   # invalid rank
        "1h",   # invalid rank
        "Ax",   # invalid suit
        "A",    # too short
        "AsK",  # too long
        "",     # empty
        "0s",   # zero is not a rank
    ],
)
def test_invalid_card_strings_raise(bad_str: str) -> None:
    """Invalid card strings should raise ValueError."""
    with pytest.raises(ValueError):
        card_to_int(bad_str)


def test_card_to_int_non_string_raises() -> None:
    """Non-string input should raise ValueError."""
    with pytest.raises(ValueError):
        card_to_int(42)  # type: ignore[arg-type]


# ── parse_hand ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "hand_str",
    ["AsKs", "Td9d", "2c3h", "AhAd"],
)
def test_parse_hand_valid(hand_str: str) -> None:
    """Valid 4-char hand strings should return a list of two ints."""
    result = parse_hand(hand_str)
    assert len(result) == 2
    assert result[0] == Card.new(hand_str[:2])
    assert result[1] == Card.new(hand_str[2:])


def test_parse_hand_duplicate_card_raises() -> None:
    """A hand with the same card twice should raise ValueError."""
    with pytest.raises(ValueError, match="duplicate"):
        parse_hand("AsAs")


@pytest.mark.parametrize(
    "bad_hand",
    ["As", "AsKsQh", "", "XXXX"],
)
def test_parse_hand_wrong_length_raises(bad_hand: str) -> None:
    """Hand strings that aren't exactly 4 characters should raise."""
    with pytest.raises(ValueError):
        parse_hand(bad_hand)


# ── Deck basics ──────────────────────────────────────────────────────

def test_fresh_deck_has_52_cards(fresh_deck: Deck) -> None:
    """A new deck should contain exactly 52 cards."""
    assert fresh_deck.size == 52
    assert len(fresh_deck) == 52


def test_deck_no_duplicates(fresh_deck: Deck) -> None:
    """All 52 cards in a fresh deck should be unique."""
    all_cards = fresh_deck.deal(52)
    assert len(set(all_cards)) == 52


# ── Deck.deal ────────────────────────────────────────────────────────

def test_deal_removes_cards(fresh_deck: Deck) -> None:
    """Dealing N cards should reduce the deck size by N."""
    fresh_deck.deal(5)
    assert fresh_deck.size == 47


def test_deal_returns_correct_count(fresh_deck: Deck) -> None:
    """deal(n) should return exactly n cards."""
    cards = fresh_deck.deal(7)
    assert len(cards) == 7


def test_deal_too_many_raises(fresh_deck: Deck) -> None:
    """Dealing more cards than remain should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot deal"):
        fresh_deck.deal(53)


def test_deal_zero_raises(fresh_deck: Deck) -> None:
    """Dealing 0 or negative cards should raise ValueError."""
    with pytest.raises(ValueError, match="at least 1"):
        fresh_deck.deal(0)


def test_dealt_cards_not_in_deck(fresh_deck: Deck) -> None:
    """Cards that have been dealt should no longer be in the deck."""
    dealt = fresh_deck.deal(5)
    for card in dealt:
        assert card not in fresh_deck


# ── Deck.remove ──────────────────────────────────────────────────────

def test_remove_known_cards(fresh_deck: Deck) -> None:
    """Removing specific cards should shrink the deck accordingly."""
    hero = parse_hand("AsKs")
    fresh_deck.remove(hero)
    assert fresh_deck.size == 50
    for card in hero:
        assert card not in fresh_deck


def test_remove_missing_card_raises(fresh_deck: Deck) -> None:
    """Removing a card not in the deck should raise ValueError."""
    card = card_to_int("As")
    fresh_deck.remove([card])  # first removal succeeds
    with pytest.raises(ValueError):
        fresh_deck.remove([card])  # second should fail