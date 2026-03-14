"""Card and Deck representations for the poker simulator.

Converts human-readable card strings (e.g. "As", "Kh", "2c") into the
integer format that the `treys` library expects, and provides a Deck
that can shuffle and deal without replacement.
"""

from __future__ import annotations

import random
from treys import Card
from treys import Deck as _TreysDeck


# Valid ranks and suits used for input validation
VALID_RANKS = set("23456789TJQKA")
VALID_SUITS = set("shdc")


def card_to_int(card_str: str) -> int:
    """Convert a human-readable card string to a treys integer.

    Parameters
    ----------
    card_str : str
        Two-character string like "As", "Kh", "Td", "2c".
        First character is rank (2-9, T, J, Q, K, A).
        Second character is suit (s, h, d, c).

    Returns
    -------
    int
        The treys integer representation.

    Raises
    ------
    ValueError
        If the string is not exactly two characters or contains
        an invalid rank or suit.
    """
    if not isinstance(card_str, str) or len(card_str) != 2:
        raise ValueError(
            f"Card string must be exactly 2 characters, got: {card_str!r}"
        )

    rank, suit = card_str[0], card_str[1]

    if rank not in VALID_RANKS:
        raise ValueError(
            f"Invalid rank '{rank}' in '{card_str}'. "
            f"Valid ranks: {sorted(VALID_RANKS)}"
        )
    if suit not in VALID_SUITS:
        raise ValueError(
            f"Invalid suit '{suit}' in '{card_str}'. "
            f"Valid suits: {sorted(VALID_SUITS)}"
        )

    return Card.new(card_str)


def parse_hand(hand_str: str) -> list[int]:
    """Parse a four-character hand string into two treys card integers.

    Parameters
    ----------
    hand_str : str
        Four-character string representing two hole cards,
        e.g. "AsKs", "Td9d", "2c2h".

    Returns
    -------
    list[int]
        A list of two treys integer cards.

    Raises
    ------
    ValueError
        If the string is not exactly four characters or contains
        invalid card representations.
    """
    if not isinstance(hand_str, str) or len(hand_str) != 4:
        raise ValueError(
            f"Hand string must be exactly 4 characters, got: {hand_str!r}"
        )

    card1 = card_to_int(hand_str[:2])
    card2 = card_to_int(hand_str[2:])

    if card1 == card2:
        raise ValueError(
            f"Hand contains duplicate card: '{hand_str[:2]}'"
        )

    return [card1, card2]


class Deck:
    """A standard 52-card deck using treys integer format.

    Supports shuffling and dealing cards without replacement.
    """

    _ALL_CARDS: list[int] = _TreysDeck.GetFullDeck()

    def __init__(self) -> None:
        """Create a new shuffled 52-card deck."""
        self._cards: list[int] = list(self._ALL_CARDS)
        self.shuffle()

    @property
    def size(self) -> int:
        """Number of cards remaining in the deck."""
        return len(self._cards)

    def shuffle(self) -> None:
        """Shuffle the remaining cards in place."""
        random.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[int]:
        """Deal *n* cards from the top of the deck.

        Parameters
        ----------
        n : int
            Number of cards to deal (default 1).

        Returns
        -------
        list[int]
            The dealt cards as treys integers.

        Raises
        ------
        ValueError
            If *n* is less than 1 or more cards are requested
            than remain in the deck.
        """
        if n < 1:
            raise ValueError(f"Must deal at least 1 card, got {n}")
        if n > self.size:
            raise ValueError(
                f"Cannot deal {n} cards — only {self.size} remain"
            )

        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def remove(self, cards: list[int]) -> None:
        """Remove specific cards from the deck.

        This is used by the simulator to remove hero's known hole
        cards before dealing the board and villain hands.

        Parameters
        ----------
        cards : list[int]
            Cards to remove (treys integers).

        Raises
        ------
        ValueError
            If any card is not present in the deck.
        """
        for card in cards:
            try:
                self._cards.remove(card)
            except ValueError:
                raise ValueError(
                    f"Card {Card.int_to_pretty_str(card)} not in deck"
                )

    def __len__(self) -> int:
        return self.size

    def __contains__(self, card: int) -> bool:
        return card in self._cards