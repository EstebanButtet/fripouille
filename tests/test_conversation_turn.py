"""Tests for conversational turn preparation."""

from __future__ import annotations

import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.turn import (
    ConversationTurn,
    build_conversation_turn,
)


class ConversationTurnTests(unittest.TestCase):
    """Validate separation of history from the current user turn."""

    def test_splits_history_from_current_user_message(self) -> None:
        """Previous messages should remain context for the current turn."""
        messages = (
            ConversationMessage(
                role="user",
                content="Qui es-tu ?",
            ),
            ConversationMessage(
                role="assistant",
                content="Je suis Fripouille.",
            ),
            ConversationMessage(
                role="user",
                content="Et que penses-tu de mon projet ?",
            ),
        )

        turn = build_conversation_turn(messages)

        self.assertIsInstance(
            turn,
            ConversationTurn,
        )
        self.assertEqual(
            turn.history,
            messages[:-1],
        )
        self.assertIs(
            turn.current_user_message,
            messages[-1],
        )

    def test_single_user_message_has_empty_history(self) -> None:
        """A first user message should form a turn without history."""
        message = ConversationMessage(
            role="user",
            content="Bonjour.",
        )

        turn = build_conversation_turn(
            (
                message,
            )
        )

        self.assertEqual(
            turn.history,
            (),
        )
        self.assertIs(
            turn.current_user_message,
            message,
        )

    def test_rejects_turn_without_current_user_message(self) -> None:
        """A conversational turn must end with a user message."""
        messages = (
            ConversationMessage(
                role="user",
                content="Bonjour.",
            ),
            ConversationMessage(
                role="assistant",
                content="Salut.",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "latest conversation message must be from the user",
        ):
            build_conversation_turn(messages)


if __name__ == "__main__":
    unittest.main()
