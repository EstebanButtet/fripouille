"""Tests for conversational turn preparation."""

from __future__ import annotations

import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.core.context import ConversationContext
from assistant_ia.intelligence.turn import (
    MAX_PROJECTED_HISTORY_CHARACTERS,
    MAX_PROJECTED_HISTORY_MESSAGES,
    ConversationTurn,
    build_conversation_turn,
    project_conversation_history,
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

    def test_keeps_only_four_recent_complete_exchanges(self) -> None:
        """The eight-message limit should preserve four recent pairs."""
        history = tuple(
            message
            for index in range(5)
            for message in (
                ConversationMessage(
                    role="user",
                    content=f"Question {index}",
                ),
                ConversationMessage(
                    role="assistant",
                    content=f"Réponse {index}",
                ),
            )
        )
        current = ConversationMessage(
            role="user",
            content="Question courante intégrale.",
        )

        turn = build_conversation_turn(history + (current,))

        self.assertEqual(
            len(turn.history),
            MAX_PROJECTED_HISTORY_MESSAGES,
        )
        self.assertEqual(turn.history, history[2:])
        self.assertIs(turn.current_user_message, current)

    def test_character_budget_keeps_recent_complete_pairs(self) -> None:
        """The text budget should retain a chronological pair suffix."""
        history = tuple(
            message
            for index in range(4)
            for message in (
                ConversationMessage(
                    role="user",
                    content=f"u{index}" + "x" * 698,
                ),
                ConversationMessage(
                    role="assistant",
                    content=f"a{index}" + "y" * 98,
                ),
            )
        )

        projection = project_conversation_history(history)

        self.assertEqual(projection, history[2:])
        self.assertLessEqual(
            sum(len(message.content) for message in projection),
            MAX_PROJECTED_HISTORY_CHARACTERS,
        )

    def test_does_not_truncate_one_oversized_recent_exchange(self) -> None:
        """An indivisible pair over budget should be omitted whole."""
        history = (
            ConversationMessage(
                role="user",
                content="u" * 1600,
            ),
            ConversationMessage(
                role="assistant",
                content="a" * 1500,
            ),
        )
        current = ConversationMessage(
            role="user",
            content="c" * 4000,
        )

        turn = build_conversation_turn(history + (current,))

        self.assertEqual(turn.history, ())
        self.assertEqual(
            turn.current_user_message.content,
            "c" * 4000,
        )

    def test_projection_leaves_complete_context_unchanged(self) -> None:
        """Projection should never remove application history."""
        context = ConversationContext()

        for index in range(5):
            context.add_user_message(f"Question {index}")
            context.add_assistant_message(f"Réponse {index}")

        context.add_user_message("Question courante")
        complete_history = context.messages

        turn = build_conversation_turn(complete_history)

        self.assertEqual(context.messages, complete_history)
        self.assertEqual(context.message_count, 11)
        self.assertEqual(len(turn.history), 8)

    def test_preserves_reasonable_suffix_for_unusual_history(self) -> None:
        """Unexpected standalone roles should remain deterministic."""
        history = tuple(
            ConversationMessage(
                role="assistant",
                content=f"Message {index}",
            )
            for index in range(10)
        )

        projection = project_conversation_history(history)

        self.assertEqual(projection, history[-8:])


if __name__ == "__main__":
    unittest.main()
