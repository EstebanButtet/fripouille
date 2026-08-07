"""Tests for the personal assistant core."""

from __future__ import annotations

import unittest

from assistant_ia.core.assistant import (
    ACTION_UNAVAILABLE_MESSAGE,
    AssistantCore,
    AssistantCoreError,
)
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.model_client import ModelClientError
from assistant_ia.intelligence.response import ModelResponse


class FakeModelClient:
    """Return predefined model responses and record received messages."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        """Create a fake client with ordered predefined responses."""
        self._responses = responses.copy()
        self.received_messages: list[
            tuple[ConversationMessage, ...]
        ] = []

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Record the messages and return the next predefined response."""
        self.received_messages.append(messages)

        if not self._responses:
            raise AssertionError("No fake model response remains.")

        return self._responses.pop(0)


class FailingModelClient:
    """Simulate a predictable language model failure."""

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Raise the same error type as a real model client."""
        raise ModelClientError("Simulated model failure.")


class AssistantCoreTests(unittest.TestCase):
    """Validate assistant orchestration and structured intent handling."""

    def test_processes_conversation_response(self) -> None:
        """Normal conversation content should be returned unchanged."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Une pile produit un courant électrique.",
                    model="fake-model",
                    intent=Intent(name="conversation"),
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        result = assistant.process_message("Explique une pile.")

        self.assertEqual(
            result,
            "Une pile produit un courant électrique.",
        )
        self.assertEqual(assistant.last_intent, Intent(name="conversation"))
        self.assertEqual(
            [
                (message.role, message.content)
                for message in assistant.context.messages
            ],
            [
                ("user", "Explique une pile."),
                (
                    "assistant",
                    "Une pile produit un courant électrique.",
                ),
            ],
        )

    def test_blocks_unavailable_action_content(self) -> None:
        """Action intentions should never claim successful execution."""
        intent = Intent(
            name="save_memory",
            parameters={"date": "24 août"},
        )
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="La mémoire a été enregistrée.",
                    model="fake-model",
                    intent=intent,
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        result = assistant.process_message(
            "Souviens-toi que mon examen est le 24 août."
        )

        self.assertEqual(result, ACTION_UNAVAILABLE_MESSAGE)
        self.assertIs(assistant.last_intent, intent)
        self.assertEqual(
            assistant.context.messages[-1].content,
            ACTION_UNAVAILABLE_MESSAGE,
        )
        self.assertNotIn(
            "enregistrée",
            assistant.context.messages[-1].content,
        )

    def test_sends_complete_ordered_history(self) -> None:
        """Each request should include the complete ordered conversation."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Bonjour.",
                    model="fake-model",
                    intent=Intent(name="conversation"),
                ),
                ModelResponse(
                    content="Je vais bien.",
                    model="fake-model",
                    intent=Intent(name="conversation"),
                ),
            ]
        )
        assistant = AssistantCore(model_client=client)

        assistant.process_message("Bonjour.")
        assistant.process_message("Comment vas-tu ?")

        self.assertEqual(len(client.received_messages), 2)
        self.assertEqual(
            [
                (message.role, message.content)
                for message in client.received_messages[1]
            ],
            [
                ("user", "Bonjour."),
                ("assistant", "Bonjour."),
                ("user", "Comment vas-tu ?"),
            ],
        )

    def test_reset_clears_context_and_last_intent(self) -> None:
        """Reset should remove messages and temporary intent state."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Réponse simulée.",
                    model="fake-model",
                    intent=Intent(name="create_task"),
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        assistant.process_message("Crée une tâche.")
        assistant.reset_conversation()

        self.assertEqual(assistant.context.message_count, 0)
        self.assertIsNone(assistant.last_intent)

    def test_instances_keep_independent_state(self) -> None:
        """Separate assistant instances should not share conversation state."""
        first_client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Première réponse.",
                    model="fake-model",
                    intent=Intent(name="conversation"),
                )
            ]
        )
        second_client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Deuxième réponse.",
                    model="fake-model",
                    intent=Intent(name="conversation"),
                )
            ]
        )

        first_assistant = AssistantCore(model_client=first_client)
        second_assistant = AssistantCore(model_client=second_client)

        first_assistant.process_message("Premier message.")

        self.assertEqual(first_assistant.context.message_count, 2)
        self.assertEqual(second_assistant.context.message_count, 0)
        self.assertIsNone(second_assistant.last_intent)

    def test_converts_model_client_error(self) -> None:
        """Model client failures should become assistant core errors."""
        assistant = AssistantCore(model_client=FailingModelClient())

        with self.assertRaisesRegex(
            AssistantCoreError,
            "local language model could not produce a response",
        ):
            assistant.process_message("Bonjour.")


if __name__ == "__main__":
    unittest.main()
