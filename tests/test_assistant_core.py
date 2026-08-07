"""Tests for the personal assistant core."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

from assistant_ia.actions.action import Action
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.core.assistant import (
    ACTION_EXECUTION_ERROR_MESSAGE,
    ACTION_UNAVAILABLE_MESSAGE,
    AssistantCore,
    AssistantCoreError,
)
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.model_client import ModelClientError
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.errors import RepositoryError


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
            raise AssertionError(
                "No fake model response remains."
            )

        return self._responses.pop(0)


class FailingModelClient:
    """Simulate a predictable language model failure."""

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Raise the same error type as a real model client."""
        raise ModelClientError(
            "Simulated model failure."
        )


class AssistantCoreTests(unittest.TestCase):
    """Validate assistant orchestration and action execution."""

    def test_processes_conversation_response(self) -> None:
        """Normal conversation content should be returned unchanged."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content=(
                        "Une pile produit un courant électrique."
                    ),
                    model="fake-model",
                    intent=Intent(name="conversation"),
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        result = assistant.process_message(
            "Explique une pile."
        )

        self.assertEqual(
            result,
            "Une pile produit un courant électrique.",
        )
        self.assertEqual(
            assistant.last_intent,
            Intent(name="conversation"),
        )
        self.assertEqual(
            [
                (message.role, message.content)
                for message in assistant.context.messages
            ],
            [
                (
                    "user",
                    "Explique une pile.",
                ),
                (
                    "assistant",
                    "Une pile produit un courant électrique.",
                ),
            ],
        )

    def test_executes_registered_action(self) -> None:
        """An executable intent should use its registered handler."""
        intent = Intent(
            name="save_memory",
            parameters={
                "content": "Mon examen est le 24 août.",
            },
        )
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Le souvenir a été enregistré.",
                    model="fake-model",
                    intent=intent,
                )
            ]
        )
        registry = ActionRegistry(
            actions=(
                Action(
                    name="save_memory",
                    handler=lambda parameters: (
                        "Souvenir réellement enregistré : "
                        f"{parameters['content']}"
                    ),
                ),
            )
        )
        assistant = AssistantCore(
            model_client=client,
            action_registry=registry,
        )

        result = assistant.process_message(
            "Souviens-toi que mon examen est le 24 août."
        )

        self.assertEqual(
            result,
            "Souvenir réellement enregistré : "
            "Mon examen est le 24 août.",
        )
        self.assertNotEqual(
            result,
            "Le souvenir a été enregistré.",
        )
        self.assertEqual(
            assistant.context.messages[-1].content,
            result,
        )

    def test_blocks_unregistered_action_content(self) -> None:
        """An unregistered action should never claim success."""
        intent = Intent(
            name="save_memory",
            parameters={
                "content": "Mon examen est le 24 août.",
            },
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

        self.assertEqual(
            result,
            ACTION_UNAVAILABLE_MESSAGE,
        )
        self.assertIs(assistant.last_intent, intent)
        self.assertEqual(
            assistant.context.messages[-1].content,
            ACTION_UNAVAILABLE_MESSAGE,
        )
        self.assertNotIn(
            "enregistrée",
            assistant.context.messages[-1].content,
        )

    def test_keeps_launch_application_blocked(self) -> None:
        """Application launching should remain unavailable."""
        intent = Intent(
            name="launch_application",
            parameters={
                "application": "notepad",
            },
        )
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="L’application a été lancée.",
                    model="fake-model",
                    intent=intent,
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        result = assistant.process_message(
            "Lance le bloc-notes."
        )

        self.assertEqual(
            result,
            ACTION_UNAVAILABLE_MESSAGE,
        )

    def test_returns_action_validation_message(self) -> None:
        """Invalid action parameters should produce a precise response."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="La tâche a été créée.",
                    model="fake-model",
                    intent=Intent(
                        name="create_task",
                        parameters={},
                    ),
                )
            ]
        )
        registry = ActionRegistry(
            actions=(
                Action(
                    name="create_task",
                    handler=lambda parameters: "Tâche créée.",
                ),
            )
        )
        assistant = AssistantCore(
            model_client=client,
            action_registry=registry,
        )

        result = assistant.process_message(
            "Crée une tâche."
        )

        self.assertEqual(
            result,
            "Paramètre requis manquant : title.",
        )
        self.assertEqual(
            assistant.context.messages[-1].content,
            result,
        )

    def test_converts_action_execution_error(self) -> None:
        """Persistence failures should produce a safe visible message."""
        def failing_handler(
            parameters: Mapping[str, str],
        ) -> str:
            raise RepositoryError(
                "Internal persistence details."
            )

        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Le souvenir a été enregistré.",
                    model="fake-model",
                    intent=Intent(
                        name="save_memory",
                        parameters={
                            "content": "Souvenir.",
                        },
                    ),
                )
            ]
        )
        registry = ActionRegistry(
            actions=(
                Action(
                    name="save_memory",
                    handler=failing_handler,
                ),
            )
        )
        assistant = AssistantCore(
            model_client=client,
            action_registry=registry,
        )

        result = assistant.process_message(
            "Mémorise ce souvenir."
        )

        self.assertEqual(
            result,
            ACTION_EXECUTION_ERROR_MESSAGE,
        )
        self.assertNotIn(
            "Internal persistence details",
            result,
        )
        self.assertEqual(
            assistant.context.messages[-1].content,
            ACTION_EXECUTION_ERROR_MESSAGE,
        )

    def test_recovers_explicit_missing_journal_content(
        self,
    ) -> None:
        """Explicit text after a colon should repair model omission."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Entrée enregistrée.",
                    model="fake-model",
                    intent=Intent(
                        name="write_journal",
                        parameters={
                            "entry_date": "2026-08-07",
                        },
                    ),
                )
            ]
        )
        registry = ActionRegistry(
            actions=(
                Action(
                    name="write_journal",
                    handler=lambda parameters: (
                        "Journal reçu : "
                        f"{parameters['content']} "
                        f"({parameters['entry_date']})"
                    ),
                ),
            )
        )
        assistant = AssistantCore(
            model_client=client,
            action_registry=registry,
        )

        result = assistant.process_message(
            "Écris dans mon journal pour la date "
            "2026-08-07 : TEST E8 journal local."
        )

        self.assertEqual(
            result,
            "Journal reçu : TEST E8 journal local. "
            "(2026-08-07)",
        )
        self.assertEqual(
            assistant.last_intent,
            Intent(
                name="write_journal",
                parameters={
                    "content": "TEST E8 journal local.",
                    "entry_date": "2026-08-07",
                },
            ),
        )

    def test_does_not_invent_missing_journal_content(
        self,
    ) -> None:
        """Missing journal content should remain invalid without a separator."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Entrée enregistrée.",
                    model="fake-model",
                    intent=Intent(
                        name="write_journal",
                        parameters={
                            "entry_date": "2026-08-07",
                        },
                    ),
                )
            ]
        )
        registry = ActionRegistry(
            actions=(
                Action(
                    name="write_journal",
                    handler=lambda parameters: (
                        "Entrée enregistrée."
                    ),
                ),
            )
        )
        assistant = AssistantCore(
            model_client=client,
            action_registry=registry,
        )

        result = assistant.process_message(
            "Écris quelque chose dans mon journal."
        )

        self.assertEqual(
            result,
            "Paramètre requis manquant : content.",
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

        self.assertEqual(
            len(client.received_messages),
            2,
        )
        self.assertEqual(
            [
                (message.role, message.content)
                for message in client.received_messages[1]
            ],
            [
                (
                    "user",
                    "Bonjour.",
                ),
                (
                    "assistant",
                    "Bonjour.",
                ),
                (
                    "user",
                    "Comment vas-tu ?",
                ),
            ],
        )

    def test_reset_clears_context_and_last_intent(self) -> None:
        """Reset should remove messages and temporary intent state."""
        client = FakeModelClient(
            responses=[
                ModelResponse(
                    content="Réponse simulée.",
                    model="fake-model",
                    intent=Intent(
                        name="create_task",
                        parameters={},
                    ),
                )
            ]
        )
        assistant = AssistantCore(model_client=client)

        assistant.process_message("Crée une tâche.")
        assistant.reset_conversation()

        self.assertEqual(
            assistant.context.message_count,
            0,
        )
        self.assertIsNone(assistant.last_intent)

    def test_instances_keep_independent_state(self) -> None:
        """Separate assistant instances should not share state."""
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

        first_assistant = AssistantCore(
            model_client=first_client
        )
        second_assistant = AssistantCore(
            model_client=second_client
        )

        first_assistant.process_message(
            "Premier message."
        )

        self.assertEqual(
            first_assistant.context.message_count,
            2,
        )
        self.assertEqual(
            second_assistant.context.message_count,
            0,
        )
        self.assertIsNone(second_assistant.last_intent)

    def test_converts_model_client_error(self) -> None:
        """Model failures should become assistant core errors."""
        assistant = AssistantCore(
            model_client=FailingModelClient()
        )

        with self.assertRaisesRegex(
            AssistantCoreError,
            "local language model could not produce a response",
        ):
            assistant.process_message("Bonjour.")


if __name__ == "__main__":
    unittest.main()
