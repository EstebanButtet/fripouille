"""Tests for the interactive terminal interface."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from assistant_ia.application import (
    ApplicationInitializationError,
)
from assistant_ia.core.assistant import AssistantCoreError
from assistant_ia.interfaces.terminal import (
    DATABASE_ERROR_MESSAGE,
    MODEL_ERROR_MESSAGE,
    run_terminal,
)


class FakeAssistant:
    """Provide deterministic terminal assistant responses."""

    def __init__(
        self,
        responses: list[str] | None = None,
    ) -> None:
        """Store ordered responses and observable state."""
        self._responses = (
            responses.copy()
            if responses is not None
            else []
        )
        self.received_messages: list[str] = []
        self.reset_count = 0

    def process_message(
        self,
        user_message: str,
    ) -> str:
        """Record a message and return the next response."""
        self.received_messages.append(user_message)

        if not self._responses:
            raise AssertionError(
                "No fake assistant response remains."
            )

        return self._responses.pop(0)

    def reset_conversation(self) -> None:
        """Record one conversation reset."""
        self.reset_count += 1


class FailingAssistant:
    """Simulate a core processing failure."""

    def process_message(
        self,
        user_message: str,
    ) -> str:
        """Raise the same error type as the real assistant core."""
        raise AssistantCoreError(
            "Simulated core failure."
        )

    def reset_conversation(self) -> None:
        """Provide the terminal reset interface."""


class TerminalTests(unittest.TestCase):
    """Validate terminal assembly and command handling."""

    def test_uses_default_application_assembly(self) -> None:
        """Normal messages should use the assembled assistant."""
        assistant = FakeAssistant(
            responses=[
                "Réponse persistante.",
            ]
        )
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_assistant",
                return_value=assistant,
            ) as build_assistant,
            patch(
                "builtins.input",
                side_effect=[
                    "Bonjour.",
                    "/quit",
                ],
            ),
            redirect_stdout(output),
        ):
            run_terminal()

        build_assistant.assert_called_once_with()
        self.assertEqual(
            assistant.received_messages,
            [
                "Bonjour.",
            ],
        )
        self.assertIn(
            "Assistant > Réponse persistante.",
            output.getvalue(),
        )
        self.assertIn(
            "Assistant > À bientôt.",
            output.getvalue(),
        )

    def test_reset_command_uses_assistant_core(self) -> None:
        """The reset command should clear the current conversation."""
        assistant = FakeAssistant()
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_assistant",
                return_value=assistant,
            ),
            patch(
                "builtins.input",
                side_effect=[
                    "/reset",
                    "/quit",
                ],
            ),
            redirect_stdout(output),
        ):
            run_terminal()

        self.assertEqual(
            assistant.reset_count,
            1,
        )
        self.assertIn(
            "La conversation a été réinitialisée.",
            output.getvalue(),
        )

    def test_displays_model_error_and_keeps_running(self) -> None:
        """Core failures should produce the existing safe message."""
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_assistant",
                return_value=FailingAssistant(),
            ),
            patch(
                "builtins.input",
                side_effect=[
                    "Bonjour.",
                    "/quit",
                ],
            ),
            redirect_stdout(output),
        ):
            run_terminal()

        self.assertIn(
            MODEL_ERROR_MESSAGE,
            output.getvalue(),
        )
        self.assertIn(
            "Assistant > À bientôt.",
            output.getvalue(),
        )

    def test_stops_when_database_initialization_fails(self) -> None:
        """Database failures should stop before reading user input."""
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_assistant",
                side_effect=ApplicationInitializationError(
                    "Simulated database failure."
                ),
            ),
            patch(
                "builtins.input",
            ) as user_input,
            redirect_stdout(output),
        ):
            run_terminal()

        user_input.assert_not_called()
        self.assertIn(
            DATABASE_ERROR_MESSAGE,
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
