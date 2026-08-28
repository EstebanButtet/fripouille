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
from assistant_ia.security.confirmation import ConfirmationRequest
from assistant_ia.interfaces.terminal import (
    DATABASE_ERROR_MESSAGE,
    MODEL_ERROR_MESSAGE,
    request_terminal_confirmation,
    run_terminal,
)


class FakeRuntime:
    """Provide deterministic terminal runtime responses."""

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


class FailingRuntime:
    """Simulate a core processing failure through the runtime boundary."""

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

    def test_terminal_confirmation_accepts_french_yes(
        self,
    ) -> None:
        """Only explicit French affirmative answers should confirm."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        for answer in (
            "o",
            "oui",
            " OUI ",
        ):
            with (
                self.subTest(answer=answer),
                patch(
                    "builtins.input",
                    return_value=answer,
                ) as user_input,
            ):
                result = request_terminal_confirmation(
                    request
                )

            self.assertTrue(result)
            user_input.assert_called_once_with(
                "Confirmer : lancer Bloc-notes ? [o/N] "
            )

    def test_terminal_confirmation_rejects_other_answers(
        self,
    ) -> None:
        """Nonaffirmative terminal answers should fail closed."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        for answer in (
            "",
            "n",
            "non",
            "yes",
            "y",
            "ok",
        ):
            with (
                self.subTest(answer=answer),
                patch(
                    "builtins.input",
                    return_value=answer,
                ),
            ):
                result = request_terminal_confirmation(
                    request
                )

            self.assertFalse(result)

    def test_uses_default_application_runtime(self) -> None:
        """Normal messages should use the assembled runtime."""
        runtime = FakeRuntime(
            responses=[
                "Réponse persistante.",
            ]
        )
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_runtime",
                return_value=runtime,
            ) as build_runtime,
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

        build_runtime.assert_called_once_with(
            confirmation_handler=request_terminal_confirmation,
        )
        self.assertEqual(
            runtime.received_messages,
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

    def test_reset_command_uses_runtime(self) -> None:
        """The reset command should use the runtime boundary."""
        runtime = FakeRuntime()
        output = StringIO()

        with (
            patch(
                "assistant_ia.interfaces.terminal."
                "build_default_runtime",
                return_value=runtime,
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
            runtime.reset_count,
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
                "build_default_runtime",
                return_value=FailingRuntime(),
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
                "build_default_runtime",
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
