"""Tests for validated executable assistant actions."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

from assistant_ia.actions.action import (
    Action,
    ActionExecutionError,
    ActionValidationError,
)
from assistant_ia.intelligence.intent import Intent
from assistant_ia.memory.errors import RepositoryError


class ActionTests(unittest.TestCase):
    """Validate the action execution boundary."""

    def test_executes_valid_intent_and_normalizes_result(self) -> None:
        """A valid intent should reach its handler exactly once."""
        received_parameters: Mapping[str, str] | None = None

        def handler(parameters: Mapping[str, str]) -> str:
            nonlocal received_parameters
            received_parameters = parameters
            return " Action exécutée. "

        action = Action(
            name="create_task",
            handler=handler,
        )
        intent = Intent(
            name="create_task",
            parameters={
                "title": "Réviser la biologie",
                "due_at": "demain",
            },
        )

        result = action.execute(intent)

        self.assertEqual(result, "Action exécutée.")
        self.assertIs(received_parameters, intent.parameters)

    def test_rejects_missing_required_parameter(self) -> None:
        """Required parameters should be checked before execution."""
        handler_called = False

        def handler(parameters: Mapping[str, str]) -> str:
            nonlocal handler_called
            handler_called = True
            return "Résultat."

        action = Action(
            name="create_task",
            handler=handler,
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "Paramètre requis manquant : title",
        ):
            action.execute(
                Intent(
                    name="create_task",
                    parameters={},
                )
            )

        self.assertFalse(handler_called)

    def test_rejects_unexpected_parameter(self) -> None:
        """Parameters outside the contract should never reach a handler."""
        handler_called = False

        def handler(parameters: Mapping[str, str]) -> str:
            nonlocal handler_called
            handler_called = True
            return "Résultat."

        action = Action(
            name="create_task",
            handler=handler,
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "Paramètre non autorisé : priority",
        ):
            action.execute(
                Intent(
                    name="create_task",
                    parameters={
                        "title": "Réviser",
                        "priority": "high",
                    },
                )
            )

        self.assertFalse(handler_called)

    def test_rejects_mismatched_intent(self) -> None:
        """An action should execute only its own intent."""
        action = Action(
            name="save_memory",
            handler=lambda parameters: "Résultat.",
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "ne correspond pas",
        ):
            action.execute(
                Intent(
                    name="find_memory",
                    parameters={
                        "query": "examen",
                    },
                )
            )

    def test_rejects_unsafe_action_name(self) -> None:
        """Blocked actions should not receive executable definitions."""
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported executable action",
        ):
            Action(
                name="launch_application",
                handler=lambda parameters: "Résultat.",
            )

    def test_converts_parameter_error_from_handler(self) -> None:
        """Handler parameter errors should become validation errors."""
        def handler(parameters: Mapping[str, str]) -> str:
            raise ValueError("Invalid identifier.")

        action = Action(
            name="complete_task",
            handler=handler,
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "paramètres de l’action sont invalides",
        ):
            action.execute(
                Intent(
                    name="complete_task",
                    parameters={
                        "task_id": "abc",
                    },
                )
            )

    def test_converts_repository_error_from_handler(self) -> None:
        """Persistence failures should become execution errors."""
        def handler(parameters: Mapping[str, str]) -> str:
            raise RepositoryError("Persistence failed.")

        action = Action(
            name="save_memory",
            handler=handler,
        )

        with self.assertRaisesRegex(
            ActionExecutionError,
            "n’a pas pu être exécutée",
        ):
            action.execute(
                Intent(
                    name="save_memory",
                    parameters={
                        "content": "Souvenir.",
                    },
                )
            )

    def test_rejects_non_string_handler_result(self) -> None:
        """Action handlers should always produce visible text."""
        action = Action(
            name="list_tasks",
            handler=lambda parameters: None,
        )

        with self.assertRaisesRegex(
            ActionExecutionError,
            "doit être un texte",
        ):
            action.execute(
                Intent(
                    name="list_tasks",
                    parameters={},
                )
            )

    def test_rejects_empty_handler_result(self) -> None:
        """Action handlers should not return empty visible content."""
        action = Action(
            name="list_tasks",
            handler=lambda parameters: "   ",
        )

        with self.assertRaisesRegex(
            ActionExecutionError,
            "ne peut pas être vide",
        ):
            action.execute(
                Intent(
                    name="list_tasks",
                    parameters={},
                )
            )


if __name__ == "__main__":
    unittest.main()
