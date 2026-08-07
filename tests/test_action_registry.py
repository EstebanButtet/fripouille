"""Tests for the immutable assistant action registry."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

from assistant_ia.actions.action import (
    Action,
    ActionValidationError,
)
from assistant_ia.actions.registry import (
    ActionAlreadyRegisteredError,
    ActionNotRegisteredError,
    ActionRegistry,
)
from assistant_ia.intelligence.intent import Intent


class ActionRegistryTests(unittest.TestCase):
    """Validate action registration and selection."""

    def test_creates_empty_registry(self) -> None:
        """A registry should support having no executable actions."""
        registry = ActionRegistry()

        self.assertEqual(registry.action_count, 0)
        self.assertEqual(registry.action_names, frozenset())

    def test_registers_valid_actions(self) -> None:
        """Valid actions should be exposed through immutable metadata."""
        registry = ActionRegistry(
            actions=(
                Action(
                    name="create_task",
                    handler=lambda parameters: "Tâche créée.",
                ),
                Action(
                    name="save_memory",
                    handler=lambda parameters: "Souvenir enregistré.",
                ),
            )
        )

        self.assertEqual(registry.action_count, 2)
        self.assertEqual(
            registry.action_names,
            frozenset(
                {
                    "create_task",
                    "save_memory",
                }
            ),
        )

    def test_has_action_normalizes_name(self) -> None:
        """Action lookup should normalize surrounding whitespace."""
        registry = ActionRegistry(
            actions=(
                Action(
                    name="save_memory",
                    handler=lambda parameters: "Souvenir enregistré.",
                ),
            )
        )

        self.assertTrue(registry.has_action(" save_memory "))
        self.assertFalse(registry.has_action("find_memory"))

    def test_executes_matching_action_only(self) -> None:
        """The registry should invoke only the matching action."""
        calls: list[str] = []

        def create_task_handler(
            parameters: Mapping[str, str],
        ) -> str:
            calls.append("create_task")
            return f"Tâche créée : {parameters['title']}."

        def save_memory_handler(
            parameters: Mapping[str, str],
        ) -> str:
            calls.append("save_memory")
            return "Souvenir enregistré."

        registry = ActionRegistry(
            actions=(
                Action(
                    name="create_task",
                    handler=create_task_handler,
                ),
                Action(
                    name="save_memory",
                    handler=save_memory_handler,
                ),
            )
        )

        result = registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Réviser la biologie",
                },
            )
        )

        self.assertEqual(
            result,
            "Tâche créée : Réviser la biologie.",
        )
        self.assertEqual(calls, ["create_task"])

    def test_rejects_duplicate_action_name(self) -> None:
        """One intent should never have multiple registered actions."""
        with self.assertRaisesRegex(
            ActionAlreadyRegisteredError,
            "déjà enregistrée",
        ):
            ActionRegistry(
                actions=(
                    Action(
                        name="save_memory",
                        handler=lambda parameters: "Premier résultat.",
                    ),
                    Action(
                        name="save_memory",
                        handler=lambda parameters: "Deuxième résultat.",
                    ),
                )
            )

    def test_rejects_non_action_entry(self) -> None:
        """Registry entries should be validated Action objects."""
        with self.assertRaisesRegex(
            TypeError,
            "must be Action instances",
        ):
            ActionRegistry(
                actions=(
                    "save_memory",
                )
            )

    def test_rejects_unregistered_executable_intent(self) -> None:
        """An executable intent requires an explicitly registered action."""
        registry = ActionRegistry()

        with self.assertRaisesRegex(
            ActionNotRegisteredError,
            "Aucune action exécutable",
        ):
            registry.execute(
                Intent(
                    name="save_memory",
                    parameters={
                        "content": "Information importante.",
                    },
                )
            )

    def test_keeps_launch_application_blocked(self) -> None:
        """Application launching should remain unavailable."""
        registry = ActionRegistry()

        with self.assertRaisesRegex(
            ActionNotRegisteredError,
            "launch_application",
        ):
            registry.execute(
                Intent(
                    name="launch_application",
                    parameters={
                        "application": "notepad",
                    },
                )
            )

    def test_keeps_conversation_outside_action_execution(self) -> None:
        """Normal conversation should not become an executable action."""
        registry = ActionRegistry()

        with self.assertRaisesRegex(
            ActionNotRegisteredError,
            "conversation",
        ):
            registry.execute(
                Intent(
                    name="conversation",
                    parameters={},
                )
            )

    def test_propagates_action_validation_error(self) -> None:
        """The registry should preserve action parameter validation."""
        registry = ActionRegistry(
            actions=(
                Action(
                    name="complete_task",
                    handler=lambda parameters: "Tâche terminée.",
                ),
            )
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "Paramètre requis manquant : task_id",
        ):
            registry.execute(
                Intent(
                    name="complete_task",
                    parameters={},
                )
            )


if __name__ == "__main__":
    unittest.main()
