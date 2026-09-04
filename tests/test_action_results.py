"""Tests for verified structured action execution results."""

from __future__ import annotations

from collections.abc import Mapping
import unittest

from assistant_ia.actions.action import Action, ActionExecutionError
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.actions.result import ActionExecutionResult
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.errors import RepositoryError
from assistant_ia.runtime import AssistantRuntime, TurnDiagnostics


class FakeModelClient:
    def __init__(self, response: ModelResponse) -> None:
        self._response = response

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        return self._response


class RecordingReporter:
    def __init__(self) -> None:
        self.diagnostics: TurnDiagnostics | None = None

    def report(self, diagnostics: TurnDiagnostics) -> None:
        self.diagnostics = diagnostics


class ActionExecutionResultTests(unittest.TestCase):
    def test_plain_text_handler_is_wrapped_without_breaking_text_api(self) -> None:
        action = Action(name="list_tasks", handler=lambda parameters: "Aucune tâche.")
        intent = Intent(name="list_tasks")

        result = action.execute_result(intent)

        self.assertEqual(
            result,
            ActionExecutionResult(
                action_name="list_tasks",
                status="success",
                message="Aucune tâche.",
                attempted=True,
            ),
        )
        self.assertEqual(action.execute(intent), "Aucune tâche.")

    def test_registry_preserves_structured_cancellation_and_text_api(self) -> None:
        result = ActionExecutionResult(
            action_name="launch_application",
            status="cancelled",
            message="Lancement annulé : Bloc-notes.",
            attempted=False,
        )
        registry = ActionRegistry(
            [Action(name="launch_application", handler=lambda parameters: result)]
        )
        intent = Intent(
            name="launch_application",
            parameters={"application": "bloc-notes"},
        )

        self.assertIs(registry.execute_result(intent), result)
        self.assertEqual(registry.execute(intent), result.message)

    def test_action_rejects_structured_result_for_another_action(self) -> None:
        def handler(parameters: Mapping[str, str]) -> ActionExecutionResult:
            return ActionExecutionResult(
                action_name="list_tasks",
                status="success",
                message="Résultat incohérent.",
                attempted=True,
            )

        action = Action(name="create_task", handler=handler)

        with self.assertRaisesRegex(ActionExecutionError, "ne correspond pas"):
            action.execute_result(
                Intent(name="create_task", parameters={"title": "Tester"})
            )

    def test_result_invariants_distinguish_success_cancellation_and_errors(self) -> None:
        invalid_cases = (
            {"status": "success", "attempted": False, "error_kind": None},
            {"status": "cancelled", "attempted": True, "error_kind": None},
            {"status": "error", "attempted": False, "error_kind": None},
            {"status": "error", "attempted": True, "error_kind": "validation"},
            {"status": "error", "attempted": False, "error_kind": "execution"},
        )
        for fields in invalid_cases:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    ActionExecutionResult(
                        action_name="list_tasks",
                        message="Résultat.",
                        **fields,
                    )

    def test_core_keeps_application_success_over_model_claim(self) -> None:
        intent = Intent(name="list_tasks")
        assistant = AssistantCore(
            model_client=FakeModelClient(
                ModelResponse(
                    content="L'action a échoué.",
                    model="fake-model",
                    intent=intent,
                )
            ),
            action_registry=ActionRegistry(
                [Action(name="list_tasks", handler=lambda parameters: "Succès réel.")]
            ),
        )

        response = assistant.process_message("Liste les tâches.")

        self.assertEqual(response, "Succès réel.")
        assert assistant.last_action_result is not None
        self.assertEqual(assistant.last_action_result.status, "success")
        self.assertTrue(assistant.last_action_result.attempted)

    def test_core_structures_validation_and_execution_errors(self) -> None:
        def invalid(parameters: Mapping[str, str]) -> str:
            raise ValueError("Entrée invalide")

        def broken(parameters: Mapping[str, str]) -> str:
            raise RepositoryError("Base indisponible")

        for handler, error_kind, attempted in (
            (invalid, "validation", False),
            (broken, "execution", True),
        ):
            with self.subTest(error_kind=error_kind):
                intent = Intent(name="list_tasks")
                assistant = AssistantCore(
                    model_client=FakeModelClient(
                        ModelResponse(
                            content="Succès inventé par le modèle.",
                            model="fake-model",
                            intent=intent,
                        )
                    ),
                    action_registry=ActionRegistry(
                        [Action(name="list_tasks", handler=handler)]
                    ),
                )

                assistant.process_message("Liste les tâches.")

                result = assistant.last_action_result
                assert result is not None
                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_kind, error_kind)
                self.assertEqual(result.attempted, attempted)

    def test_runtime_diagnostics_include_structured_action_result(self) -> None:
        intent = Intent(name="list_tasks")
        assistant = AssistantCore(
            model_client=FakeModelClient(
                ModelResponse(
                    content="ignoré",
                    model="fake-model",
                    intent=intent,
                )
            ),
            action_registry=ActionRegistry(
                [Action(name="list_tasks", handler=lambda parameters: "Aucune tâche.")]
            ),
        )
        reporter = RecordingReporter()
        runtime = AssistantRuntime(assistant, diagnostic_reporter=reporter)

        runtime.process_message("Liste les tâches.")

        assert reporter.diagnostics is not None
        self.assertEqual(reporter.diagnostics.action_result, assistant.last_action_result)


if __name__ == "__main__":
    unittest.main()
