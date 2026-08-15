"""Tests for application-level assistant runtime orchestration."""

from __future__ import annotations

import unittest

from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.runtime import AssistantRuntime


class FakeModelClient:
    """Return deterministic model responses."""

    def __init__(
        self,
        responses: list[ModelResponse],
    ) -> None:
        self._responses = responses.copy()

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        if not self._responses:
            raise AssertionError(
                "No fake model response remains."
            )

        return self._responses.pop(0)


class RecordingPresenter:
    """Record final responses presented by the runtime."""

    def __init__(self) -> None:
        self.responses: list[str] = []

    def present(
        self,
        response: str,
    ) -> None:
        self.responses.append(
            response
        )


class AssistantRuntimeTests(unittest.TestCase):
    """Validate orchestration above the conversational core."""

    def test_process_message_returns_core_response(self) -> None:
        assistant = AssistantCore(
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Salut.",
                        model="fake-model",
                        intent=Intent(
                            name="conversation",
                        ),
                    )
                ]
            )
        )

        runtime = AssistantRuntime(
            assistant
        )

        self.assertEqual(
            runtime.process_message("Bonjour."),
            "Salut.",
        )

    def test_presents_final_core_response(self) -> None:
        presenter = RecordingPresenter()
        assistant = AssistantCore(
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="R?ponse finale.",
                        model="fake-model",
                        intent=Intent(
                            name="conversation",
                        ),
                    )
                ]
            )
        )

        runtime = AssistantRuntime(
            assistant,
            presenter=presenter,
        )

        result = runtime.process_message(
            "Question."
        )

        self.assertEqual(
            result,
            "R?ponse finale.",
        )
        self.assertEqual(
            presenter.responses,
            [
                "R?ponse finale.",
            ],
        )

    def test_does_not_present_raw_model_content_for_action(
        self,
    ) -> None:
        presenter = RecordingPresenter()
        assistant = AssistantCore(
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Fake model action success.",
                        model="fake-model",
                        intent=Intent(
                            name="unknown",
                        ),
                    )
                ]
            )
        )

        runtime = AssistantRuntime(
            assistant,
            presenter=presenter,
        )

        result = runtime.process_message(
            "Commande inconnue."
        )

        self.assertEqual(
            presenter.responses,
            [
                result,
            ],
        )
        self.assertNotEqual(
            result,
            "Fake model action success.",
        )

    def test_reset_delegates_to_assistant_core(self) -> None:
        assistant = AssistantCore(
            model_client=FakeModelClient([])
        )
        runtime = AssistantRuntime(
            assistant
        )

        assistant.context.add_user_message(
            "Message temporaire."
        )

        runtime.reset_conversation()

        self.assertEqual(
            assistant.context.messages,
            (),
        )

    def test_exposes_owned_assistant_core(self) -> None:
        assistant = AssistantCore(
            model_client=FakeModelClient([])
        )

        runtime = AssistantRuntime(
            assistant
        )

        self.assertIs(
            runtime.assistant,
            assistant,
        )

    def test_rejects_non_assistant_core(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "requires an AssistantCore",
        ):
            AssistantRuntime(
                object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
