"""Tests for default assistant runtime assembly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_ia.application import build_default_runtime
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.runtime import AssistantRuntime


class FakeModelClient:
    """Return deterministic structured model responses."""

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
    """Record responses presented by the assembled runtime."""

    def __init__(self) -> None:
        self.responses: list[str] = []

    def present(
        self,
        response: str,
    ) -> None:
        self.responses.append(
            response
        )


class ApplicationRuntimeAssemblyTests(unittest.TestCase):
    """Validate runtime assembly without real hardware."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name)
            / "assistant.db"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_runtime_around_default_assistant(self) -> None:
        runtime = build_default_runtime(
            database=self.database,
            model_client=FakeModelClient([]),
        )

        self.assertIsInstance(
            runtime,
            AssistantRuntime,
        )
        self.assertEqual(
            runtime.assistant.action_registry.action_count,
            8,
        )

    def test_injects_presenter_into_runtime(self) -> None:
        presenter = RecordingPresenter()

        runtime = build_default_runtime(
            database=self.database,
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
            ),
            presenter=presenter,
        )

        result = runtime.process_message(
            "Bonjour."
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

    def test_runtime_preserves_final_core_authority(self) -> None:
        presenter = RecordingPresenter()

        runtime = build_default_runtime(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Fake model success.",
                        model="fake-model",
                        intent=Intent(
                            name="unknown",
                        ),
                    )
                ]
            ),
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
            "Fake model success.",
        )


if __name__ == "__main__":
    unittest.main()
