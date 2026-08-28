"""Tests for the non-persistent candidate boundary in AssistantCore."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.application import build_default_assistant
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.memory_candidates import (
    MemoryCandidateAnalysisError,
)
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.repository import SQLiteDatabase


class FakeModelClient:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response

    def generate_response(self, messages):
        return self.response


class RecordingAnalyzer:
    def __init__(self, result=(), error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.messages: list[str] = []

    def analyze(self, user_message: str):
        self.messages.append(user_message)
        if self.error is not None:
            raise self.error
        return self.result


def _response(intent_name: str = "conversation") -> ModelResponse:
    return ModelResponse(
        content="Reponse deja obtenue.",
        model="fake",
        intent=Intent(name=intent_name, parameters={}),
    )


class MemoryCandidatePipelineTests(unittest.TestCase):
    def test_analyzes_only_current_user_message_after_conversation(self) -> None:
        candidate = MemoryCandidate(
            content="Mon projet est Fripouille.",
            source_text="Mon projet est Fripouille.",
            confidence=0.9,
        )
        analyzer = RecordingAnalyzer((candidate,))
        assistant = AssistantCore(
            model_client=FakeModelClient(_response()),
            memory_candidate_analyzer=analyzer,
        )

        result = assistant.process_message("Mon projet est Fripouille.")

        self.assertEqual(result, "Reponse deja obtenue.")
        self.assertEqual(analyzer.messages, ["Mon projet est Fripouille."])
        self.assertEqual(assistant.last_memory_candidates, (candidate,))
        self.assertEqual(assistant.context.message_count, 2)
        self.assertEqual(
            assistant.context.messages[-1].content,
            "Reponse deja obtenue.",
        )

    def test_analysis_failure_does_not_break_completed_conversation(self) -> None:
        analyzer = RecordingAnalyzer(
            error=MemoryCandidateAnalysisError("offline")
        )
        assistant = AssistantCore(
            model_client=FakeModelClient(_response()),
            memory_candidate_analyzer=analyzer,
        )

        result = assistant.process_message("Mon projet est Fripouille.")

        self.assertEqual(result, "Reponse deja obtenue.")
        self.assertEqual(assistant.last_memory_candidates, ())
        self.assertEqual(
            tuple(message.role for message in assistant.context.messages),
            ("user", "assistant"),
        )

    def test_action_intent_never_reaches_candidate_analyzer(self) -> None:
        analyzer = RecordingAnalyzer()
        assistant = AssistantCore(
            model_client=FakeModelClient(_response("save_memory")),
            action_registry=ActionRegistry(),
            memory_candidate_analyzer=analyzer,
        )

        assistant.process_message("Souviens-toi de ceci.")

        self.assertEqual(analyzer.messages, [])
        self.assertEqual(assistant.last_memory_candidates, ())

    def test_reset_clears_candidates_without_changing_identity(self) -> None:
        candidate = MemoryCandidate(
            content="Mon projet est Fripouille.",
            source_text="Mon projet est Fripouille.",
            confidence=0.9,
        )
        analyzer = RecordingAnalyzer((candidate,))
        assistant = AssistantCore(
            model_client=FakeModelClient(_response()),
            memory_candidate_analyzer=analyzer,
        )
        identity_name = assistant.person_context.assistant_name
        assistant.process_message("Mon projet est Fripouille.")

        assistant.reset_conversation()

        self.assertEqual(assistant.last_memory_candidates, ())
        self.assertEqual(assistant.context.message_count, 0)
        self.assertEqual(
            assistant.person_context.assistant_name,
            identity_name,
        )

    def test_default_application_detects_without_writing_sqlite(self) -> None:
        message = "Mon projet est Fripouille."
        candidate = MemoryCandidate(
            content=message,
            source_text=message,
            confidence=0.9,
        )
        analyzer = RecordingAnalyzer((candidate,))
        model_client = FakeModelClient(_response())

        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(Path(directory) / "assistant.db")
            with (
                patch(
                    "assistant_ia.application.OllamaModelClient",
                    return_value=model_client,
                ),
                patch(
                    "assistant_ia.application.OllamaMemoryCandidateAnalyzer",
                    return_value=analyzer,
                ) as analyzer_class,
            ):
                assistant = build_default_assistant(database=database)

            assistant.process_message(message)

            with database.connect() as connection:
                memory_count = connection.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()[0]

        analyzer_class.assert_called_once_with()
        self.assertEqual(assistant.last_memory_candidates, (candidate,))
        self.assertEqual(
            assistant.pending_memory_promotion.candidate,
            candidate,
        )
        self.assertEqual(memory_count, 0)


if __name__ == "__main__":
    unittest.main()
