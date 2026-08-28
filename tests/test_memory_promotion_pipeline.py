"""Tests for explicit consent around persistent memory promotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.actions.action import Action
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.capabilities.context import (
    CapabilityContext,
    render_capability_context,
)
from assistant_ia.core.assistant import (
    MEMORY_PROMOTION_ERROR_MESSAGE,
    AssistantCore,
)
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.promotion import MemoryPromotionService
from assistant_ia.memory.repository import SQLiteDatabase


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)

    def __call__(self) -> datetime:
        return self.values.pop(0)


class FakeModelClient:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate_response(self, messages):
        self.calls += 1
        return self.responses.pop(0)


class SequenceAnalyzer:
    def __init__(self, *results: tuple[MemoryCandidate, ...]) -> None:
        self.results = list(results)
        self.messages: list[str] = []

    def analyze(self, user_message: str):
        self.messages.append(user_message)
        return self.results.pop(0) if self.results else ()


def _conversation_response(content: str = "Reponse normale.") -> ModelResponse:
    return ModelResponse(
        content=content,
        model="fake",
        intent=Intent(name="conversation", parameters={}),
    )


class MemoryPromotionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.first_time = datetime(2026, 8, 28, tzinfo=timezone.utc)
        self.second_time = self.first_time + timedelta(hours=1)
        self.repository = MemoryRepository(
            self.database,
            clock=SequenceClock(
                self.first_time,
                self.second_time,
                self.second_time + timedelta(hours=1),
            ),
        )
        self.service = MemoryPromotionService(self.repository)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _candidate(
        self,
        content: str,
        confidence: float = 0.9,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            content=content,
            source_text=content,
            confidence=confidence,
        )

    def _assistant(
        self,
        analyzer: SequenceAnalyzer,
        *responses: ModelResponse,
        action_registry: ActionRegistry | None = None,
    ) -> tuple[AssistantCore, FakeModelClient]:
        model_client = FakeModelClient(*responses)
        return (
            AssistantCore(
                model_client=model_client,
                action_registry=action_registry,
                memory_candidate_analyzer=analyzer,
                memory_promotion_service=self.service,
            ),
            model_client,
        )

    def test_create_requires_yes_and_preserves_candidate_metadata(self) -> None:
        content = "Mon logiciel prefere pour la CAO est SolidWorks."
        candidate = self._candidate(content, confidence=0.87)
        analyzer = SequenceAnalyzer((candidate,))
        assistant, model_client = self._assistant(
            analyzer,
            _conversation_response(),
        )
        identity_name = assistant.person_context.assistant_name

        proposal_response = assistant.process_message(content)

        self.assertIn("Veux-tu que je la garde", proposal_response)
        self.assertEqual(self.repository.list_memories(), ())
        self.assertEqual(
            assistant.pending_memory_promotion.operation,
            "create",
        )

        confirmation_response = assistant.process_message("Oui.")
        memories = self.repository.list_memories()

        self.assertIn("Souvenir enregistré", confirmation_response)
        self.assertEqual(model_client.calls, 1)
        self.assertEqual(analyzer.messages, [content])
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source, "conversation_analysis")
        self.assertEqual(memories[0].source_text, content)
        self.assertEqual(memories[0].confidence, 0.87)
        self.assertEqual(memories[0].created_at, memories[0].updated_at)
        self.assertIsNone(assistant.pending_memory_promotion)
        self.assertEqual(
            assistant.person_context.assistant_name,
            identity_name,
        )

    def test_refusal_writes_nothing_and_is_not_analyzed(self) -> None:
        content = "Je prefere boire du cafe sans sucre."
        analyzer = SequenceAnalyzer((self._candidate(content),))
        assistant, model_client = self._assistant(
            analyzer,
            _conversation_response(),
        )
        assistant.process_message(content)

        response = assistant.process_message("Non.")

        self.assertIn("ne conserverai pas", response)
        self.assertEqual(self.repository.list_memories(), ())
        self.assertEqual(analyzer.messages, [content])
        self.assertEqual(model_client.calls, 1)
        self.assertIsNone(assistant.pending_memory_promotion)

    def test_unrelated_message_cancels_pending_and_runs_normally(self) -> None:
        first = "Mon projet est Fripouille."
        analyzer = SequenceAnalyzer((self._candidate(first),), ())
        assistant, model_client = self._assistant(
            analyzer,
            _conversation_response("Premiere reponse."),
            _conversation_response("Deuxieme reponse."),
        )
        assistant.process_message(first)

        response = assistant.process_message("Parlons d'autre chose.")

        self.assertEqual(response, "Deuxieme reponse.")
        self.assertEqual(model_client.calls, 2)
        self.assertEqual(self.repository.list_memories(), ())
        self.assertIsNone(assistant.pending_memory_promotion)

    def test_exact_duplicate_creates_no_pending_and_changes_nothing(self) -> None:
        content = "Mon logiciel prefere est SolidWorks."
        original = self.repository.save_memory(content)
        analyzer = SequenceAnalyzer((self._candidate(content),))
        assistant, _ = self._assistant(
            analyzer,
            _conversation_response(),
        )

        response = assistant.process_message(content)

        self.assertIn("connais déjà", response)
        self.assertIsNone(assistant.pending_memory_promotion)
        self.assertEqual(self.repository.list_memories(), (original,))
        self.assertEqual(
            self.repository.list_memories()[0].updated_at,
            original.updated_at,
        )

    def test_explicit_correction_updates_only_after_confirmation(self) -> None:
        original = self.repository.save_memory(
            "Mon logiciel prefere pour la CAO est SolidWorks."
        )
        correction = (
            "En fait, je prefere maintenant Fusion 360 pour la CAO."
        )
        candidate = self._candidate(correction, confidence=0.92)
        analyzer = SequenceAnalyzer((candidate,))
        assistant, _ = self._assistant(
            analyzer,
            _conversation_response(),
        )

        proposal_response = assistant.process_message(correction)

        self.assertIn("correction du souvenir", proposal_response)
        self.assertEqual(self.repository.list_memories(), (original,))

        assistant.process_message("Oui")
        updated = self.repository.list_memories()[0]

        self.assertEqual(updated.id, original.id)
        self.assertEqual(updated.created_at, original.created_at)
        self.assertGreater(updated.updated_at, original.updated_at)
        self.assertEqual(updated.content, correction)
        self.assertEqual(updated.source, "conversation_analysis")
        self.assertEqual(updated.source_text, correction)
        self.assertEqual(updated.confidence, 0.92)

    def test_conflict_is_not_resolved_without_explicit_consent(self) -> None:
        original = self.repository.save_memory("Je prefere SolidWorks.")
        conflict = "Je prefere Fusion 360."
        analyzer = SequenceAnalyzer((self._candidate(conflict),))
        assistant, _ = self._assistant(
            analyzer,
            _conversation_response(),
        )

        response = assistant.process_message(conflict)

        self.assertIn("peut contredire", response)
        self.assertEqual(self.repository.list_memories(), (original,))

        assistant.process_message("Non")
        self.assertEqual(self.repository.list_memories(), (original,))

    def test_confirmation_never_executes_registered_pc_action(self) -> None:
        launches: list[object] = []
        registry = ActionRegistry(
            (
                Action(
                    name="launch_application",
                    handler=lambda parameters: (
                        launches.append(parameters) or "launched"
                    ),
                ),
            )
        )
        content = "Mon projet est Fripouille."
        analyzer = SequenceAnalyzer((self._candidate(content),))
        assistant, model_client = self._assistant(
            analyzer,
            _conversation_response(),
            action_registry=registry,
        )
        assistant.process_message(content)

        assistant.process_message("Oui")

        self.assertEqual(launches, [])
        self.assertEqual(model_client.calls, 1)

    def test_persistence_failure_returns_safe_message(self) -> None:
        content = "Mon projet est Fripouille."
        analyzer = SequenceAnalyzer((self._candidate(content),))
        assistant, _ = self._assistant(
            analyzer,
            _conversation_response(),
        )
        assistant.process_message(content)
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_memory_insert
                BEFORE INSERT ON memories
                BEGIN
                    SELECT RAISE(ABORT, 'rejected');
                END
                """
            )

        response = assistant.process_message("Oui")

        self.assertEqual(response, MEMORY_PROMOTION_ERROR_MESSAGE)
        self.assertEqual(self.repository.list_memories(), ())

    def test_two_candidates_still_create_only_one_pending_proposal(self) -> None:
        first = self._candidate("Mon projet est Fripouille.")
        second = self._candidate("Mon outil est SolidWorks.")
        analyzer = SequenceAnalyzer((first, second))
        assistant, _ = self._assistant(
            analyzer,
            _conversation_response(),
        )

        assistant.process_message("Deux informations durables.")

        self.assertEqual(assistant.last_memory_candidates, (first, second))
        self.assertEqual(
            assistant.pending_memory_promotion.candidate,
            first,
        )
        self.assertEqual(self.repository.list_memories(), ())

    def test_capabilities_do_not_claim_silent_memory_storage(self) -> None:
        rendered = render_capability_context(
            CapabilityContext(
                available_actions=("save_memory",),
                automatic_memory_retrieval=True,
            )
        )

        self.assertIn(
            "Automatic contextual memory retrieval is available.",
            rendered,
        )
        self.assertIn("when the user asks", rendered)
        self.assertNotIn("automatic memory storage", rendered.casefold())


if __name__ == "__main__":
    unittest.main()
