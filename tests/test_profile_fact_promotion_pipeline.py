"""Tests for explicit confirmation and person-scoped profile promotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.core.assistant import PROFILE_PROMOTION_ERROR_MESSAGE, AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.profile_models import ProfileFactCandidate
from assistant_ia.people.profile_promotion import ProfileFactPromotionService


class FakeModelClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = None

    def generate_response(self, messages):
        self.calls += 1
        self.messages = messages
        return ModelResponse(
            content="Réponse normale.",
            model="fake",
            intent=Intent(name="conversation", parameters={}),
        )


class FakeActionModelClient(FakeModelClient):
    def generate_response(self, messages):
        self.calls += 1
        self.messages = messages
        return ModelResponse(
            content="Action proposée.",
            model="fake",
            intent=Intent(name="list_tasks", parameters={}),
        )


class SequenceProfileAnalyzer:
    def __init__(self, *results: tuple[ProfileFactCandidate, ...]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def analyze(self, message: str, *, person_id: int):
        self.calls.append((message, person_id))
        return self.results.pop(0) if self.results else ()


class RecordingMemoryAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, message: str):
        self.calls += 1
        return ()


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)

    def __call__(self) -> datetime:
        return self.values.pop(0)


class ProfileFactPromotionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.person_repository = PersonRepository(self.database)
        self.este = self.person_repository.get_person(DEFAULT_PERSON_ID)
        assert self.este is not None
        self.alice = self.person_repository.create_person("Alice")
        first = datetime(2026, 9, 2, 10, tzinfo=timezone.utc)
        self.repository = ProfileFactRepository(
            self.database,
            clock=SequenceClock(
                first,
                first + timedelta(hours=1),
                first + timedelta(hours=2),
                first + timedelta(hours=3),
            ),
        )
        self.service = ProfileFactPromotionService(self.repository)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _candidate(
        self,
        content: str,
        *,
        person_id: int | None = None,
        category: str = "communication_preference",
    ) -> ProfileFactCandidate:
        return ProfileFactCandidate(
            person_id=person_id if person_id is not None else self.este.id,
            category=category,  # type: ignore[arg-type]
            content=content,
            source_text=content,
            confidence=0.9,
        )

    def _assistant(
        self,
        analyzer: SequenceProfileAnalyzer,
        *,
        memory_analyzer: RecordingMemoryAnalyzer | None = None,
    ) -> tuple[AssistantCore, FakeModelClient]:
        model = FakeModelClient()
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=self.este,
        )
        return (
            AssistantCore(
                model_client=model,
                person_context=context,
                profile_fact_candidate_analyzer=analyzer,
                profile_fact_promotion_service=self.service,
                memory_candidate_analyzer=memory_analyzer,
            ),
            model,
        )

    def test_new_fact_writes_only_after_confirmation_for_active_person(self) -> None:
        message = "Je préfère les réponses courtes."
        candidate = self._candidate(message)
        analyzer = SequenceProfileAnalyzer((candidate,))
        assistant, model = self._assistant(analyzer)

        response = assistant.process_message(message)

        self.assertIn("profil", response)
        self.assertEqual(self.repository.list_profile_facts(self.este.id), ())
        self.assertEqual(analyzer.calls, [(message, self.este.id)])
        self.assertEqual(assistant.pending_profile_fact_promotion.candidate, candidate)

        confirmed = assistant.process_message("Oui")
        facts = self.repository.list_profile_facts(self.este.id)
        self.assertIn("Fait de profil enregistré", confirmed)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].person_id, self.este.id)
        self.assertEqual(model.calls, 1)

    def test_refusal_writes_nothing(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        assistant, _ = self._assistant(SequenceProfileAnalyzer((candidate,)))
        assistant.process_message(candidate.content)

        response = assistant.process_message("Non")

        self.assertIn("n'ajoute pas", response)
        self.assertEqual(self.repository.list_profile_facts(self.este.id), ())

    def test_exact_duplicate_never_writes_or_requests_confirmation(self) -> None:
        content = "Je préfère les réponses courtes."
        original = self.repository.create_profile_fact(
            self.este.id, "communication_preference", content
        )
        candidate = self._candidate("  JE PRÉFÈRE LES RÉPONSES COURTES  ")
        assistant, _ = self._assistant(SequenceProfileAnalyzer((candidate,)))

        response = assistant.process_message(candidate.content)

        self.assertIn("figure déjà", response)
        self.assertIsNone(assistant.pending_profile_fact_promotion)
        self.assertEqual(self.repository.list_profile_facts(self.este.id), (original,))

    def test_explicit_correction_updates_only_after_yes(self) -> None:
        original = self.repository.create_profile_fact(
            self.este.id,
            "communication_preference",
            "Je préfère les réponses courtes.",
        )
        correction = "En fait, je préfère les réponses détaillées."
        candidate = self._candidate(correction)
        assistant, _ = self._assistant(SequenceProfileAnalyzer((candidate,)))

        proposal = assistant.process_message(correction)
        self.assertIn("correction", proposal)
        self.assertEqual(self.repository.list_profile_facts(self.este.id), (original,))

        assistant.process_message("Oui")
        updated = self.repository.list_profile_facts(self.este.id)[0]
        self.assertEqual(updated.id, original.id)
        self.assertEqual(updated.created_at, original.created_at)
        self.assertGreater(updated.updated_at, original.updated_at)
        self.assertEqual(updated.content, correction)

    def test_simple_conflict_is_not_resolved_after_refusal(self) -> None:
        original = self.repository.create_profile_fact(
            self.este.id,
            "communication_preference",
            "Je préfère les réponses courtes.",
        )
        conflict = self._candidate("Je préfère les réponses détaillées.")
        assistant, _ = self._assistant(SequenceProfileAnalyzer((conflict,)))

        response = assistant.process_message(conflict.content)
        self.assertIn("peut contredire", response)
        assistant.process_message("Non")
        self.assertEqual(self.repository.list_profile_facts(self.este.id), (original,))

    def test_comparison_never_collides_with_another_person(self) -> None:
        same = "Je préfère les réponses courtes."
        alice_fact = self.repository.create_profile_fact(
            self.alice.id, "communication_preference", same
        )
        candidate = self._candidate(same, person_id=self.este.id)

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "create")
        self.assertEqual(
            self.repository.list_profile_facts(self.alice.id),
            (alice_fact,),
        )

    def test_profile_candidate_is_not_also_sent_to_memory(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        memory_analyzer = RecordingMemoryAnalyzer()
        assistant, _ = self._assistant(
            SequenceProfileAnalyzer((candidate,)),
            memory_analyzer=memory_analyzer,
        )

        assistant.process_message(candidate.content)

        self.assertEqual(memory_analyzer.calls, 0)
        self.assertEqual(MemoryRepository(self.database).list_memories(), ())

    def test_confirmation_cannot_apply_after_active_person_changes(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        assistant, model = self._assistant(SequenceProfileAnalyzer((candidate,)))
        assistant.process_message(candidate.content)
        assistant.person_context.set_active_persistent_person(self.alice)

        response = assistant.process_message("Oui")

        self.assertEqual(response, PROFILE_PROMOTION_ERROR_MESSAGE)
        self.assertEqual(self.repository.list_profile_facts(self.este.id), ())
        self.assertEqual(self.repository.list_profile_facts(self.alice.id), ())
        self.assertEqual(model.calls, 1)

    def test_stale_proposal_cannot_target_a_newly_created_fact(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        proposal = self.service.propose(candidate)
        self.repository.save_candidate(candidate)

        fact = self.service.apply_confirmed(proposal)

        self.assertEqual(len(self.repository.list_profile_facts(self.este.id)), 1)
        self.assertEqual(fact, self.repository.list_profile_facts(self.este.id)[0])

    def test_profile_facts_are_not_injected_into_conversation_context(self) -> None:
        self.repository.create_profile_fact(
            self.este.id, "personal_fact", "Mon code secret fictif est Zéphyr."
        )
        assistant, model = self._assistant(SequenceProfileAnalyzer(()))

        assistant.process_message("Bonjour")

        rendered_messages = " ".join(message.content for message in model.messages)
        self.assertNotIn("Zéphyr", rendered_messages)

    def test_unresolved_session_profile_never_reaches_analyzer(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        analyzer = SequenceProfileAnalyzer((candidate,))
        assistant = AssistantCore(
            model_client=FakeModelClient(),
            profile_fact_candidate_analyzer=analyzer,
            profile_fact_promotion_service=self.service,
        )

        assistant.process_message(candidate.content)

        self.assertIsNone(assistant.person_context.active_person_id)
        self.assertEqual(analyzer.calls, [])
        self.assertEqual(assistant.last_profile_fact_candidates, ())

    def test_action_intent_never_reaches_profile_analyzer(self) -> None:
        candidate = self._candidate("Je préfère les réponses courtes.")
        analyzer = SequenceProfileAnalyzer((candidate,))
        assistant = AssistantCore(
            model_client=FakeActionModelClient(),
            person_context=ActivePersonContext(
                assistant_name="Fripouille",
                default_person=self.este,
            ),
            profile_fact_candidate_analyzer=analyzer,
            profile_fact_promotion_service=self.service,
        )

        assistant.process_message("Liste mes tâches")

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(self.repository.list_profile_facts(self.este.id), ())


if __name__ == "__main__":
    unittest.main()
