"""Tests for application-owned subjects during memory promotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.core.assistant import MEMORY_PROMOTION_ERROR_MESSAGE, AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import MemoryCandidate, MemoryPersonLink
from assistant_ia.memory.promotion import MemoryPromotionService
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository


class FakeModelClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_response(self, messages):
        self.calls += 1
        return ModelResponse(
            content="Réponse normale.",
            model="fake",
            intent=Intent(name="conversation", parameters={}),
        )


class SequenceAnalyzer:
    def __init__(self, candidate: MemoryCandidate) -> None:
        self.candidate = candidate
        self.messages: list[str] = []

    def analyze(self, user_message: str):
        self.messages.append(user_message)
        return (self.candidate,)


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)

    def __call__(self) -> datetime:
        return self.values.pop(0)


class MemoryPersonPipelineTests(unittest.TestCase):
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
        base_time = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        self.repository = MemoryRepository(
            self.database,
            clock=SequenceClock(
                base_time,
                base_time + timedelta(minutes=1),
            ),
        )
        self.service = MemoryPromotionService(self.repository)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _candidate(self, content: str) -> MemoryCandidate:
        return MemoryCandidate(
            content=content,
            source_text=content,
            confidence=0.9,
        )

    def _assistant(
        self,
        candidate: MemoryCandidate,
    ) -> tuple[AssistantCore, FakeModelClient, SequenceAnalyzer]:
        model = FakeModelClient()
        analyzer = SequenceAnalyzer(candidate)
        assistant = AssistantCore(
            model_client=model,
            person_context=ActivePersonContext(
                assistant_name="Fripouille",
                default_person=self.este,
            ),
            memory_candidate_analyzer=analyzer,
            memory_promotion_service=self.service,
        )
        return assistant, model, analyzer

    def test_first_person_memory_links_active_person_after_confirmation(
        self,
    ) -> None:
        candidate = self._candidate("Je travaille sur Fripouille.")
        assistant, model, analyzer = self._assistant(candidate)

        assistant.process_message(candidate.content)

        proposal = assistant.pending_memory_promotion
        assert proposal is not None
        self.assertEqual(proposal.subject_person_id, self.este.id)
        self.assertEqual(self.repository.list_memories(), ())

        assistant.process_message("Oui")
        memory = self.repository.list_memories()[0]
        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (MemoryPersonLink(memory.id, self.este.id),),
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(analyzer.messages, [candidate.content])

    def test_general_fact_remains_unassigned_after_confirmation(self) -> None:
        candidate = self._candidate("Paris est la capitale de la France.")
        assistant, _, _ = self._assistant(candidate)

        assistant.process_message(candidate.content)
        proposal = assistant.pending_memory_promotion
        assert proposal is not None
        self.assertIsNone(proposal.subject_person_id)
        assistant.process_message("Oui")

        memory = self.repository.list_memories()[0]
        self.assertEqual(self.repository.list_person_links(memory.id), ())
        self.assertEqual(self.repository.list_unassigned_memories(), (memory,))

    def test_third_party_name_is_not_resolved_to_registered_person(self) -> None:
        candidate = self._candidate("Alice travaille chez Exemple SA.")
        assistant, _, _ = self._assistant(candidate)

        assistant.process_message(candidate.content)
        proposal = assistant.pending_memory_promotion
        assert proposal is not None
        self.assertIsNone(proposal.subject_person_id)
        assistant.process_message("Oui")

        memory = self.repository.list_memories()[0]
        self.assertEqual(self.repository.list_person_links(memory.id), ())
        self.assertEqual(
            self.repository.list_memories_for_person(self.alice.id),
            (),
        )

    def test_confirmation_cannot_follow_a_change_of_active_person(self) -> None:
        candidate = self._candidate("Je travaille sur Fripouille.")
        assistant, model, _ = self._assistant(candidate)
        assistant.process_message(candidate.content)
        assistant.person_context.set_active_persistent_person(self.alice)

        response = assistant.process_message("Oui")

        self.assertEqual(response, MEMORY_PROMOTION_ERROR_MESSAGE)
        self.assertEqual(self.repository.list_memories(), ())
        self.assertEqual(model.calls, 1)

    def test_memory_promotion_does_not_modify_profile_facts(self) -> None:
        profile_repository = ProfileFactRepository(self.database)
        profile_fact = profile_repository.create_profile_fact(
            self.este.id,
            "interest",
            "J'aime la robotique.",
        )
        candidate = self._candidate("Je travaille sur Fripouille.")
        assistant, _, _ = self._assistant(candidate)

        assistant.process_message(candidate.content)
        assistant.process_message("Oui")

        self.assertEqual(
            profile_repository.list_profile_facts(self.este.id),
            (profile_fact,),
        )


if __name__ == "__main__":
    unittest.main()
