"""Tests for persistent, person-scoped profile facts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.memory.repository import DatabaseError, SQLiteDatabase
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.profile_models import ProfileFact, ProfileFactCandidate


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)

    def __call__(self) -> datetime:
        return self.values.pop(0)


class ProfileFactModelTests(unittest.TestCase):
    def test_fact_is_distinct_from_candidate_and_normalizes_utc(self) -> None:
        instant = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
        fact = ProfileFact(
            id=4,
            person_id=2,
            category="preference",
            content="  Je préfère le thé.  ",
            source="conversation_analysis",
            source_text="Je préfère le thé.",
            confidence=0.91,
            created_at=instant,
            updated_at=instant,
        )
        candidate = ProfileFactCandidate(
            person_id=2,
            category="preference",
            content="Je préfère le thé.",
            source_text="Je préfère le thé.",
            confidence=0.91,
        )

        self.assertEqual(fact.content, "Je préfère le thé.")
        self.assertNotIsInstance(candidate, ProfileFact)
        self.assertFalse(hasattr(candidate, "id"))

    def test_candidate_requires_explicit_positive_person_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            ProfileFactCandidate(
                person_id=0,
                category="interest",
                content="J'aime la CAO.",
                source_text="J'aime la CAO.",
                confidence=0.9,
            )

    def test_category_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown profile fact category"):
            ProfileFactCandidate(
                person_id=1,
                category="secret",  # type: ignore[arg-type]
                content="Texte.",
                source_text="Texte.",
                confidence=0.9,
            )


class ProfileFactRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.person_repository = PersonRepository(self.database)
        self.alice = self.person_repository.create_person("Alice")
        self.bob = self.person_repository.create_person("Bob")
        first = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
        self.repository = ProfileFactRepository(
            self.database,
            clock=SequenceClock(
                first,
                first + timedelta(hours=1),
                first + timedelta(hours=2),
            ),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_create_get_and_list_profile_fact(self) -> None:
        fact = self.repository.create_profile_fact(
            self.alice.id,
            "interest",
            "J'aime la robotique.",
        )

        self.assertEqual(self.repository.get_profile_fact(fact.id), fact)
        self.assertEqual(
            self.repository.list_profile_facts(self.alice.id),
            (fact,),
        )
        self.assertEqual(fact.person_id, self.alice.id)
        self.assertEqual(fact.source, "explicit_user")
        self.assertIsNone(fact.source_text)

    def test_list_is_isolated_between_people(self) -> None:
        alice_fact = self.repository.create_profile_fact(
            self.alice.id, "preference", "Je préfère le thé."
        )
        bob_fact = self.repository.create_profile_fact(
            self.bob.id, "preference", "Je préfère le thé."
        )

        self.assertEqual(
            self.repository.list_profile_facts(self.alice.id),
            (alice_fact,),
        )
        self.assertEqual(
            self.repository.list_profile_facts(self.bob.id),
            (bob_fact,),
        )
        self.assertNotEqual(alice_fact.id, bob_fact.id)

    def test_foreign_key_rejects_unknown_person(self) -> None:
        with self.assertRaisesRegex(DatabaseError, "SQLite database operation failed"):
            self.repository.create_profile_fact(
                999, "personal_fact", "Je vis à Lausanne."
            )

    def test_confirmed_candidate_preserves_provenance(self) -> None:
        candidate = ProfileFactCandidate(
            person_id=self.alice.id,
            category="communication_preference",
            content="Je préfère les réponses courtes.",
            source_text="Je préfère les réponses courtes.",
            confidence=0.88,
        )

        fact = self.repository.save_candidate(candidate)

        self.assertEqual(fact.source, "conversation_analysis")
        self.assertEqual(fact.source_text, candidate.source_text)
        self.assertEqual(fact.confidence, 0.88)

    def test_update_cannot_change_person(self) -> None:
        fact = self.repository.create_profile_fact(
            self.alice.id, "preference", "Je préfère le thé."
        )
        candidate = ProfileFactCandidate(
            person_id=self.bob.id,
            category="preference",
            content="Je préfère le café.",
            source_text="Je préfère le café.",
            confidence=0.9,
        )

        with self.assertRaisesRegex(ValueError, "cannot change its person"):
            self.repository.update_profile_fact(fact.id, candidate)

    def test_update_then_delete_preserves_inspectable_lifecycle(self) -> None:
        original = self.repository.create_profile_fact(
            self.alice.id, "habit", "Je marche le matin."
        )
        candidate = ProfileFactCandidate(
            person_id=self.alice.id,
            category="habit",
            content="Je marche le soir.",
            source_text="En fait, je marche le soir.",
            confidence=0.93,
        )

        updated = self.repository.update_profile_fact(original.id, candidate)
        deleted = self.repository.delete_profile_fact(updated.id)

        self.assertEqual(updated.id, original.id)
        self.assertEqual(updated.created_at, original.created_at)
        self.assertGreater(updated.updated_at, original.updated_at)
        self.assertEqual(deleted, updated)
        self.assertIsNone(self.repository.get_profile_fact(original.id))


if __name__ == "__main__":
    unittest.main()
