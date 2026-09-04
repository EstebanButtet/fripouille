"""Tests for persistent relationships and explicitly uncertain observations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.memory.repository import DatabaseError, SQLiteDatabase
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.observation_repository import (
    ObservationNotFoundError,
    ObservationRepository,
)
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.relationship_repository import (
    PersonRelationshipNotFoundError,
    PersonRelationshipRepository,
)
from assistant_ia.people.social_models import Observation, PersonRelationship


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        return self._values.pop(0)


class SocialRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        persons = PersonRepository(self.database)
        self.este = persons.get_person(DEFAULT_PERSON_ID)
        assert self.este is not None
        self.alice = persons.create_person("Alice")
        self.bob = persons.create_person("Bob")
        self.base_time = datetime(2026, 9, 4, 10, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_new_person_has_no_invented_relationship(self) -> None:
        repository = PersonRelationshipRepository(self.database)

        self.assertIsNone(repository.get_relationship(self.alice.id))
        self.assertIsNone(repository.get_relationship(self.bob.id))

    def test_relationship_is_unique_isolated_and_deterministically_updated(self) -> None:
        repository = PersonRelationshipRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
                self.base_time + timedelta(minutes=2),
                self.base_time + timedelta(minutes=3),
            ),
        )
        este = repository.create_relationship(
            self.este.id, "familiar", "playful"
        )
        alice = repository.create_relationship(self.alice.id)

        updated = repository.update_relationship(
            self.este.id,
            familiarity="close",
            interaction_style="direct",
        )

        self.assertIsInstance(updated, PersonRelationship)
        self.assertEqual(updated.created_at, este.created_at)
        self.assertGreater(updated.updated_at, este.updated_at)
        self.assertEqual(repository.get_relationship(self.este.id), updated)
        self.assertEqual(repository.get_relationship(self.alice.id), alice)
        with self.assertRaises(DatabaseError):
            repository.create_relationship(self.este.id)

    def test_relationship_never_changes_assistant_identity_or_security(self) -> None:
        identity = build_default_identity()
        repository = PersonRelationshipRepository(
            self.database, clock=SequenceClock(self.base_time)
        )
        repository.create_relationship(self.este.id, "close", "playful")

        self.assertEqual(identity, build_default_identity())
        self.assertEqual(
            identity.relationship_to_user,
            replace(identity).relationship_to_user,
        )
        self.assertFalse(hasattr(repository.get_relationship(self.este.id), "trust"))

    def test_relationship_delete_and_foreign_key(self) -> None:
        repository = PersonRelationshipRepository(
            self.database, clock=SequenceClock(self.base_time, self.base_time)
        )
        relationship = repository.create_relationship(self.alice.id)

        self.assertEqual(repository.delete_relationship(self.alice.id), relationship)
        self.assertIsNone(repository.get_relationship(self.alice.id))
        with self.assertRaises(PersonRelationshipNotFoundError):
            repository.delete_relationship(self.alice.id)
        with self.assertRaises(DatabaseError):
            repository.create_relationship(999)

    def test_observations_are_multiple_private_and_unconfirmed(self) -> None:
        repository = ObservationRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
                self.base_time + timedelta(minutes=2),
            ),
        )
        first = repository.create_observation(
            self.este.id,
            "communication",
            "Semble préférer les réponses directes.",
            source="conversation_analysis",
            source_text="Va droit au but.",
            confidence=0.8,
        )
        second = repository.create_observation(
            self.este.id, "behavior", "Semble apprécier l'humour."
        )
        alice = repository.create_observation(
            self.alice.id, "context", "Découvre le projet."
        )

        self.assertIsInstance(first, Observation)
        self.assertEqual(first.status, "unconfirmed")
        self.assertEqual(repository.get_observation(first.id), first)
        self.assertEqual(repository.list_observations(self.este.id), (second, first))
        self.assertEqual(repository.list_observations(self.alice.id), (alice,))

    def test_observation_is_not_profile_fact_and_can_be_deleted(self) -> None:
        observations = ObservationRepository(
            self.database, clock=SequenceClock(self.base_time)
        )
        profiles = ProfileFactRepository(self.database)
        observation = observations.create_observation(
            self.este.id, "preference", "Semble préférer Python."
        )

        self.assertEqual(profiles.list_profile_facts(self.este.id), ())
        self.assertEqual(observations.delete_observation(observation.id), observation)
        self.assertIsNone(observations.get_observation(observation.id))
        with self.assertRaises(ObservationNotFoundError):
            observations.delete_observation(observation.id)

    def test_observation_foreign_key_and_analysis_evidence_are_enforced(self) -> None:
        repository = ObservationRepository(
            self.database,
            clock=SequenceClock(self.base_time, self.base_time),
        )

        with self.assertRaises(DatabaseError):
            repository.create_observation(999, "context", "Signal isolé.")
        with self.assertRaisesRegex(ValueError, "exact source text"):
            repository.create_observation(
                self.este.id,
                "communication",
                "Signal analysé.",
                source="conversation_analysis",
            )


if __name__ == "__main__":
    unittest.main()
