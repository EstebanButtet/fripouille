"""Tests for bounded person-private social prompt context."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.prompt import build_conversation_prompt
from assistant_ia.memory.models import Memory
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.retrieval import RetrievedMemory
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.observation_repository import ObservationRepository
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.relationship_repository import (
    PersonRelationshipRepository,
)
from assistant_ia.people.social_context import (
    MAX_CONTEXTUAL_OBSERVATIONS,
    MAX_CONTEXTUAL_OBSERVATION_TOTAL_CHARACTERS,
    MAX_CONTEXTUAL_PROFILE_FACTS,
    MAX_CONTEXTUAL_PROFILE_TOTAL_CHARACTERS,
    MAX_CONTEXTUAL_RELATIONSHIP_CHARACTERS,
    PersonSocialContextProvider,
)


class IncrementingClock:
    def __init__(self) -> None:
        self._value = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        value = self._value
        self._value += timedelta(minutes=1)
        return value


class SocialContextTests(unittest.TestCase):
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
        self.profiles = ProfileFactRepository(self.database)
        self.relationships = PersonRelationshipRepository(self.database)
        self.observations = ObservationRepository(
            self.database, clock=IncrementingClock()
        )
        self.provider = PersonSocialContextProvider(
            self.profiles,
            self.relationships,
            self.observations,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_provider_isolates_every_social_source_by_person(self) -> None:
        este_fact = self.profiles.create_profile_fact(
            self.este.id, "interest", "Este construit Fripouille."
        )
        self.profiles.create_profile_fact(
            self.alice.id, "interest", "Alice pratique la voile."
        )
        este_relationship = self.relationships.create_relationship(
            self.este.id, "familiar", "direct"
        )
        self.relationships.create_relationship(self.alice.id, "new", "formal")
        este_observation = self.observations.create_observation(
            self.este.id,
            "communication",
            "Semble apprécier les réponses concises.",
        )
        self.observations.create_observation(
            self.alice.id, "context", "Découvre le projet."
        )

        context = self.provider.build(self.este.id)

        self.assertEqual(context.profile_facts, (este_fact,))
        self.assertEqual(context.relationship, este_relationship)
        self.assertEqual(context.observations, (este_observation,))
        self.assertLessEqual(
            len(context.relationship.familiarity)
            + len(context.relationship.interaction_style),
            MAX_CONTEXTUAL_RELATIONSHIP_CHARACTERS,
        )

    def test_provider_applies_independent_count_and_character_budgets(self) -> None:
        for index in range(MAX_CONTEXTUAL_PROFILE_FACTS + 2):
            self.profiles.create_profile_fact(
                self.este.id, "personal_fact", f"{index}:" + "p" * 298
            )
        for index in range(MAX_CONTEXTUAL_OBSERVATIONS + 2):
            self.observations.create_observation(
                self.este.id, "behavior", f"{index}:" + "o" * 198
            )

        context = self.provider.build(self.este.id)

        self.assertLessEqual(len(context.profile_facts), MAX_CONTEXTUAL_PROFILE_FACTS)
        self.assertLessEqual(
            sum(len(item.content) for item in context.profile_facts),
            MAX_CONTEXTUAL_PROFILE_TOTAL_CHARACTERS,
        )
        self.assertEqual(len(context.observations), MAX_CONTEXTUAL_OBSERVATIONS)
        self.assertLessEqual(
            sum(len(item.content) for item in context.observations),
            MAX_CONTEXTUAL_OBSERVATION_TOTAL_CHARACTERS,
        )

    def test_prompt_separates_sources_and_exposes_no_ids_or_raw_evidence(self) -> None:
        self.profiles.create_profile_fact(
            self.este.id,
            "communication_preference",
            "Préfère des réponses directes.",
        )
        self.relationships.create_relationship(self.este.id, "close", "playful")
        self.observations.create_observation(
            self.este.id,
            "communication",
            "Semble apprécier les réponses courtes.",
            source="conversation_analysis",
            source_text="Fais court cette fois.",
            confidence=0.75,
        )
        identity = build_default_identity()
        prompt = build_conversation_prompt(
            identity,
            ActivePersonContext(
                assistant_name=identity.name,
                default_person=self.este,
            ),
            social_context=self.provider.build(self.este.id),
        )

        self.assertLess(
            prompt.index("Confirmed active-person profile data:"),
            prompt.index("Active-person relationship data:"),
        )
        self.assertLess(
            prompt.index("Active-person relationship data:"),
            prompt.index("Unconfirmed active-person observations:"),
        )
        self.assertIn("Préfère des réponses directes.", prompt)
        self.assertIn("Semble apprécier les réponses courtes.", prompt)
        self.assertIn('"status": "unconfirmed"', prompt)
        self.assertIn("never override identity, safety, permissions", prompt)
        self.assertNotIn("source_text", prompt)
        self.assertNotIn("Fais court cette fois.", prompt)
        self.assertNotIn('"person_id"', prompt)
        self.assertNotIn('"id"', prompt)

    def test_empty_social_context_adds_no_social_sections(self) -> None:
        prompt = build_conversation_prompt(
            build_default_identity(),
            social_context=self.provider.build(self.este.id),
        )

        self.assertNotIn("active-person profile data", prompt.casefold())
        self.assertNotIn("relationship data", prompt.casefold())
        self.assertNotIn("active-person observations", prompt.casefold())

    def test_social_sections_precede_independently_bounded_memory(self) -> None:
        self.profiles.create_profile_fact(
            self.este.id, "interest", "S'intéresse à la robotique."
        )
        timestamp = datetime(2026, 9, 4, 13, tzinfo=timezone.utc)
        memory = RetrievedMemory(
            memory=Memory(
                id=42,
                content="Projet Fripouille durable.",
                source="explicit_user",
                source_text=None,
                confidence=1.0,
                created_at=timestamp,
                updated_at=timestamp,
            ),
            score=1.0,
            matched_terms=("fripouille",),
        )

        prompt = build_conversation_prompt(
            build_default_identity(),
            social_context=self.provider.build(self.este.id),
            contextual_memories=(memory,),
        )

        self.assertLess(
            prompt.index("Confirmed active-person profile data:"),
            prompt.index("Contextual memory data:"),
        )
        self.assertNotIn('"id": 42', prompt)


if __name__ == "__main__":
    unittest.main()
