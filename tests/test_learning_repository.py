"""Tests for inspectable behavioral experiences and lesson candidates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.outcomes import ExperienceOutcome, OutcomeMeasurement, UserFeedback
from assistant_ia.learning.repository import (
    BehavioralExperienceInUseError,
    BehavioralExperienceNotFoundError,
    BehavioralLearningRepository,
    BehavioralLessonCandidateNotFoundError,
)
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import DatabaseError, SQLiteDatabase
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.observation_repository import ObservationRepository
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        return self._values.pop(0)


class BehavioralLearningRepositoryTests(unittest.TestCase):
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
        self.base_time = datetime(2026, 9, 4, 10, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_experience(
        self,
        repository: BehavioralLearningRepository,
        *,
        person_id: int | None = DEFAULT_PERSON_ID,
        suffix: str = "",
    ):
        return repository.create_experience(
            person_id=person_id,
            context=f"Question conceptuelle{suffix}",
            objective="Faire comprendre le concept",
            strategy="Commencer par une intuition",
            result="La personne reformule correctement",
            evaluation="Approche utile, à réessayer",
            provenance=ExperienceProvenance(
                source_type="conversation_turn",
                source_reference="session-local-turn",
                source_text="Explique-moi d'abord intuitivement.",
            ),
        )

    def test_create_read_and_list_preserve_provenance_and_person_scope(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(self.base_time, self.base_time + timedelta(minutes=1)),
        )
        este = self._create_experience(repository)
        global_experience = self._create_experience(
            repository, person_id=None, suffix=" globale"
        )

        self.assertEqual(repository.get_experience(este.id), este)
        self.assertEqual(este.created_at, self.base_time)
        self.assertEqual(este.updated_at, self.base_time)
        self.assertEqual(este.provenance.source_type, "conversation_turn")
        self.assertEqual(
            repository.list_person_experiences(self.este.id), (este,)
        )
        self.assertEqual(
            repository.list_person_experiences(self.alice.id), ()
        )
        self.assertEqual(
            repository.list_global_experiences(), (global_experience,)
        )

    def test_structured_outcome_feedback_and_measurements_round_trip(self) -> None:
        repository = BehavioralLearningRepository(self.database)
        outcome = ExperienceOutcome(
            status="failure",
            kind="user_feedback",
            summary="La personne demande une correction.",
            result_code="clarification_requested",
            feedback=UserFeedback("correction", "Commence par l'intuition."),
            measurements=(OutcomeMeasurement("durée", 2.5, "s"),),
        )
        experience = repository.create_experience(
            person_id=self.este.id,
            context="Explication",
            objective="Comprendre",
            strategy="Équations d'abord",
            result=outcome.summary,
            provenance=ExperienceProvenance(
                source_type="conversation_turn",
                source_text="Commence par l'intuition.",
            ),
            outcome=outcome,
        )
        loaded = repository.get_experience(experience.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.outcome_status, "failure")
        self.assertEqual(loaded.feedback_kind, "correction")
        self.assertEqual(loaded.feedback_text, "Commence par l'intuition.")
        self.assertEqual(loaded.measurements[0].unit, "s")
    def test_update_preserves_scope_provenance_and_creation_time(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(self.base_time, self.base_time + timedelta(minutes=1)),
        )
        experience = self._create_experience(repository)

        updated = repository.update_experience(
            experience.id,
            context=experience.context,
            objective=experience.objective,
            strategy="Donner d'abord un exemple concret",
            result="La reformulation était partielle",
            evaluation="Nouvelle correction explicite",
        )

        self.assertEqual(updated.person_id, experience.person_id)
        self.assertEqual(updated.provenance, experience.provenance)
        self.assertEqual(updated.created_at, experience.created_at)
        self.assertGreater(updated.updated_at, experience.updated_at)
        self.assertEqual(updated.strategy, "Donner d'abord un exemple concret")

    def test_invalidation_is_visible_but_excluded_from_active_list(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(self.base_time, self.base_time + timedelta(minutes=1)),
        )
        experience = self._create_experience(repository)

        invalidated = repository.invalidate_experience(
            experience.id, "Résultat attribué au mauvais tour."
        )

        self.assertEqual(invalidated.status, "invalidated")
        self.assertEqual(repository.list_person_experiences(self.este.id), ())
        self.assertEqual(
            repository.list_person_experiences(
                self.este.id, include_invalidated=True
            ),
            (invalidated,),
        )
        with self.assertRaisesRegex(ValueError, "already invalidated"):
            repository.invalidate_experience(experience.id, "Encore")

    def test_delete_unlinked_experience_and_missing_errors(self) -> None:
        repository = BehavioralLearningRepository(
            self.database, clock=SequenceClock(self.base_time)
        )
        experience = self._create_experience(repository)

        self.assertEqual(repository.delete_experience(experience.id), experience)
        self.assertIsNone(repository.get_experience(experience.id))
        with self.assertRaises(BehavioralExperienceNotFoundError):
            repository.delete_experience(experience.id)

    def test_candidate_keeps_exact_sources_without_becoming_a_rule(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(self.base_time, self.base_time + timedelta(minutes=1)),
        )
        experience = self._create_experience(repository)

        candidate = repository.create_lesson_candidate(
            source_experience_ids=(experience.id,),
            context_pattern="Demande d'explication conceptuelle",
            proposed_strategy="Commencer par l'intuition",
            rationale="La reformulation correcte a suivi cette stratégie.",
        )

        self.assertEqual(candidate.person_id, self.este.id)
        self.assertEqual(candidate.source_experience_ids, (experience.id,))
        self.assertEqual(
            repository.list_person_lesson_candidates(self.este.id), (candidate,)
        )
        self.assertEqual(
            repository.list_person_lesson_candidates(self.alice.id), ()
        )
        with self.database.connect() as connection:
            rule_tables = connection.execute(
                """
                SELECT COUNT(*) FROM behavioral_rules
                """
            ).fetchall()
        self.assertEqual(rule_tables, [(0,)])
        with self.assertRaises(BehavioralExperienceInUseError):
            repository.delete_experience(experience.id)

    def test_candidate_rejects_missing_invalidated_or_mixed_scope_sources(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
                self.base_time + timedelta(minutes=2),
                self.base_time + timedelta(minutes=3),
            ),
        )
        este = self._create_experience(repository)
        alice = self._create_experience(repository, person_id=self.alice.id)

        with self.assertRaisesRegex(ValueError, "one exact person scope"):
            repository.create_lesson_candidate(
                source_experience_ids=(este.id, alice.id),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )
        with self.assertRaises(BehavioralExperienceNotFoundError):
            repository.create_lesson_candidate(
                source_experience_ids=(999,),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )
        repository.invalidate_experience(este.id, "Source erronée")
        with self.assertRaisesRegex(ValueError, "Invalidated experiences"):
            repository.create_lesson_candidate(
                source_experience_ids=(este.id,),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )

    def test_candidate_can_be_corrected_invalidated_deleted_then_source_deleted(self) -> None:
        repository = BehavioralLearningRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
                self.base_time + timedelta(minutes=2),
                self.base_time + timedelta(minutes=3),
            ),
        )
        experience = self._create_experience(repository)
        candidate = repository.create_lesson_candidate(
            source_experience_ids=(experience.id,),
            context_pattern="Ancien contexte",
            proposed_strategy="Ancienne stratégie",
            rationale="Première formulation",
        )

        corrected = repository.update_lesson_candidate(
            candidate.id,
            context_pattern="Contexte corrigé",
            proposed_strategy="Stratégie corrigée",
            rationale="Correction manuelle",
        )
        invalidated = repository.invalidate_lesson_candidate(
            candidate.id, "Contre-exemple découvert."
        )

        self.assertEqual(corrected.source_experience_ids, (experience.id,))
        self.assertEqual(invalidated.status, "invalidated")
        self.assertEqual(
            repository.list_person_lesson_candidates(self.este.id), ()
        )
        self.assertEqual(
            repository.list_person_lesson_candidates(
                self.este.id, include_invalidated=True
            ),
            (invalidated,),
        )
        self.assertEqual(repository.delete_lesson_candidate(candidate.id), invalidated)
        self.assertEqual(repository.delete_experience(experience.id).id, experience.id)
        with self.assertRaises(BehavioralLessonCandidateNotFoundError):
            repository.delete_lesson_candidate(candidate.id)

    def test_foreign_key_and_domain_separation_are_enforced(self) -> None:
        repository = BehavioralLearningRepository(
            self.database, clock=SequenceClock(self.base_time)
        )

        with self.assertRaises(DatabaseError):
            self._create_experience(repository, person_id=999)

        self.assertEqual(MemoryRepository(self.database).list_memories(), ())
        self.assertEqual(
            ProfileFactRepository(self.database).list_profile_facts(self.este.id), ()
        )
        self.assertEqual(
            ObservationRepository(self.database).list_observations(self.este.id), ()
        )


if __name__ == "__main__":
    unittest.main()
