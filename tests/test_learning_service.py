"""Tests for the application-owned behavioral learning entry point."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.application import build_default_assistant
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.repository import BehavioralLearningRepository
from assistant_ia.learning.service import (
    BehavioralLearningScopeError,
    BehavioralLearningService,
)
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.models import PersonProfile
from assistant_ia.people.person_repository import PersonRepository


class FakeModelClient:
    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        return ModelResponse(
            content="Réponse sans apprentissage automatique.",
            model="fake-model",
            intent=Intent(name="conversation"),
        )


class BehavioralLearningServiceTests(unittest.TestCase):
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
        self.person_context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=self.este,
        )
        self.repository = BehavioralLearningRepository(self.database)
        self.service = BehavioralLearningService(
            self.repository,
            self.person_context,
        )
        self.provenance = ExperienceProvenance(
            source_type="manual_entry",
            source_reference="audit-manuel",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _record_active(self):
        return self.service.record_active_person_experience(
            context="Question conceptuelle",
            objective="Faire comprendre",
            strategy="Commencer par l'intuition",
            result="Reformulation correcte",
            evaluation=None,
            provenance=self.provenance,
        )

    def test_records_for_application_resolved_active_person(self) -> None:
        este_experience = self._record_active()
        self.person_context.set_active_persistent_person(self.alice)
        alice_experience = self._record_active()

        self.assertEqual(este_experience.person_id, self.este.id)
        self.assertEqual(alice_experience.person_id, self.alice.id)
        self.assertEqual(
            self.service.list_active_person_experiences(),
            (alice_experience,),
        )
        self.person_context.reset()
        self.assertEqual(
            self.service.list_active_person_experiences(),
            (este_experience,),
        )

    def test_unresolved_session_person_cannot_receive_persistent_learning(self) -> None:
        self.person_context.set_active_person(PersonProfile(name="Visiteur"))

        with self.assertRaisesRegex(
            BehavioralLearningScopeError,
            "resolved persistent person",
        ):
            self._record_active()
        self.assertEqual(self.repository.list_global_experiences(), ())

    def test_global_records_are_explicit_and_separate(self) -> None:
        personal = self._record_active()
        global_experience = self.service.record_global_experience(
            context="Commande applicative",
            objective="Exécuter une action validée",
            strategy="Passer par le registre",
            result="Action refusée sans permission",
            provenance=ExperienceProvenance(
                source_type="action_execution",
                source_reference="launch_application",
            ),
        )

        self.assertEqual(global_experience.person_id, None)
        self.assertEqual(
            self.repository.list_global_experiences(), (global_experience,)
        )
        self.assertEqual(
            self.service.list_active_person_experiences(), (personal,)
        )
        candidate = self.service.propose_global_lesson(
            source_experiences=(global_experience,),
            context_pattern="Action applicative sans permission",
            proposed_strategy="Conserver le refus par défaut",
            rationale="Le registre a correctement bloqué l'action.",
        )
        self.assertIsNone(candidate.person_id)
        self.assertEqual(
            self.repository.list_global_lesson_candidates(), (candidate,)
        )

    def test_experience_never_creates_a_lesson_candidate_automatically(self) -> None:
        experience = self._record_active()

        self.assertEqual(
            self.service.list_active_person_lesson_candidates(), ()
        )
        candidate = self.service.propose_active_person_lesson(
            source_experiences=(experience,),
            context_pattern="Question conceptuelle",
            proposed_strategy="Commencer par l'intuition",
            rationale="Une expérience observée, encore insuffisante pour une règle.",
        )

        self.assertEqual(candidate.source_experience_ids, (experience.id,))
        self.assertEqual(candidate.status, "active")
        self.assertFalse(hasattr(candidate, "confirmed"))

    def test_candidate_rejects_other_scope_and_stale_objects(self) -> None:
        este_experience = self._record_active()
        self.person_context.set_active_persistent_person(self.alice)

        with self.assertRaisesRegex(
            BehavioralLearningScopeError,
            "requested person scope",
        ):
            self.service.propose_active_person_lesson(
                source_experiences=(este_experience,),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )

        self.person_context.reset()
        stale = replace(este_experience, result="Résultat forgé")
        with self.assertRaisesRegex(
            BehavioralLearningScopeError,
            "current persisted experiences",
        ):
            self.service.propose_active_person_lesson(
                source_experiences=(stale,),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )

    def test_global_candidate_rejects_personal_sources(self) -> None:
        personal = self._record_active()

        with self.assertRaisesRegex(
            BehavioralLearningScopeError,
            "requested person scope",
        ):
            self.service.propose_global_lesson(
                source_experiences=(personal,),
                context_pattern="Contexte",
                proposed_strategy="Stratégie",
                rationale="Raison",
            )

    def test_default_application_exposes_service_but_records_nothing_implicitly(self) -> None:
        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(),
        )

        response = assistant.process_message("Explique ce concept.")

        self.assertEqual(response, "Réponse sans apprentissage automatique.")
        self.assertIsInstance(
            assistant.behavioral_learning_service,
            BehavioralLearningService,
        )
        service = assistant.behavioral_learning_service
        assert service is not None
        self.assertEqual(service.list_active_person_experiences(), ())
        self.assertEqual(service.list_active_person_lesson_candidates(), ())

    def test_core_rejects_an_invalid_learning_service(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "must be a BehavioralLearningService",
        ):
            AssistantCore(
                model_client=FakeModelClient(),
                behavioral_learning_service="repository",
            )


if __name__ == "__main__":
    unittest.main()
