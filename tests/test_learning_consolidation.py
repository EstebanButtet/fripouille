"""Tests for deterministic evidence consolidation and explicit rule confirmation."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.outcomes import ExperienceOutcome
from assistant_ia.learning.repository import BehavioralLearningRepository
from assistant_ia.learning.service import BehavioralLearningService, BehavioralLearningScopeError
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.person_repository import PersonRepository


class ConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.db = SQLiteDatabase(Path(self.tmp.name) / "assistant.db")
        self.db.initialize()
        people = PersonRepository(self.db)
        person = people.get_person(DEFAULT_PERSON_ID)
        assert person is not None
        self.context = ActivePersonContext(assistant_name="Fripouille", default_person=person)
        self.repo = BehavioralLearningRepository(self.db)
        self.service = BehavioralLearningService(self.repo, self.context)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _experience(self, strategy: str, status: str, source: str):
        attempt = self.service.begin_active_person_attempt(
            context="explication", objective="comprendre", strategy=strategy
        )
        return self.service.record_active_person_outcome(
            attempt,
            ExperienceOutcome(status=status, kind="reported_result", summary=f"Résultat {source}"),
            provenance=ExperienceProvenance(source_type="manual_entry", source_reference=source),
        )

    def test_consolidation_classifies_favorable_contradictory_and_invalidated(self) -> None:
        favorable = self._experience("intuition d'abord", "success", "a")
        contradictory = self._experience("équations d'abord", "success", "b")
        invalidated = self._experience("intuition d'abord", "success", "c")
        candidate = self.service.propose_active_person_lesson(
            source_experiences=(favorable, contradictory, invalidated),
            context_pattern="explication",
            proposed_strategy="intuition d'abord",
            rationale="Proposition à examiner.",
        )
        self.repo.invalidate_experience(invalidated.id, "Correction utilisateur")
        summary = self.service.consolidate_lesson_candidate(candidate)
        self.assertEqual(summary.favorable_experience_ids, (favorable.id,))
        self.assertEqual(summary.contradictory_experience_ids, (contradictory.id,))
        self.assertEqual(summary.excluded_experience_ids, (invalidated.id,))
        self.assertFalse(summary.can_be_confirmed)
        with self.assertRaises(ValueError):
            self.service.confirm_active_person_lesson(candidate, "Je confirme")

    def test_explicit_confirmation_persists_rule_with_sources_and_scope(self) -> None:
        first = self._experience("intuition d'abord", "success", "first")
        second = self._experience("intuition d'abord", "partial", "second")
        candidate = self.service.propose_active_person_lesson(
            source_experiences=(first, second), context_pattern="explication",
            proposed_strategy="intuition d'abord", rationale="Deux retours favorables.",
        )
        rule = self.service.confirm_active_person_lesson(candidate, "Confirmé par l'application")
        self.assertEqual(rule.person_id, DEFAULT_PERSON_ID)
        self.assertEqual(rule.source_experience_ids, (first.id, second.id))
        self.assertEqual(self.repo.list_rules(person_id=DEFAULT_PERSON_ID), (rule,))
        self.assertEqual(self.repo.list_rules(person_id=None), ())
        invalidated = self.repo.invalidate_confirmed_rule(rule.id, "Règle corrigée")
        self.assertEqual(invalidated.status, "invalidated")
        self.repo.delete_confirmed_rule(rule.id)
        self.assertEqual(self.repo.list_rules(person_id=DEFAULT_PERSON_ID), ())

    def test_scope_and_global_rules_remain_separate(self) -> None:
        global_attempt = self.service.begin_global_attempt(
            context="action", objective="réussir", strategy="procédure"
        )
        global_experience = self.service.record_global_outcome(
            global_attempt,
            ExperienceOutcome("success", "reported_result", "Réussi."),
            provenance=ExperienceProvenance(source_type="manual_entry", source_reference="global"),
        )
        candidate = self.service.propose_global_lesson(
            source_experiences=(global_experience,), context_pattern="action",
            proposed_strategy="procédure", rationale="Résultat global.",
        )
        rule = self.service.confirm_global_lesson(candidate, "Confirmation globale")
        self.assertIsNone(rule.person_id)
        self.context.set_active_person(self.context.default_person)
        with self.assertRaises(BehavioralLearningScopeError):
            self.service.confirm_active_person_lesson(candidate, "Mauvaise portée")


if __name__ == "__main__":
    unittest.main()
