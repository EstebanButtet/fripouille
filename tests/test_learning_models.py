"""Tests for explicit behavioral learning domain boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from assistant_ia.learning.models import (
    BehavioralExperience,
    BehavioralLessonCandidate,
    ExperienceProvenance,
)


class LearningModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 4, 10, tzinfo=timezone.utc)
        self.provenance = ExperienceProvenance(
            source_type="conversation_turn",
            source_reference="session-local-turn",
            source_text="Explique-moi avec une intuition.",
        )

    def test_experience_normalizes_content_and_preserves_exact_evidence(self) -> None:
        experience = BehavioralExperience(
            id=1,
            person_id=2,
            context="  Question conceptuelle  ",
            objective=" Comprendre ",
            strategy=" Commencer par une intuition ",
            result=" La personne reformule correctement ",
            evaluation=None,
            provenance=self.provenance,
            status="active",
            invalidation_reason=None,
            created_at=self.now,
            updated_at=self.now,
        )

        self.assertEqual(experience.context, "Question conceptuelle")
        self.assertEqual(
            experience.provenance.source_text,
            "Explique-moi avec une intuition.",
        )

    def test_provenance_requires_evidence_for_conversation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact source text"):
            ExperienceProvenance(source_type="conversation_turn")

    def test_provenance_requires_reference_for_actions_and_external_systems(self) -> None:
        for source_type in ("action_execution", "external_system"):
            with self.subTest(source_type=source_type):
                with self.assertRaisesRegex(ValueError, "source reference"):
                    ExperienceProvenance(source_type=source_type)

    def test_invalidation_status_and_reason_must_agree(self) -> None:
        experience = BehavioralExperience(
            id=1,
            person_id=None,
            context="Contexte",
            objective="Objectif",
            strategy="Stratégie",
            result="Résultat",
            evaluation=None,
            provenance=ExperienceProvenance(source_type="manual_entry"),
            status="active",
            invalidation_reason=None,
            created_at=self.now,
            updated_at=self.now,
        )

        with self.assertRaisesRegex(ValueError, "require a reason"):
            replace(experience, status="invalidated")
        with self.assertRaisesRegex(ValueError, "cannot have"):
            replace(experience, invalidation_reason="Erreur de saisie")

    def test_lesson_candidate_requires_unique_source_experiences(self) -> None:
        common = {
            "id": 1,
            "person_id": None,
            "context_pattern": "Question conceptuelle",
            "proposed_strategy": "Commencer par l'intuition",
            "rationale": "Résultat observé",
            "status": "active",
            "invalidation_reason": None,
            "created_at": self.now,
            "updated_at": self.now,
        }

        with self.assertRaisesRegex(ValueError, "at least one"):
            BehavioralLessonCandidate(source_experience_ids=(), **common)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            BehavioralLessonCandidate(source_experience_ids=(1, 1), **common)

    def test_models_have_no_confirmed_rule_or_identity_fields(self) -> None:
        experience_fields = BehavioralExperience.__dataclass_fields__
        candidate_fields = BehavioralLessonCandidate.__dataclass_fields__

        self.assertNotIn("identity", experience_fields)
        self.assertNotIn("role", experience_fields)
        self.assertNotIn("mood", experience_fields)
        self.assertNotIn("confirmed", candidate_fields)
        self.assertNotIn("rule", candidate_fields)


if __name__ == "__main__":
    unittest.main()
