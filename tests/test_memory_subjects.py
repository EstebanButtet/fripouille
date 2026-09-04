"""Tests for deterministic active-person subject attribution."""

from __future__ import annotations

import unittest

from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.subjects import candidate_targets_active_person


def _candidate(source_text: str) -> MemoryCandidate:
    return MemoryCandidate(
        content=source_text,
        source_text=source_text,
        confidence=0.9,
    )


class MemorySubjectPolicyTests(unittest.TestCase):
    def test_explicit_first_person_targets_active_person(self) -> None:
        for statement in (
            "Je travaille sur Fripouille.",
            "J'ai commencé le prototype.",
            "Mon projet utilise Python.",
            "Nous sommes allés à Genève ensemble.",
        ):
            with self.subTest(statement=statement):
                self.assertTrue(
                    candidate_targets_active_person(_candidate(statement))
                )

    def test_general_or_third_party_statement_has_no_automatic_subject(
        self,
    ) -> None:
        for statement in (
            "Paris est la capitale de la France.",
            "Alice travaille chez Exemple SA.",
            "Mon frère travaille chez Exemple SA.",
            "Je sais qu'Alice travaille chez Exemple SA.",
            "Je te rappelle que Paris est la capitale de la France.",
        ):
            with self.subTest(statement=statement):
                self.assertFalse(
                    candidate_targets_active_person(_candidate(statement))
                )

    def test_model_selected_excerpt_cannot_invent_personal_subject(self) -> None:
        candidate = MemoryCandidate(
            content="Paris est la capitale de la France.",
            source_text="Je parle de géographie : Paris est la capitale de la France.",
            confidence=0.9,
        )

        self.assertFalse(candidate_targets_active_person(candidate))


if __name__ == "__main__":
    unittest.main()
