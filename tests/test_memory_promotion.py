"""Tests for deterministic and controlled memory promotion decisions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.promotion import (
    MemoryPromotionProposal,
    MemoryPromotionService,
    normalize_memory_equivalence,
)
from assistant_ia.memory.repository import SQLiteDatabase


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values)

    def __call__(self) -> datetime:
        return self.values.pop(0)


class MemoryPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.first_time = datetime(2026, 8, 28, tzinfo=timezone.utc)
        self.second_time = self.first_time + timedelta(hours=1)
        self.repository = MemoryRepository(
            self.database,
            clock=SequenceClock(self.first_time, self.second_time),
        )
        self.service = MemoryPromotionService(self.repository)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _candidate(
        self,
        content: str,
        source_text: str | None = None,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            content=content,
            source_text=source_text or content,
            confidence=0.9,
        )

    def test_new_candidate_proposes_create_without_writing(self) -> None:
        candidate = self._candidate("Mon projet durable est Fripouille.")

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "create")
        self.assertIsNone(proposal.related_memory)
        self.assertTrue(proposal.requires_confirmation)
        self.assertEqual(self.repository.list_memories(), ())

    def test_exact_duplicate_is_known_without_timestamp_change(self) -> None:
        original = self.repository.save_memory(
            "Mon logiciel prefere est SolidWorks."
        )
        candidate = self._candidate(
            "  MON logiciel prefere est SolidWorks!!!  "
        )

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "already_known")
        self.assertEqual(proposal.related_memory, original)
        self.assertFalse(proposal.requires_confirmation)
        self.assertEqual(self.repository.list_memories(), (original,))
        self.assertEqual(
            self.repository.list_memories()[0].updated_at,
            original.updated_at,
        )

    def test_quasi_duplicate_is_never_merged_automatically(self) -> None:
        original = self.repository.save_memory(
            "Mon logiciel prefere pour la CAO est SolidWorks."
        )
        candidate = self._candidate(
            "Je prefere SolidWorks pour faire de la CAO."
        )

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "possible_duplicate")
        self.assertEqual(proposal.related_memory, original)
        self.assertEqual(self.repository.list_memories(), (original,))

    def test_explicit_correction_selects_application_owned_memory(self) -> None:
        original = self.repository.save_memory(
            "Mon logiciel prefere pour la CAO est SolidWorks."
        )
        candidate = self._candidate(
            "En fait, je prefere maintenant Fusion 360 pour la CAO."
        )

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "update")
        self.assertEqual(proposal.related_memory.id, original.id)
        self.assertEqual(self.repository.list_memories(), (original,))

    def test_non_corrective_preference_conflict_is_not_resolved(self) -> None:
        original = self.repository.save_memory("Je prefere SolidWorks.")
        candidate = self._candidate("Je prefere Fusion 360.")

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "conflict")
        self.assertEqual(proposal.related_memory, original)
        self.assertEqual(self.repository.list_memories(), (original,))

    def test_unrelated_preferences_are_not_classified_as_conflict(self) -> None:
        self.repository.save_memory(
            "Mon logiciel prefere pour la CAO est SolidWorks."
        )
        candidate = self._candidate(
            "Je prefere boire du cafe sans sucre."
        )

        proposal = self.service.propose(candidate)

        self.assertEqual(proposal.operation, "create")

    def test_confirmed_create_and_update_use_repository_authority(self) -> None:
        create = self.service.propose(
            self._candidate("Mon outil est SolidWorks.")
        )
        created = self.service.apply_confirmed(create)
        correction = self.service.propose(
            self._candidate("En fait, mon outil est Fusion 360.")
        )
        updated = self.service.apply_confirmed(correction)

        self.assertEqual(updated.id, created.id)
        self.assertEqual(updated.created_at, created.created_at)
        self.assertEqual(updated.content, "En fait, mon outil est Fusion 360.")
        self.assertEqual(len(self.repository.list_memories()), 1)

    def test_proposal_is_strict_and_immutable(self) -> None:
        candidate = self._candidate("Mon projet est Fripouille.")
        proposal = MemoryPromotionProposal("create", candidate)

        with self.assertRaises(FrozenInstanceError):
            proposal.operation = "conflict"
        with self.assertRaises(TypeError):
            MemoryPromotionProposal("update", candidate)
        with self.assertRaises(ValueError):
            MemoryPromotionProposal("invalid", candidate)

    def test_invalid_candidate_cannot_be_proposed(self) -> None:
        with self.assertRaises(TypeError):
            self.service.propose("not a candidate")

    def test_equivalence_normalizes_unicode_spacing_case_and_punctuation(
        self,
    ) -> None:
        self.assertEqual(
            normalize_memory_equivalence("  Café, PRÉFÉRÉ ! "),
            "cafe prefere",
        )


if __name__ == "__main__":
    unittest.main()
