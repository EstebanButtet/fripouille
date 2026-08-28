"""Tests for the persistent memory repository."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from assistant_ia.memory.errors import MemoryNotFoundError
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase


class SequenceClock:
    """Return deterministic timestamps in a predefined order."""

    def __init__(self, *values: datetime) -> None:
        """Store the ordered timestamps returned by the clock."""
        self._values = list(values)

    def __call__(self) -> datetime:
        """Return the next predefined timestamp."""
        if not self._values:
            raise AssertionError(
                "No deterministic clock value remains."
            )

        return self._values.pop(0)


class MemoryRepositoryTests(unittest.TestCase):
    """Validate memory saving, searching and deletion."""

    def setUp(self) -> None:
        """Create an isolated initialized database for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = SQLiteDatabase(
            self.root / "assistant.db"
        )
        self.database.initialize()

        self.first_time = datetime(
            2026,
            8,
            7,
            1,
            0,
            tzinfo=timezone.utc,
        )
        self.second_time = datetime(
            2026,
            8,
            7,
            2,
            0,
            tzinfo=timezone.utc,
        )
        self.third_time = datetime(
            2026,
            8,
            7,
            3,
            0,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        """Remove the isolated temporary database."""
        self.temporary_directory.cleanup()

    def _repository(
        self,
        *clock_values: datetime,
    ) -> MemoryRepository:
        """Create a repository using deterministic clock values."""
        return MemoryRepository(
            self.database,
            clock=SequenceClock(*clock_values),
        )

    def test_save_memory_returns_persisted_model(self) -> None:
        """Saving a memory should return its normalized stored model."""
        repository = self._repository(self.first_time)

        memory = repository.save_memory(
            " Mon examen est le 24 août. "
        )

        self.assertEqual(memory.id, 1)
        self.assertEqual(
            memory.content,
            "Mon examen est le 24 août.",
        )
        self.assertEqual(memory.source, "explicit_user")
        self.assertIsNone(memory.source_text)
        self.assertEqual(memory.confidence, 1.0)
        self.assertEqual(memory.created_at, self.first_time)
        self.assertEqual(memory.updated_at, self.first_time)

    def test_rejects_empty_content_without_writing(self) -> None:
        """Empty memory content should be rejected before persistence."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "Memory content cannot be empty",
        ):
            repository.save_memory("   ")

        with self.assertRaisesRegex(
            ValueError,
            "Memory search query cannot be empty",
        ):
            repository.find_memories(" ")

    def test_persists_between_repository_instances(self) -> None:
        """A saved memory should remain available to another repository."""
        first_repository = self._repository(self.first_time)
        saved_memory = first_repository.save_memory(
            "Information persistante."
        )

        second_repository = self._repository()

        self.assertEqual(
            second_repository.find_memories("persistante"),
            (saved_memory,),
        )

    def test_find_returns_empty_tuple_without_match(self) -> None:
        """A search without matches should return an empty tuple."""
        repository = self._repository(self.first_time)
        repository.save_memory("Information enregistrée.")

        self.assertEqual(
            repository.find_memories("introuvable"),
            (),
        )

    def test_find_matches_content(self) -> None:
        """Searches should return memories containing the query."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        matching_memory = repository.save_memory(
            "Mon examen de biologie approche."
        )
        repository.save_memory(
            "Acheter du pain."
        )

        self.assertEqual(
            repository.find_memories("biologie"),
            (matching_memory,),
        )

    def test_find_is_case_insensitive_for_ascii_text(self) -> None:
        """SQLite searching should ignore ASCII letter case."""
        repository = self._repository(self.first_time)
        memory = repository.save_memory(
            "EXAMEN prévu lundi."
        )

        self.assertEqual(
            repository.find_memories("examen"),
            (memory,),
        )

    def test_find_orders_newest_memories_first(self) -> None:
        """Matching memories should be ordered from newest to oldest."""
        repository = self._repository(
            self.first_time,
            self.second_time,
            self.third_time,
        )
        oldest = repository.save_memory(
            "Premier souvenir commun."
        )
        middle = repository.save_memory(
            "Deuxième souvenir commun."
        )
        newest = repository.save_memory(
            "Troisième souvenir commun."
        )

        self.assertEqual(
            repository.find_memories("souvenir commun"),
            (
                newest,
                middle,
                oldest,
            ),
        )

    def test_find_respects_result_limit(self) -> None:
        """Memory searches should respect their explicit result limit."""
        repository = self._repository(
            self.first_time,
            self.second_time,
            self.third_time,
        )
        repository.save_memory("Souvenir partagé 1.")
        second = repository.save_memory("Souvenir partagé 2.")
        third = repository.save_memory("Souvenir partagé 3.")

        self.assertEqual(
            repository.find_memories(
                "Souvenir partagé",
                limit=2,
            ),
            (
                third,
                second,
            ),
        )

    def test_lists_a_bounded_recent_candidate_set(self) -> None:
        """Repository listing should support bounded local processing."""
        repository = self._repository(
            self.first_time,
            self.second_time,
            self.third_time,
        )
        repository.save_memory("Premier souvenir.")
        second = repository.save_memory("Deuxième souvenir.")
        third = repository.save_memory("Troisième souvenir.")

        self.assertEqual(
            repository.list_memories(limit=2),
            (third, second),
        )

    def test_rejects_invalid_memory_list_limit(self) -> None:
        """Repository candidate listings must remain strictly bounded."""
        repository = self._repository()

        for limit in (0, 1001):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(
                    ValueError,
                    "between 1 and 1000",
                ),
            ):
                repository.list_memories(limit=limit)

    def test_rejects_invalid_result_limit(self) -> None:
        """Memory result limits should remain within safe bounds."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "between 1 and 100",
        ):
            repository.find_memories(
                "souvenir",
                limit=0,
            )

    def test_treats_percent_as_literal_text(self) -> None:
        """Percent signs should not become LIKE wildcards."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        literal_match = repository.save_memory(
            "Progression à 100%."
        )
        repository.save_memory(
            "Progression à 1000."
        )

        self.assertEqual(
            repository.find_memories("%"),
            (literal_match,),
        )

    def test_treats_underscore_as_literal_text(self) -> None:
        """Underscores should not become LIKE wildcards."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        literal_match = repository.save_memory(
            "Code interne code_a."
        )
        repository.save_memory(
            "Code interne codeXa."
        )

        self.assertEqual(
            repository.find_memories("_"),
            (literal_match,),
        )

    def test_treats_escape_character_as_literal_text(self) -> None:
        """Escape characters should remain searchable as literal text."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        literal_match = repository.save_memory(
            "Attention ! Important."
        )
        repository.save_memory(
            "Attention importante."
        )

        self.assertEqual(
            repository.find_memories("!"),
            (literal_match,),
        )

    def test_delete_memory_returns_removed_model(self) -> None:
        """Deleting by identifier should return the removed memory."""
        repository = self._repository(self.first_time)
        memory = repository.save_memory(
            "Souvenir à supprimer."
        )

        deleted_memory = repository.delete_memory(memory.id)

        self.assertEqual(deleted_memory, memory)
        self.assertEqual(
            repository.find_memories("supprimer"),
            (),
        )

    def test_delete_removes_only_selected_identifier(self) -> None:
        """Equal memory contents should remain independently addressable."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        first_memory = repository.save_memory(
            "Même contenu."
        )
        second_memory = repository.save_memory(
            "Même contenu."
        )

        repository.delete_memory(first_memory.id)

        self.assertEqual(
            repository.find_memories("Même contenu"),
            (second_memory,),
        )

    def test_rejects_missing_memory(self) -> None:
        """Deleting an unknown identifier should be explicit."""
        repository = self._repository()

        with self.assertRaisesRegex(
            MemoryNotFoundError,
            "Memory 42 does not exist",
        ):
            repository.delete_memory(42)

    def test_keeps_separate_databases_independent(self) -> None:
        """Memory repositories should remain isolated by database."""
        first_repository = self._repository(self.first_time)
        first_repository.save_memory(
            "Première base."
        )

        second_database = SQLiteDatabase(
            self.root / "second.db"
        )
        second_database.initialize()
        second_repository = MemoryRepository(
            second_database,
            clock=SequenceClock(),
        )

        self.assertEqual(
            second_repository.find_memories("base"),
            (),
        )


if __name__ == "__main__":
    unittest.main()
