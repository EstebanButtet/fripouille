"""Tests for the persistent journal repository."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from assistant_ia.memory.journal_repository import JournalRepository
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


class JournalRepositoryTests(unittest.TestCase):
    """Validate journal entry persistence."""

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
            2,
            0,
            tzinfo=timezone.utc,
        )
        self.second_time = datetime(
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
    ) -> JournalRepository:
        """Create a repository using deterministic clock values."""
        return JournalRepository(
            self.database,
            clock=SequenceClock(*clock_values),
        )

    def _stored_entry_count(self) -> int:
        """Return the number of entries stored in the test database."""
        with self.database.connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM journal_entries
                """
            ).fetchone()

        if result is None:
            raise AssertionError(
                "Journal entry count could not be read."
            )

        return result[0]

    def test_write_journal_entry_returns_persisted_model(self) -> None:
        """Writing an entry should return its normalized stored model."""
        repository = self._repository(self.first_time)

        entry = repository.write_journal_entry(
            " Journée productive. ",
            entry_date=date(2026, 8, 7),
        )

        self.assertEqual(entry.id, 1)
        self.assertEqual(entry.content, "Journée productive.")
        self.assertEqual(entry.entry_date, date(2026, 8, 7))
        self.assertEqual(entry.created_at, self.first_time)

    def test_rejects_empty_content_without_writing(self) -> None:
        """Empty journal content should be rejected before persistence."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "Journal content cannot be empty",
        ):
            repository.write_journal_entry(
                "   ",
                entry_date=date(2026, 8, 7),
            )

        self.assertEqual(self._stored_entry_count(), 0)

    def test_rejects_datetime_as_entry_date(self) -> None:
        """Journal dates should not contain time information."""
        repository = self._repository()

        with self.assertRaisesRegex(
            TypeError,
            "Journal entry date must be a date",
        ):
            repository.write_journal_entry(
                "Entrée.",
                entry_date=datetime(
                    2026,
                    8,
                    7,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            )

        self.assertEqual(self._stored_entry_count(), 0)

    def test_rejects_non_date_entry_date(self) -> None:
        """Journal dates should be actual date objects."""
        repository = self._repository()

        with self.assertRaisesRegex(
            TypeError,
            "Journal entry date must be a date",
        ):
            repository.write_journal_entry(
                "Entrée.",
                entry_date="2026-08-07",
            )

        self.assertEqual(self._stored_entry_count(), 0)

    def test_rejects_naive_creation_time(self) -> None:
        """Technical journal timestamps should include a timezone."""
        repository = self._repository(
            datetime(2026, 8, 7, 2, 0)
        )

        with self.assertRaisesRegex(
            ValueError,
            "must include timezone information",
        ):
            repository.write_journal_entry(
                "Entrée.",
                entry_date=date(2026, 8, 7),
            )

        self.assertEqual(self._stored_entry_count(), 0)

    def test_persists_iso_values_between_connections(self) -> None:
        """Journal values should be persisted in unambiguous ISO formats."""
        repository = self._repository(self.first_time)

        repository.write_journal_entry(
            "Entrée persistante.",
            entry_date=date(2026, 8, 7),
        )

        with self.database.connect() as connection:
            stored_row = connection.execute(
                """
                SELECT
                    content,
                    entry_date,
                    created_at
                FROM journal_entries
                WHERE id = ?
                """,
                (1,),
            ).fetchone()

        self.assertEqual(
            stored_row,
            (
                "Entrée persistante.",
                "2026-08-07",
                "2026-08-07T02:00:00+00:00",
            ),
        )

    def test_assigns_stable_identifiers(self) -> None:
        """Each journal entry should receive a stable identifier."""
        repository = self._repository(
            self.first_time,
            self.second_time,
        )

        first_entry = repository.write_journal_entry(
            "Première entrée.",
            entry_date=date(2026, 8, 7),
        )
        second_entry = repository.write_journal_entry(
            "Deuxième entrée.",
            entry_date=date(2026, 8, 8),
        )

        self.assertEqual(first_entry.id, 1)
        self.assertEqual(second_entry.id, 2)
        self.assertEqual(self._stored_entry_count(), 2)

    def test_keeps_separate_databases_independent(self) -> None:
        """Journal repositories should remain isolated by database."""
        first_repository = self._repository(self.first_time)
        first_repository.write_journal_entry(
            "Première base.",
            entry_date=date(2026, 8, 7),
        )

        second_database = SQLiteDatabase(
            self.root / "second.db"
        )
        second_database.initialize()

        with second_database.connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM journal_entries
                """
            ).fetchone()

        self.assertEqual(result, (0,))


if __name__ == "__main__":
    unittest.main()
