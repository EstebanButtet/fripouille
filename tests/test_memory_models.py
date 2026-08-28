"""Tests for persistent task, memory and journal models."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

from assistant_ia.memory.models import (
    JournalEntry,
    Memory,
    Task,
)


class TaskTests(unittest.TestCase):
    """Validate persistent task model behavior."""

    def setUp(self) -> None:
        """Create deterministic timestamps for task tests."""
        self.created_at = datetime(
            2026,
            8,
            7,
            1,
            0,
            tzinfo=timezone.utc,
        )
        self.completed_at = datetime(
            2026,
            8,
            7,
            2,
            0,
            tzinfo=timezone.utc,
        )

    def test_normalizes_valid_pending_task(self) -> None:
        """A valid pending task should normalize its fields."""
        local_timezone = timezone(timedelta(hours=2))
        due_at = datetime(
            2026,
            8,
            8,
            10,
            0,
            tzinfo=local_timezone,
        )

        task = Task(
            id=1,
            title=" Réviser la biologie ",
            due_at=due_at,
            status="pending",
            created_at=self.created_at,
            completed_at=None,
        )

        self.assertEqual(task.id, 1)
        self.assertEqual(task.title, "Réviser la biologie")
        self.assertEqual(
            task.due_at,
            datetime(
                2026,
                8,
                8,
                8,
                0,
                tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.completed_at)

    def test_accepts_valid_completed_task(self) -> None:
        """A completed task should contain a completion timestamp."""
        task = Task(
            id=3,
            title="Réviser",
            due_at=None,
            status="completed",
            created_at=self.created_at,
            completed_at=self.completed_at,
        )

        self.assertEqual(task.status, "completed")
        self.assertEqual(task.completed_at, self.completed_at)

    def test_rejects_invalid_identifier(self) -> None:
        """Task identifiers should be strictly positive integers."""
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            Task(
                id=0,
                title="Réviser",
                due_at=None,
                status="pending",
                created_at=self.created_at,
                completed_at=None,
            )

    def test_rejects_empty_title(self) -> None:
        """Task titles should contain meaningful text."""
        with self.assertRaisesRegex(
            ValueError,
            "Task title cannot be empty",
        ):
            Task(
                id=1,
                title="   ",
                due_at=None,
                status="pending",
                created_at=self.created_at,
                completed_at=None,
            )

    def test_rejects_unknown_status(self) -> None:
        """Task statuses should belong to the authorized set."""
        with self.assertRaisesRegex(
            ValueError,
            "Unknown task status",
        ):
            Task(
                id=1,
                title="Réviser",
                due_at=None,
                status="cancelled",
                created_at=self.created_at,
                completed_at=None,
            )

    def test_rejects_pending_task_with_completion_time(self) -> None:
        """Pending tasks cannot already contain a completion time."""
        with self.assertRaisesRegex(
            ValueError,
            "Pending tasks cannot have a completion time",
        ):
            Task(
                id=1,
                title="Réviser",
                due_at=None,
                status="pending",
                created_at=self.created_at,
                completed_at=self.completed_at,
            )

    def test_rejects_completed_task_without_completion_time(self) -> None:
        """Completed tasks should contain a completion time."""
        with self.assertRaisesRegex(
            ValueError,
            "Completed tasks must have a completion time",
        ):
            Task(
                id=1,
                title="Réviser",
                due_at=None,
                status="completed",
                created_at=self.created_at,
                completed_at=None,
            )

    def test_rejects_completion_before_creation(self) -> None:
        """Completion cannot occur before task creation."""
        with self.assertRaisesRegex(
            ValueError,
            "cannot precede",
        ):
            Task(
                id=1,
                title="Réviser",
                due_at=None,
                status="completed",
                created_at=self.completed_at,
                completed_at=self.created_at,
            )

    def test_rejects_naive_datetime(self) -> None:
        """Technical task timestamps should include a timezone."""
        with self.assertRaisesRegex(
            ValueError,
            "must include timezone information",
        ):
            Task(
                id=1,
                title="Réviser",
                due_at=None,
                status="pending",
                created_at=datetime(2026, 8, 7, 1, 0),
                completed_at=None,
            )

    def test_is_immutable(self) -> None:
        """Persisted tasks should be immutable."""
        task = Task(
            id=1,
            title="Réviser",
            due_at=None,
            status="pending",
            created_at=self.created_at,
            completed_at=None,
        )

        with self.assertRaises(FrozenInstanceError):
            task.title = "Dormir"


class MemoryTests(unittest.TestCase):
    """Validate persistent memory model behavior."""

    def setUp(self) -> None:
        """Create deterministic timestamps for memory tests."""
        self.created_at = datetime(
            2026,
            8,
            7,
            2,
            0,
            tzinfo=timezone.utc,
        )
        self.updated_at = datetime(
            2026,
            8,
            7,
            3,
            0,
            tzinfo=timezone.utc,
        )

    def _memory(self, **overrides: object) -> Memory:
        """Build one memory while allowing focused invalid values."""
        values = {
            "id": 2,
            "content": "Mon examen est le 24 août.",
            "source": "explicit_user",
            "source_text": None,
            "confidence": 1.0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        values.update(overrides)

        return Memory(**values)  # type: ignore[arg-type]

    def test_normalizes_valid_memory(self) -> None:
        """A valid v3 memory should normalize its stored fields."""
        local_timezone = timezone(timedelta(hours=2))

        memory = self._memory(
            content=" Mon examen est le 24 août. ",
            source=" explicit_user ",
            source_text="  Je passe mon examen le 24 août.  ",
            confidence=0.75,
            created_at=datetime(
                2026,
                8,
                7,
                4,
                0,
                tzinfo=local_timezone,
            ),
            updated_at=datetime(
                2026,
                8,
                7,
                5,
                0,
                tzinfo=local_timezone,
            ),
        )

        self.assertEqual(memory.id, 2)
        self.assertEqual(
            memory.content,
            "Mon examen est le 24 août.",
        )
        self.assertEqual(memory.source, "explicit_user")
        self.assertEqual(
            memory.source_text,
            "  Je passe mon examen le 24 août.  ",
        )
        self.assertEqual(memory.confidence, 0.75)
        self.assertEqual(
            memory.created_at,
            datetime(
                2026,
                8,
                7,
                2,
                0,
                tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(
            memory.updated_at,
            datetime(
                2026,
                8,
                7,
                3,
                0,
                tzinfo=timezone.utc,
            ),
        )

    def test_rejects_empty_memory_content(self) -> None:
        """Memories should contain meaningful text."""
        with self.assertRaisesRegex(
            ValueError,
            "Memory content cannot be empty",
        ):
            self._memory(
                content=" ",
            )

    def test_rejects_empty_or_unknown_source(self) -> None:
        """Memory provenance must be one of the known mechanisms."""
        with self.assertRaisesRegex(
            ValueError,
            "Memory source cannot be empty",
        ):
            self._memory(source=" ")

        with self.assertRaisesRegex(
            ValueError,
            "Unknown memory source",
        ):
            self._memory(source="invented")

    def test_accepts_null_source_text(self) -> None:
        """Exact source evidence may be unavailable."""
        memory = self._memory(source_text=None)

        self.assertIsNone(memory.source_text)

    def test_accepts_non_empty_exact_source_text(self) -> None:
        """Available user evidence should be preserved exactly."""
        memory = self._memory(
            source_text="  Texte utilisateur exact.  "
        )

        self.assertEqual(
            memory.source_text,
            "  Texte utilisateur exact.  ",
        )

    def test_rejects_empty_source_text(self) -> None:
        """Present source evidence cannot be blank."""
        with self.assertRaisesRegex(
            ValueError,
            "Memory source text cannot be empty",
        ):
            self._memory(source_text="   ")

    def test_accepts_confidence_boundaries(self) -> None:
        """Both inclusive confidence boundaries should be valid."""
        for confidence in (0.0, 1.0):
            with self.subTest(confidence=confidence):
                memory = self._memory(
                    confidence=confidence
                )

                self.assertEqual(
                    memory.confidence,
                    confidence,
                )

    def test_rejects_confidence_outside_boundaries(self) -> None:
        """Confidence must remain within its inclusive interval."""
        for confidence in (-0.01, 1.01):
            with (
                self.subTest(confidence=confidence),
                self.assertRaisesRegex(
                    ValueError,
                    "between 0.0 and 1.0",
                ),
            ):
                self._memory(
                    confidence=confidence
                )

    def test_rejects_update_before_creation(self) -> None:
        """A memory cannot be updated before it was created."""
        with self.assertRaisesRegex(
            ValueError,
            "cannot precede",
        ):
            self._memory(
                created_at=self.updated_at,
                updated_at=self.created_at,
            )

    def test_is_immutable(self) -> None:
        """Persistent memory metadata should remain immutable."""
        memory = self._memory()

        with self.assertRaises(FrozenInstanceError):
            memory.content = "Autre contenu."


class JournalEntryTests(unittest.TestCase):
    """Validate persistent journal entry model behavior."""

    def test_normalizes_valid_journal_entry(self) -> None:
        """A valid journal entry should normalize its content."""
        entry = JournalEntry(
            id=4,
            content=" Journée productive. ",
            entry_date=date(2026, 8, 7),
            created_at=datetime(
                2026,
                8,
                7,
                2,
                30,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(entry.id, 4)
        self.assertEqual(entry.content, "Journée productive.")
        self.assertEqual(entry.entry_date, date(2026, 8, 7))

    def test_rejects_datetime_as_entry_date(self) -> None:
        """Journal entry dates should be dates without a time."""
        with self.assertRaisesRegex(
            TypeError,
            "must be a date",
        ):
            JournalEntry(
                id=1,
                content="Entrée.",
                entry_date=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )

    def test_rejects_empty_journal_content(self) -> None:
        """Journal entries should contain meaningful text."""
        with self.assertRaisesRegex(
            ValueError,
            "Journal content cannot be empty",
        ):
            JournalEntry(
                id=1,
                content="   ",
                entry_date=date(2026, 8, 7),
                created_at=datetime.now(timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
