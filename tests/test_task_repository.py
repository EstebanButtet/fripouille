"""Tests for the persistent task repository."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from assistant_ia.memory.errors import (
    RepositoryError,
    TaskAlreadyCompletedError,
    TaskNotFoundError,
)
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.task_repository import TaskRepository


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


class TaskRepositoryTests(unittest.TestCase):
    """Validate task creation, listing and completion."""

    def setUp(self) -> None:
        """Create an isolated initialized database for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = SQLiteDatabase(
            self.root / "assistant.db"
        )
        self.database.initialize()

        self.created_at = datetime(
            2026,
            8,
            7,
            2,
            0,
            tzinfo=timezone.utc,
        )
        self.second_created_at = datetime(
            2026,
            8,
            7,
            3,
            0,
            tzinfo=timezone.utc,
        )
        self.completed_at = datetime(
            2026,
            8,
            7,
            4,
            0,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        """Remove the isolated temporary database."""
        self.temporary_directory.cleanup()

    def _repository(
        self,
        *clock_values: datetime,
    ) -> TaskRepository:
        """Create a repository using deterministic clock values."""
        return TaskRepository(
            self.database,
            clock=SequenceClock(*clock_values),
        )

    def test_create_task_returns_persisted_model(self) -> None:
        """Creating a task should return its stored model."""
        local_timezone = timezone(timedelta(hours=2))
        due_at = datetime(
            2026,
            8,
            8,
            10,
            0,
            tzinfo=local_timezone,
        )
        repository = self._repository(self.created_at)

        task = repository.create_task(
            " Réviser la biologie ",
            due_at=due_at,
        )

        self.assertEqual(task.id, 1)
        self.assertEqual(task.title, "Réviser la biologie")
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.created_at, self.created_at)
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
        self.assertIsNone(task.completed_at)

    def test_rejects_empty_title_without_writing(self) -> None:
        """An empty title should be rejected before persistence."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "Task title cannot be empty",
        ):
            repository.create_task("   ")

        self.assertEqual(repository.list_tasks(), ())

    def test_rejects_naive_due_date(self) -> None:
        """Task due dates should include timezone information."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "must include timezone information",
        ):
            repository.create_task(
                "Réviser",
                due_at=datetime(2026, 8, 8, 10, 0),
            )

    def test_lists_no_pending_tasks(self) -> None:
        """An empty database should return an empty tuple."""
        repository = self._repository()

        self.assertEqual(repository.list_tasks(), ())

    def test_lists_tasks_in_creation_order(self) -> None:
        """Tasks should be listed in deterministic creation order."""
        repository = self._repository(
            self.created_at,
            self.second_created_at,
        )

        first_task = repository.create_task("Première")
        second_task = repository.create_task("Deuxième")

        self.assertEqual(
            repository.list_tasks(),
            (
                first_task,
                second_task,
            ),
        )

    def test_limits_list_results(self) -> None:
        """Task listing should respect its explicit result limit."""
        third_created_at = datetime(
            2026,
            8,
            7,
            4,
            0,
            tzinfo=timezone.utc,
        )
        repository = self._repository(
            self.created_at,
            self.second_created_at,
            third_created_at,
        )

        first_task = repository.create_task("Première")
        second_task = repository.create_task("Deuxième")
        repository.create_task("Troisième")

        self.assertEqual(
            repository.list_tasks(limit=2),
            (
                first_task,
                second_task,
            ),
        )

    def test_rejects_invalid_result_limit(self) -> None:
        """Task result limits should remain within safe bounds."""
        repository = self._repository()

        with self.assertRaisesRegex(
            ValueError,
            "between 1 and 100",
        ):
            repository.list_tasks(limit=0)

    def test_persists_between_repository_instances(self) -> None:
        """Tasks should remain available through another repository."""
        first_repository = self._repository(self.created_at)
        created_task = first_repository.create_task("Persistante")

        second_repository = self._repository()

        self.assertEqual(
            second_repository.list_tasks(),
            (created_task,),
        )

    def test_completes_existing_task(self) -> None:
        """Completing a task should persist and return its new state."""
        repository = self._repository(
            self.created_at,
            self.completed_at,
        )
        created_task = repository.create_task("Réviser")

        completed_task = repository.complete_task(created_task.id)

        self.assertEqual(completed_task.id, created_task.id)
        self.assertEqual(completed_task.status, "completed")
        self.assertEqual(
            completed_task.completed_at,
            self.completed_at,
        )
        self.assertEqual(repository.list_tasks(), ())

    def test_lists_completed_tasks_by_status(self) -> None:
        """Completed tasks should be available through status filtering."""
        repository = self._repository(
            self.created_at,
            self.completed_at,
        )
        task = repository.create_task("Réviser")
        completed_task = repository.complete_task(task.id)

        self.assertEqual(
            repository.list_tasks(status="completed"),
            (completed_task,),
        )
        self.assertEqual(
            repository.list_tasks(status=None),
            (completed_task,),
        )

    def test_rejects_missing_task(self) -> None:
        """Completing an unknown identifier should be explicit."""
        repository = self._repository(self.completed_at)

        with self.assertRaisesRegex(
            TaskNotFoundError,
            "Task 42 does not exist",
        ):
            repository.complete_task(42)

    def test_rejects_double_completion(self) -> None:
        """A completed task should not be completed twice."""
        first_repository = self._repository(
            self.created_at,
            self.completed_at,
        )
        task = first_repository.create_task("Réviser")
        first_repository.complete_task(task.id)

        later_time = datetime(
            2026,
            8,
            7,
            5,
            0,
            tzinfo=timezone.utc,
        )
        second_repository = self._repository(later_time)

        with self.assertRaisesRegex(
            TaskAlreadyCompletedError,
            "already completed",
        ):
            second_repository.complete_task(task.id)

    def test_rolls_back_completion_before_creation(self) -> None:
        """An incoherent completion time should not modify the task."""
        repository = self._repository(self.second_created_at)
        task = repository.create_task("Réviser")

        invalid_repository = self._repository(self.created_at)

        with self.assertRaisesRegex(
            RepositoryError,
            "cannot precede creation time",
        ):
            invalid_repository.complete_task(task.id)

        self.assertEqual(
            repository.list_tasks(),
            (task,),
        )

    def test_keeps_separate_databases_independent(self) -> None:
        """Repositories using different databases should remain isolated."""
        first_repository = self._repository(self.created_at)
        first_repository.create_task("Première base")

        second_database = SQLiteDatabase(
            self.root / "second.db"
        )
        second_database.initialize()
        second_repository = TaskRepository(
            second_database,
            clock=SequenceClock(),
        )

        self.assertEqual(second_repository.list_tasks(), ())


if __name__ == "__main__":
    unittest.main()
