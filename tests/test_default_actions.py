"""Tests for the default persistent assistant actions."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from assistant_ia.actions.action import ActionValidationError
from assistant_ia.actions.defaults import (
    build_default_action_registry,
)
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.intelligence.intent import Intent
from assistant_ia.memory.journal_repository import JournalRepository
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.task_repository import TaskRepository


class SequenceClock:
    """Return deterministic datetimes in a predefined order."""

    def __init__(self, *values: datetime) -> None:
        """Store the ordered datetime values."""
        self._values = list(values)

    def __call__(self) -> datetime:
        """Return the next deterministic datetime."""
        if not self._values:
            raise AssertionError(
                "No deterministic clock value remains."
            )

        return self._values.pop(0)


class SequenceDateProvider:
    """Return deterministic dates in a predefined order."""

    def __init__(self, *values: date) -> None:
        """Store the ordered date values."""
        self._values = list(values)

    def __call__(self) -> date:
        """Return the next deterministic date."""
        if not self._values:
            raise AssertionError(
                "No deterministic date value remains."
            )

        return self._values.pop(0)


class DefaultActionTests(unittest.TestCase):
    """Validate the seven persistent default actions."""

    def setUp(self) -> None:
        """Create an isolated initialized database."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()

        self.first_time = datetime(
            2026,
            8,
            7,
            8,
            0,
            tzinfo=timezone.utc,
        )
        self.second_time = datetime(
            2026,
            8,
            7,
            9,
            0,
            tzinfo=timezone.utc,
        )
        self.third_time = datetime(
            2026,
            8,
            7,
            10,
            0,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        """Remove the temporary database."""
        self.temporary_directory.cleanup()

    def _registry(
        self,
        *,
        task_times: tuple[datetime, ...] = (),
        memory_times: tuple[datetime, ...] = (),
        journal_times: tuple[datetime, ...] = (),
        current_dates: tuple[date, ...] = (),
    ) -> ActionRegistry:
        """Build a registry using deterministic repositories."""
        return build_default_action_registry(
            task_repository=TaskRepository(
                self.database,
                clock=SequenceClock(*task_times),
            ),
            memory_repository=MemoryRepository(
                self.database,
                clock=SequenceClock(*memory_times),
            ),
            journal_repository=JournalRepository(
                self.database,
                clock=SequenceClock(*journal_times),
            ),
            current_date=SequenceDateProvider(
                *current_dates
            ),
        )

    def test_registers_seven_default_actions(self) -> None:
        """The default registry should expose all safe actions."""
        registry = self._registry()

        self.assertEqual(
            registry.action_count,
            7,
        )
        self.assertEqual(
            registry.action_names,
            frozenset(
                {
                    "create_task",
                    "list_tasks",
                    "complete_task",
                    "save_memory",
                    "find_memory",
                    "delete_memory",
                    "write_journal",
                }
            ),
        )

    def test_creates_task_without_due_date(self) -> None:
        """Task creation should persist a task without a deadline."""
        registry = self._registry(
            task_times=(
                self.first_time,
            ),
        )

        result = registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Réviser la biologie",
                },
            )
        )

        self.assertEqual(
            result,
            "Tâche créée : [#1] Réviser la biologie.",
        )

        tasks = TaskRepository(
            self.database
        ).list_tasks()

        self.assertEqual(
            len(tasks),
            1,
        )
        self.assertIsNone(
            tasks[0].due_at
        )

    def test_creates_task_with_timezone_aware_due_date(
        self,
    ) -> None:
        """A valid deadline should be normalized and persisted."""
        registry = self._registry(
            task_times=(
                self.first_time,
            ),
        )

        result = registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Appeler le médecin",
                    "due_at": (
                        "2026-08-08T10:00:00+02:00"
                    ),
                },
            )
        )

        self.assertEqual(
            result,
            "Tâche créée : [#1] Appeler le médecin "
            "— échéance : 08.08.2026 à 08:00 UTC.",
        )

    def test_rejects_ambiguous_task_due_date(self) -> None:
        """Relative dates should not be converted silently."""
        registry = self._registry()

        with self.assertRaisesRegex(
            ActionValidationError,
            "ISO 8601",
        ):
            registry.execute(
                Intent(
                    name="create_task",
                    parameters={
                        "title": "Réviser",
                        "due_at": "demain",
                    },
                )
            )

        self.assertEqual(
            TaskRepository(
                self.database
            ).list_tasks(),
            (),
        )

    def test_lists_empty_pending_tasks(self) -> None:
        """An empty pending list should produce an explicit response."""
        registry = self._registry()

        result = registry.execute(
            Intent(
                name="list_tasks",
                parameters={},
            )
        )

        self.assertEqual(
            result,
            "Aucune tâche en attente.",
        )

    def test_lists_and_completes_tasks(self) -> None:
        """Task actions should share the same persistent state."""
        registry = self._registry(
            task_times=(
                self.first_time,
                self.second_time,
                self.third_time,
            ),
        )

        registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Première tâche",
                },
            )
        )
        registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Deuxième tâche",
                },
            )
        )

        pending_result = registry.execute(
            Intent(
                name="list_tasks",
                parameters={
                    "status": "pending",
                },
            )
        )

        self.assertEqual(
            pending_result,
            "Tâches en attente :\n"
            "• [#1] Première tâche\n"
            "• [#2] Deuxième tâche",
        )

        completed_result = registry.execute(
            Intent(
                name="complete_task",
                parameters={
                    "task_id": "1",
                },
            )
        )

        self.assertEqual(
            completed_result,
            "Tâche terminée : [#1] Première tâche.",
        )

        list_result = registry.execute(
            Intent(
                name="list_tasks",
                parameters={
                    "status": "completed",
                },
            )
        )

        self.assertEqual(
            list_result,
            "Tâches terminées :\n"
            "✓ [#1] Première tâche",
        )

    def test_rejects_missing_task(self) -> None:
        """Completing an unknown task should be explicit."""
        registry = self._registry(
            task_times=(
                self.first_time,
            ),
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "tâche #42 n’existe pas",
        ):
            registry.execute(
                Intent(
                    name="complete_task",
                    parameters={
                        "task_id": "42",
                    },
                )
            )

    def test_rejects_already_completed_task(self) -> None:
        """A task should not be completed twice."""
        registry = self._registry(
            task_times=(
                self.first_time,
                self.second_time,
                self.third_time,
            ),
        )

        registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Tâche unique",
                },
            )
        )
        registry.execute(
            Intent(
                name="complete_task",
                parameters={
                    "task_id": "1",
                },
            )
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "déjà terminée",
        ):
            registry.execute(
                Intent(
                    name="complete_task",
                    parameters={
                        "task_id": "1",
                    },
                )
            )

    def test_rejects_invalid_task_status(self) -> None:
        """Task status filters should use the authorized values."""
        registry = self._registry()

        with self.assertRaisesRegex(
            ActionValidationError,
            "pending, completed ou all",
        ):
            registry.execute(
                Intent(
                    name="list_tasks",
                    parameters={
                        "status": "urgent",
                    },
                )
            )

    def test_saves_and_finds_memories(self) -> None:
        """Memory actions should persist and find literal content."""
        registry = self._registry(
            memory_times=(
                self.first_time,
                self.second_time,
            ),
        )

        first_result = registry.execute(
            Intent(
                name="save_memory",
                parameters={
                    "content": "Mon examen est le 24 août.",
                },
            )
        )
        registry.execute(
            Intent(
                name="save_memory",
                parameters={
                    "content": "Acheter du pain.",
                },
            )
        )

        self.assertEqual(
            first_result,
            "Souvenir enregistré : [#1] "
            "Mon examen est le 24 août.",
        )

        find_result = registry.execute(
            Intent(
                name="find_memory",
                parameters={
                    "query": "examen",
                },
            )
        )

        self.assertEqual(
            find_result,
            "Souvenirs trouvés :\n"
            "• [#1] Mon examen est le 24 août.",
        )

    def test_reports_missing_memory_search_result(self) -> None:
        """A memory search without a match should be explicit."""
        registry = self._registry()

        result = registry.execute(
            Intent(
                name="find_memory",
                parameters={
                    "query": "introuvable",
                },
            )
        )

        self.assertEqual(
            result,
            "Aucun souvenir trouvé.",
        )

    def test_deletes_memory_and_rejects_second_deletion(
        self,
    ) -> None:
        """Memory deletion should use one stable identifier."""
        registry = self._registry(
            memory_times=(
                self.first_time,
            ),
        )

        registry.execute(
            Intent(
                name="save_memory",
                parameters={
                    "content": "Souvenir temporaire.",
                },
            )
        )

        delete_result = registry.execute(
            Intent(
                name="delete_memory",
                parameters={
                    "memory_id": "1",
                },
            )
        )

        self.assertEqual(
            delete_result,
            "Souvenir supprimé : [#1] "
            "Souvenir temporaire.",
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "souvenir #1 n’existe pas",
        ):
            registry.execute(
                Intent(
                    name="delete_memory",
                    parameters={
                        "memory_id": "1",
                    },
                )
            )

    def test_accepts_safe_task_identifier_notation(self) -> None:
        """Task identifiers may use a harmless visible marker."""
        registry = self._registry(
            task_times=(
                self.first_time,
                self.second_time,
            ),
        )

        registry.execute(
            Intent(
                name="create_task",
                parameters={
                    "title": "Tâche identifiée",
                },
            )
        )

        result = registry.execute(
            Intent(
                name="complete_task",
                parameters={
                    "task_id": "#1.",
                },
            )
        )

        self.assertEqual(
            result,
            "Tâche terminée : [#1] Tâche identifiée.",
        )

    def test_accepts_safe_memory_identifier_notation(
        self,
    ) -> None:
        """Memory identifiers may include the French number label."""
        registry = self._registry(
            memory_times=(
                self.first_time,
            ),
        )

        registry.execute(
            Intent(
                name="save_memory",
                parameters={
                    "content": "Souvenir identifié.",
                },
            )
        )

        result = registry.execute(
            Intent(
                name="delete_memory",
                parameters={
                    "memory_id": "numéro 1",
                },
            )
        )

        self.assertEqual(
            result,
            "Souvenir supprimé : [#1] "
            "Souvenir identifié.",
        )

    def test_rejects_invalid_identifier_text(self) -> None:
        """Identifiers should reject unsafe nonnumeric content."""
        registry = self._registry()

        with self.assertRaisesRegex(
            ActionValidationError,
            "entier positif",
        ):
            registry.execute(
                Intent(
                    name="complete_task",
                    parameters={
                        "task_id": "1 OR 1=1",
                    },
                )
            )

    def test_writes_journal_using_current_date(self) -> None:
        """Journal entries without a date should use local today."""
        registry = self._registry(
            journal_times=(
                self.first_time,
            ),
            current_dates=(
                date(2026, 8, 7),
            ),
        )

        result = registry.execute(
            Intent(
                name="write_journal",
                parameters={
                    "content": "Journée productive.",
                },
            )
        )

        self.assertEqual(
            result,
            "Entrée de journal enregistrée pour le "
            "2026-08-07 : [#1] Journée productive.",
        )

    def test_writes_journal_using_explicit_date(self) -> None:
        """An explicit journal date should be preserved."""
        registry = self._registry(
            journal_times=(
                self.first_time,
            ),
        )

        result = registry.execute(
            Intent(
                name="write_journal",
                parameters={
                    "content": "Entrée rétrospective.",
                    "entry_date": "2026-08-06",
                },
            )
        )

        self.assertEqual(
            result,
            "Entrée de journal enregistrée pour le "
            "2026-08-06 : [#1] Entrée rétrospective.",
        )

    def test_rejects_invalid_journal_date(self) -> None:
        """Journal dates should use strict ISO date syntax."""
        registry = self._registry()

        with self.assertRaisesRegex(
            ActionValidationError,
            "YYYY-MM-DD",
        ):
            registry.execute(
                Intent(
                    name="write_journal",
                    parameters={
                        "content": "Entrée.",
                        "entry_date": "demain",
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
