"""Tests for the complete assistant application assembly."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from assistant_ia.application import (
    ApplicationInitializationError,
    build_default_assistant,
)
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import (
    DEFAULT_DATABASE_DIRECTORY_NAME,
    DEFAULT_DATABASE_FILENAME,
    SQLiteDatabase,
)
from assistant_ia.system.windows import WindowsApplicationLauncher


class FakeModelClient:
    """Return predefined structured model responses."""

    def __init__(
        self,
        responses: list[ModelResponse],
    ) -> None:
        """Store the ordered fake responses."""
        self._responses = responses.copy()

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Return the next predefined response."""
        if not self._responses:
            raise AssertionError(
                "No fake model response remains."
            )

        return self._responses.pop(0)



class RecordingProcessLauncher:
    """Record process launches without starting real applications."""

    def __init__(self) -> None:
        """Create an empty process launch history."""
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: tuple[str, ...],
    ) -> object:
        """Record one internally resolved process command."""
        self.commands.append(command)
        return object()


class ApplicationAssemblyTests(unittest.TestCase):
    """Validate complete persistent assistant assembly."""

    def setUp(self) -> None:
        """Create an isolated database path for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name)
            / "assistant.db"
        )
        self.database = SQLiteDatabase(
            self.database_path
        )

    def tearDown(self) -> None:
        """Remove the isolated database directory."""
        self.temporary_directory.cleanup()

    def test_builds_assistant_with_eight_actions(self) -> None:
        """Application assembly should initialize available actions."""
        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient([]),
        )

        self.assertTrue(self.database_path.exists())
        self.assertEqual(
            assistant.action_registry.action_count,
            8,
        )
        self.assertIn(
            "launch_application",
            assistant.action_registry.action_names,
        )

    def test_builds_assistant_with_default_user_database(
        self,
    ) -> None:
        """Default assembly should use the user-local database path."""
        local_app_data = (
            Path(self.temporary_directory.name)
            / "local-app-data"
        )

        with patch.dict(
            os.environ,
            {
                "LOCALAPPDATA": str(local_app_data),
            },
            clear=False,
        ):
            assistant = build_default_assistant(
                model_client=FakeModelClient([]),
            )

        expected_database_path = (
            local_app_data
            / DEFAULT_DATABASE_DIRECTORY_NAME
            / DEFAULT_DATABASE_FILENAME
        )

        self.assertTrue(
            expected_database_path.is_file()
        )
        self.assertEqual(
            assistant.action_registry.action_count,
            8,
        )

    def test_system_action_fails_closed_without_confirmation(
        self,
    ) -> None:
        """System actions should not execute without confirmation."""
        process_launcher = RecordingProcessLauncher()

        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Fake model success.",
                        model="fake-model",
                        intent=Intent(
                            name="launch_application",
                            parameters={
                                "application": "notepad",
                            },
                        ),
                    )
                ]
            ),
            windows_launcher=WindowsApplicationLauncher(
                process_launcher=process_launcher
            ),
        )

        result = assistant.process_message(
            "Lance le bloc-notes."
        )

        self.assertEqual(
            result,
            "Lancement annulé : Bloc-notes.",
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_executes_confirmed_system_action_through_core(
        self,
    ) -> None:
        """Confirmed system actions should use the injected launcher."""
        process_launcher = RecordingProcessLauncher()

        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Fake model success.",
                        model="fake-model",
                        intent=Intent(
                            name="launch_application",
                            parameters={
                                "application": "bloc-notes",
                            },
                        ),
                    )
                ]
            ),
            confirmation_handler=lambda request: True,
            windows_launcher=WindowsApplicationLauncher(
                process_launcher=process_launcher
            ),
        )

        result = assistant.process_message(
            "Lance le bloc-notes."
        )

        self.assertEqual(
            result,
            "Application lancée : Bloc-notes.",
        )
        self.assertEqual(
            process_launcher.commands,
            [
                (
                    "notepad.exe",
                ),
            ],
        )
        self.assertNotEqual(
            result,
            "Fake model success.",
        )

    def test_executes_persistent_action_through_core(self) -> None:
        """A structured action should persist through the full stack."""
        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Le souvenir a été enregistré.",
                        model="fake-model",
                        intent=Intent(
                            name="save_memory",
                            parameters={
                                "content": (
                                    "Mon examen est le 24 août."
                                ),
                            },
                        ),
                    )
                ]
            ),
        )

        result = assistant.process_message(
            "Souviens-toi que mon examen est le 24 août."
        )

        self.assertEqual(
            result,
            "Souvenir enregistré : [#1] "
            "Mon examen est le 24 août.",
        )
        self.assertEqual(
            tuple(
                memory.content
                for memory in MemoryRepository(
                    self.database
                ).find_memories("examen")
            ),
            (
                "Mon examen est le 24 août.",
            ),
        )

    def test_reuses_persistence_between_assistants(self) -> None:
        """Separate assistant sessions should share one database."""
        first_assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Souvenir enregistré.",
                        model="fake-model",
                        intent=Intent(
                            name="save_memory",
                            parameters={
                                "content": "Information persistante.",
                            },
                        ),
                    )
                ]
            ),
        )
        first_assistant.process_message(
            "Mémorise cette information."
        )

        second_assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Recherche effectuée.",
                        model="fake-model",
                        intent=Intent(
                            name="find_memory",
                            parameters={
                                "query": "persistante",
                            },
                        ),
                    )
                ]
            ),
        )

        result = second_assistant.process_message(
            "Retrouve mon information persistante."
        )

        self.assertEqual(
            result,
            "Souvenirs trouvés :\n"
            "• [#1] Information persistante.",
        )

    def test_passes_current_date_to_journal_action(self) -> None:
        """Application assembly should inject the journal date provider."""
        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient(
                [
                    ModelResponse(
                        content="Entrée enregistrée.",
                        model="fake-model",
                        intent=Intent(
                            name="write_journal",
                            parameters={
                                "content": "Journée productive.",
                            },
                        ),
                    )
                ]
            ),
            current_date=lambda: date(
                2026,
                8,
                7,
            ),
        )

        result = assistant.process_message(
            "Écris dans mon journal."
        )

        self.assertEqual(
            result,
            "Entrée de journal enregistrée pour le "
            "2026-08-07 : [#1] Journée productive.",
        )

    def test_wraps_database_initialization_failure(self) -> None:
        """Malformed databases should fail before assistant startup."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY
                )
                """
            )

        with self.assertRaisesRegex(
            ApplicationInitializationError,
            "database could not be initialized",
        ):
            build_default_assistant(
                database=self.database,
                model_client=FakeModelClient([]),
            )


if __name__ == "__main__":
    unittest.main()
