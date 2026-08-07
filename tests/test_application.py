"""Tests for the complete assistant application assembly."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from assistant_ia.application import (
    ApplicationInitializationError,
    build_default_assistant,
)
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase


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

    def test_builds_assistant_with_seven_actions(self) -> None:
        """Application assembly should initialize all safe actions."""
        assistant = build_default_assistant(
            database=self.database,
            model_client=FakeModelClient([]),
        )

        self.assertTrue(self.database_path.exists())
        self.assertEqual(
            assistant.action_registry.action_count,
            7,
        )
        self.assertNotIn(
            "launch_application",
            assistant.action_registry.action_names,
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
