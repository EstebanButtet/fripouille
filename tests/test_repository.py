"""Tests for the SQLite persistence infrastructure."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant_ia.memory import repository as repository_module
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_DATABASE_DIRECTORY_NAME,
    DEFAULT_DATABASE_FILENAME,
    DatabaseError,
    SQLiteDatabase,
    default_database_path,
)
from assistant_ia.memory.retrieval import ContextualMemoryRetriever
from assistant_ia.people.defaults import (
    DEFAULT_PERSON_ID,
    DEFAULT_PERSON_NAME,
)


class SQLiteDatabasePathTests(unittest.TestCase):
    """Validate SQLite database path handling."""

    def test_normalizes_explicit_path(self) -> None:
        """Explicit paths should become normalized absolute paths."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            supplied_path = (
                root
                / "temporary_directory"
                / ".."
                / "assistant.db"
            )

            database = SQLiteDatabase(supplied_path)

            self.assertEqual(
                database.path,
                (root / "assistant.db").resolve(),
            )
            self.assertTrue(database.path.is_absolute())

    def test_construction_does_not_create_database(self) -> None:
        """Path validation alone should not write to the filesystem."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "data" / "assistant.db"

            database = SQLiteDatabase(database_path)

            self.assertEqual(database.path, database_path.resolve())
            self.assertFalse(database_path.parent.exists())
            self.assertFalse(database_path.exists())

    def test_rejects_empty_path(self) -> None:
        """Empty path strings should be rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Database path cannot be empty",
        ):
            SQLiteDatabase("   ")

    def test_rejects_non_path_value(self) -> None:
        """Values that are not strings or path-like should be rejected."""
        with self.assertRaisesRegex(
            TypeError,
            "string or path-like object",
        ):
            SQLiteDatabase(123)

    def test_rejects_existing_directory(self) -> None:
        """An existing directory cannot be used as a database file."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "must reference a file",
            ):
                SQLiteDatabase(temporary_directory)

    def test_builds_default_path_from_local_app_data(self) -> None:
        """The default path should use the Windows local data directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            with patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(root)},
                clear=False,
            ):
                result = default_database_path()

            self.assertEqual(
                result,
                (
                    root
                    / DEFAULT_DATABASE_DIRECTORY_NAME
                    / DEFAULT_DATABASE_FILENAME
                ).resolve(),
            )
            self.assertFalse(result.exists())

    def test_rejects_missing_local_app_data(self) -> None:
        """A missing Windows local data directory should be explicit."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                DatabaseError,
                "LOCALAPPDATA is not configured",
            ):
                default_database_path()


class SQLiteDatabaseConnectionTests(unittest.TestCase):
    """Validate SQLite connection and transaction management."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        """Remove the isolated temporary directory."""
        self.temporary_directory.cleanup()

    def test_creates_parent_directory_and_database_file(self) -> None:
        """Connecting should create the parent directory and database."""
        database_path = self.root / "data" / "assistant.db"
        database = SQLiteDatabase(database_path)

        with database.connect() as connection:
            self.assertEqual(
                connection.execute("SELECT 1").fetchone(),
                (1,),
            )

        self.assertTrue(database_path.parent.is_dir())
        self.assertTrue(database_path.is_file())

    def test_enables_foreign_keys_for_each_connection(self) -> None:
        """Every opened connection should enforce foreign keys."""
        database = SQLiteDatabase(self.root / "assistant.db")

        with database.connect() as connection:
            foreign_keys_status = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()

        self.assertEqual(foreign_keys_status, (1,))

    def test_closes_connection_after_context(self) -> None:
        """Leaving the context should close the SQLite connection."""
        database = SQLiteDatabase(self.root / "assistant.db")

        with database.connect() as connection:
            connection.execute("SELECT 1")

        with self.assertRaisesRegex(
            sqlite3.ProgrammingError,
            "closed database",
        ):
            connection.execute("SELECT 1")

    def test_commits_successful_transaction(self) -> None:
        """A successful connection context should commit its changes."""
        database = SQLiteDatabase(self.root / "assistant.db")

        with database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE example_items (
                    id INTEGER PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO example_items (value) VALUES (?)",
                ("saved",),
            )

        with database.connect() as connection:
            result = connection.execute(
                "SELECT value FROM example_items"
            ).fetchone()

        self.assertEqual(result, ("saved",))

    def test_rolls_back_transaction_on_application_error(self) -> None:
        """Application errors should roll back pending database changes."""
        database = SQLiteDatabase(self.root / "assistant.db")

        with database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE example_items (
                    id INTEGER PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

        with self.assertRaisesRegex(RuntimeError, "Stop transaction"):
            with database.connect() as connection:
                connection.execute(
                    "INSERT INTO example_items (value) VALUES (?)",
                    ("not saved",),
                )
                raise RuntimeError("Stop transaction")

        with database.connect() as connection:
            result = connection.execute(
                "SELECT COUNT(*) FROM example_items"
            ).fetchone()

        self.assertEqual(result, (0,))

    def test_converts_sqlite_error(self) -> None:
        """SQLite failures should become application database errors."""
        database = SQLiteDatabase(self.root / "assistant.db")

        with self.assertRaisesRegex(
            DatabaseError,
            "SQLite database operation failed",
        ) as raised_error:
            with database.connect() as connection:
                connection.execute(
                    "SELECT * FROM missing_table"
                )

        self.assertIsInstance(
            raised_error.exception.__cause__,
            sqlite3.Error,
        )

    def test_keeps_separate_databases_independent(self) -> None:
        """Different paths should never share persisted state."""
        first_database = SQLiteDatabase(self.root / "first.db")
        second_database = SQLiteDatabase(self.root / "second.db")

        with first_database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE first_only (
                    id INTEGER PRIMARY KEY
                )
                """
            )

        with second_database.connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = ?
                  AND name = ?
                """,
                ("table", "first_only"),
            ).fetchone()

        self.assertEqual(result, (0,))

    def test_converts_parent_directory_creation_error(self) -> None:
        """Filesystem failures should become application database errors."""
        blocking_file = self.root / "blocking_file"
        blocking_file.write_text("blocked", encoding="utf-8")
        database = SQLiteDatabase(
            blocking_file / "assistant.db"
        )

        with self.assertRaisesRegex(
            DatabaseError,
            "Database directory could not be created",
        ) as raised_error:
            with database.connect():
                self.fail("The connection context should not be entered.")

        self.assertIsInstance(
            raised_error.exception.__cause__,
            OSError,
        )


class SQLiteDatabaseInitializationTests(unittest.TestCase):
    """Validate schema initialization, migration and versioning."""

    def setUp(self) -> None:
        """Create a separate temporary database path for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name)
            / "assistant.db"
        )
        self.database = SQLiteDatabase(self.database_path)

    def tearDown(self) -> None:
        """Remove the temporary database and its directory."""
        self.temporary_directory.cleanup()

    def _create_version_one_schema(self) -> None:
        """Create the technical schema used before business migrations."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL CHECK (version >= 1)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO schema_version (id, version)
                VALUES (?, ?)
                """,
                (1, 1),
            )

    def _create_version_two_schema(
        self,
        memories: tuple[tuple[int, str, str], ...] = (),
    ) -> None:
        """Create the exact schema used before memory provenance."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL CHECK (version >= 1)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO schema_version (id, version)
                VALUES (?, ?)
                """,
                (1, 2),
            )
            connection.execute(
                """
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    due_at TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL
                        CHECK (length(trim(content)) > 0),
                    created_at TEXT NOT NULL
                        CHECK (length(trim(created_at)) > 0)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE journal_entries (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO memories (
                    id,
                    content,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                memories,
            )

    def _create_version_three_schema(
        self,
        *,
        tasks: tuple[tuple[object, ...], ...] = (),
        memories: tuple[tuple[int, str, str], ...] = (),
        journal_entries: tuple[tuple[object, ...], ...] = (),
    ) -> None:
        """Create a v3 database populated through the real v2 migration."""
        self._create_version_two_schema(memories)

        with self.database.connect() as connection:
            repository_module._migrate_schema_2_to_3(connection)
            connection.execute(
                "UPDATE schema_version SET version = 3 WHERE id = 1"
            )
            connection.executemany(
                """
                INSERT INTO tasks (
                    id,
                    title,
                    due_at,
                    status,
                    created_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                tasks,
            )
            connection.executemany(
                """
                INSERT INTO journal_entries (
                    id,
                    content,
                    entry_date,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                journal_entries,
            )

    def _create_version_four_schema(
        self,
        *,
        tasks: tuple[tuple[object, ...], ...] = (),
        memories: tuple[tuple[int, str, str], ...] = (),
        journal_entries: tuple[tuple[object, ...], ...] = (),
    ) -> None:
        """Create a populated v4 database through the real person migration."""
        self._create_version_three_schema(
            tasks=tasks,
            memories=memories,
            journal_entries=journal_entries,
        )
        with self.database.connect() as connection:
            repository_module._migrate_schema_3_to_4(connection)
            connection.execute(
                "UPDATE schema_version SET version = 4 WHERE id = 1"
            )
            connection.execute(
                """
                INSERT INTO persons (id, display_name, created_at)
                VALUES (?, ?, ?)
                """,
                (7, "Alice", "2026-08-07T04:00:00+00:00"),
            )

    def _create_version_five_schema(
        self,
        *,
        tasks: tuple[tuple[object, ...], ...] = (),
        memories: tuple[tuple[int, str, str], ...] = (),
        journal_entries: tuple[tuple[object, ...], ...] = (),
    ) -> None:
        """Create a populated v5 database through the real profile migration."""
        self._create_version_four_schema(
            tasks=tasks,
            memories=memories,
            journal_entries=journal_entries,
        )
        with self.database.connect() as connection:
            repository_module._migrate_schema_4_to_5(connection)
            connection.execute(
                "UPDATE schema_version SET version = 5 WHERE id = 1"
            )
            connection.execute(
                """
                INSERT INTO profile_facts (
                    id, person_id, category, content, source, source_text,
                    confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    8,
                    7,
                    "interest",
                    "J'aime la robotique.",
                    "explicit_user",
                    None,
                    1.0,
                    "2026-08-07T05:00:00+00:00",
                    "2026-08-07T05:00:00+00:00",
                ),
            )

    def _table_names(self) -> list[tuple[str]]:
        """Return all non-internal table names in deterministic order."""
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = ?
                  AND name NOT LIKE ?
                ORDER BY name
                """,
                ("table", "sqlite_%"),
            ).fetchall()

    def _table_exists(self, table_name: str) -> bool:
        """Return whether one table exists in the temporary database."""
        with self.database.connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = ?
                  AND name = ?
                """,
                ("table", table_name),
            ).fetchone()

        return result == (1,)

    def test_initialize_creates_current_schema_version(self) -> None:
        """A fresh database should directly reach the current version."""
        self.database.initialize()

        with self.database.connect() as connection:
            version_row = connection.execute(
                """
                SELECT id, version
                FROM schema_version
                """
            ).fetchone()

        self.assertEqual(
            version_row,
            (1, CURRENT_SCHEMA_VERSION),
        )

    def test_initialize_creates_business_tables(self) -> None:
        """A fresh database should contain all required business tables."""
        self.database.initialize()

        self.assertEqual(
            self._table_names(),
            [
                ("journal_entries",),
                ("memories",),
                ("memory_people",),
                ("persons",),
                ("profile_facts",),
                ("schema_version",),
                ("tasks",),
            ],
        )

    def test_migrates_version_one_database(self) -> None:
        """A version one database should receive the business tables."""
        self._create_version_one_schema()

        self.database.initialize()

        self.assertTrue(self._table_exists("tasks"))
        self.assertTrue(self._table_exists("memories"))
        self.assertTrue(self._table_exists("journal_entries"))

    def test_migrates_empty_version_two_database(self) -> None:
        """An empty v2 memory table should reach the current schema."""
        self._create_version_two_schema()

        self.database.initialize()

        with self.database.connect() as connection:
            version = connection.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            columns = connection.execute(
                """
                SELECT name
                FROM pragma_table_info('memories')
                ORDER BY cid
                """
            ).fetchall()
            count = connection.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()

        self.assertEqual(version, (CURRENT_SCHEMA_VERSION,))
        self.assertEqual(
            columns,
            [
                ("id",),
                ("content",),
                ("source",),
                ("source_text",),
                ("confidence",),
                ("created_at",),
                ("updated_at",),
            ],
        )
        self.assertEqual(count, (0,))

    def test_migrates_version_two_memories_without_inventing_evidence(
        self,
    ) -> None:
        """V2 rows should preserve data and receive honest provenance."""
        memories = (
            (
                4,
                "  Premier contenu exact.  ",
                "2026-08-07T01:00:00+00:00",
            ),
            (
                9,
                "Deuxième contenu exact.",
                "2026-08-08T02:30:00+00:00",
            ),
        )
        self._create_version_two_schema(memories)

        self.database.initialize()

        with self.database.connect() as connection:
            migrated_rows = connection.execute(
                """
                SELECT
                    id,
                    content,
                    source,
                    source_text,
                    confidence,
                    created_at,
                    updated_at
                FROM memories
                ORDER BY id
                """
            ).fetchall()

        self.assertEqual(
            migrated_rows,
            [
                (
                    4,
                    "  Premier contenu exact.  ",
                    "legacy_explicit",
                    None,
                    1.0,
                    "2026-08-07T01:00:00+00:00",
                    "2026-08-07T01:00:00+00:00",
                ),
                (
                    9,
                    "Deuxième contenu exact.",
                    "legacy_explicit",
                    None,
                    1.0,
                    "2026-08-08T02:30:00+00:00",
                    "2026-08-08T02:30:00+00:00",
                ),
            ],
        )

    def test_version_two_migration_rolls_back_completely(
        self,
    ) -> None:
        """A v3 migration failure should restore version and v2 rows."""
        original_row = (
            7,
            "Souvenir préservé.",
            "2026-08-07T01:00:00+00:00",
        )
        self._create_version_two_schema((original_row,))
        migrate_to_v3 = repository_module._SCHEMA_MIGRATIONS[2]

        def failing_migration(
            connection: sqlite3.Connection,
        ) -> None:
            migrate_to_v3(connection)
            connection.execute(
                "SELECT * FROM table_that_does_not_exist"
            )

        with (
            patch.dict(
                repository_module._SCHEMA_MIGRATIONS,
                {
                    2: failing_migration,
                },
            ),
            self.assertRaisesRegex(
                DatabaseError,
                "SQLite database operation failed",
            ),
        ):
            self.database.initialize()

        with self.database.connect() as connection:
            version = connection.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            columns = connection.execute(
                """
                SELECT name
                FROM pragma_table_info('memories')
                ORDER BY cid
                """
            ).fetchall()
            rows = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                """
            ).fetchall()
            migrated_table = connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'memories_v2'
                """
            ).fetchone()

        self.assertEqual(version, (2,))
        self.assertEqual(
            columns,
            [
                ("id",),
                ("content",),
                ("created_at",),
            ],
        )
        self.assertEqual(rows, [original_row])
        self.assertEqual(migrated_table, (0,))

    def test_migration_updates_schema_version(self) -> None:
        """A successful migration should update its metadata version."""
        self._create_version_one_schema()

        self.database.initialize()

        with self.database.connect() as connection:
            version_row = connection.execute(
                """
                SELECT version
                FROM schema_version
                WHERE id = ?
                """,
                (1,),
            ).fetchone()

        self.assertEqual(
            version_row,
            (CURRENT_SCHEMA_VERSION,),
        )

    def test_migrates_v3_to_v4_without_losing_existing_data(self) -> None:
        """The person registry migration should preserve every v3 domain."""
        task = (
            3,
            "Tâche existante",
            None,
            "pending",
            "2026-08-07T01:00:00+00:00",
            None,
        )
        memory = (
            4,
            "Souvenir existant.",
            "2026-08-07T02:00:00+00:00",
        )
        journal_entry = (
            5,
            "Entrée existante.",
            "2026-08-07",
            "2026-08-07T03:00:00+00:00",
        )
        self._create_version_three_schema(
            tasks=(task,),
            memories=(memory,),
            journal_entries=(journal_entry,),
        )

        self.database.initialize()

        with self.database.connect() as connection:
            version = connection.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            stored_task = connection.execute(
                "SELECT * FROM tasks WHERE id = 3"
            ).fetchone()
            stored_memory = connection.execute(
                """
                SELECT
                    id,
                    content,
                    source,
                    source_text,
                    confidence,
                    created_at,
                    updated_at
                FROM memories
                WHERE id = 4
                """
            ).fetchone()
            stored_journal_entry = connection.execute(
                "SELECT * FROM journal_entries WHERE id = 5"
            ).fetchone()
            default_person = connection.execute(
                """
                SELECT id, display_name
                FROM persons
                WHERE id = ?
                """,
                (DEFAULT_PERSON_ID,),
            ).fetchone()

        self.assertEqual(version, (CURRENT_SCHEMA_VERSION,))
        self.assertEqual(stored_task, task)
        self.assertEqual(
            stored_memory,
            (
                memory[0],
                memory[1],
                "legacy_explicit",
                None,
                1.0,
                memory[2],
                memory[2],
            ),
        )
        self.assertEqual(stored_journal_entry, journal_entry)
        self.assertEqual(
            default_person,
            (DEFAULT_PERSON_ID, DEFAULT_PERSON_NAME),
        )

    def test_initialize_is_idempotent(self) -> None:
        """Repeated initialization should not alter a current schema."""
        self.database.initialize()
        restarted_database = SQLiteDatabase(self.database_path)
        restarted_database.initialize()

        with restarted_database.connect() as connection:
            version_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM schema_version
                """
            ).fetchone()

        self.assertEqual(version_count, (1,))
        self.assertEqual(
            self._table_names(),
            [
                ("journal_entries",),
                ("memories",),
                ("memory_people",),
                ("persons",),
                ("profile_facts",),
                ("schema_version",),
                ("tasks",),
            ],
        )

    def test_migrates_v4_to_v5_without_losing_existing_data(self) -> None:
        """Profile facts migration should preserve every existing v4 domain."""
        task = (
            3, "Tâche v4", None, "pending",
            "2026-08-07T01:00:00+00:00", None,
        )
        memory = (4, "Souvenir v4.", "2026-08-07T02:00:00+00:00")
        journal = (
            5, "Journal v4.", "2026-08-07",
            "2026-08-07T03:00:00+00:00",
        )
        self._create_version_four_schema(
            tasks=(task,), memories=(memory,), journal_entries=(journal,)
        )

        self.database.initialize()

        with self.database.connect() as connection:
            version = connection.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            stored_task = connection.execute(
                "SELECT * FROM tasks WHERE id = 3"
            ).fetchone()
            stored_memory = connection.execute(
                """
                SELECT id, content, source, source_text, confidence,
                       created_at, updated_at
                FROM memories
                WHERE id = 4
                """
            ).fetchone()
            memory_columns = connection.execute(
                "SELECT name FROM pragma_table_info('memories') ORDER BY cid"
            ).fetchall()
            stored_journal = connection.execute(
                "SELECT content FROM journal_entries WHERE id = 5"
            ).fetchone()
            stored_person = connection.execute(
                "SELECT display_name FROM persons WHERE id = 7"
            ).fetchone()
            facts = connection.execute(
                "SELECT COUNT(*) FROM profile_facts"
            ).fetchone()

        self.assertEqual(version, (CURRENT_SCHEMA_VERSION,))
        self.assertEqual(stored_task, task)
        self.assertEqual(
            stored_memory,
            (
                memory[0], memory[1], "legacy_explicit", None, 1.0,
                memory[2], memory[2],
            ),
        )
        self.assertEqual(
            memory_columns,
            [
                ("id",), ("content",), ("source",), ("source_text",),
                ("confidence",), ("created_at",), ("updated_at",),
            ],
        )
        self.assertEqual(stored_journal, (journal[1],))
        self.assertEqual(stored_person, ("Alice",))
        self.assertEqual(facts, (0,))

    def test_profile_facts_schema_has_person_foreign_key_and_index(self) -> None:
        """The v5 table should enforce its subject and expose its scoped index."""
        self.database.initialize()

        with self.database.connect() as connection:
            foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(profile_facts)"
            ).fetchall()
            indexes = connection.execute(
                "PRAGMA index_list(profile_facts)"
            ).fetchall()

        self.assertTrue(
            any(
                row[2] == "persons" and row[3] == "person_id"
                for row in foreign_keys
            )
        )
        self.assertIn(
            "idx_profile_facts_person_category",
            {row[1] for row in indexes},
        )

    def test_migrates_v5_to_v6_without_assigning_historical_memories(
        self,
    ) -> None:
        """Existing memories stay unassigned while every v5 domain survives."""
        task = (
            3, "Tâche v5", None, "pending",
            "2026-08-07T01:00:00+00:00", None,
        )
        memory = (4, "Souvenir historique.", "2026-08-07T02:00:00+00:00")
        journal = (
            5, "Journal v5.", "2026-08-07",
            "2026-08-07T03:00:00+00:00",
        )
        self._create_version_five_schema(
            tasks=(task,),
            memories=(memory,),
            journal_entries=(journal,),
        )

        self.database.initialize()

        with self.database.connect() as connection:
            version = connection.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            stored_task = connection.execute(
                "SELECT title FROM tasks WHERE id = 3"
            ).fetchone()
            stored_memory = connection.execute(
                "SELECT content FROM memories WHERE id = 4"
            ).fetchone()
            stored_journal = connection.execute(
                "SELECT content FROM journal_entries WHERE id = 5"
            ).fetchone()
            stored_person = connection.execute(
                "SELECT display_name FROM persons WHERE id = 7"
            ).fetchone()
            stored_profile_fact = connection.execute(
                "SELECT content FROM profile_facts WHERE id = 8"
            ).fetchone()
            links = connection.execute(
                "SELECT * FROM memory_people"
            ).fetchall()

        self.assertEqual(version, (CURRENT_SCHEMA_VERSION,))
        self.assertEqual(stored_task, (task[1],))
        self.assertEqual(stored_memory, (memory[1],))
        self.assertEqual(stored_journal, (journal[1],))
        self.assertEqual(stored_person, ("Alice",))
        self.assertEqual(stored_profile_fact, ("J'aime la robotique.",))
        self.assertEqual(links, [])
        memory_repository = MemoryRepository(self.database)
        self.assertEqual(
            tuple(
                item.id
                for item in memory_repository.list_unassigned_memories()
            ),
            (memory[0],),
        )
        retrieved = ContextualMemoryRetriever(memory_repository).retrieve(
            "souvenir historique"
        )
        self.assertEqual(
            tuple(item.memory.id for item in retrieved),
            (memory[0],),
        )

    def test_initialize_does_not_duplicate_default_person(self) -> None:
        """Repeated storage initialization should retain one default row."""
        self.database.initialize()

        with self.database.connect() as connection:
            initial_row = connection.execute(
                """
                SELECT id, display_name, created_at
                FROM persons
                WHERE id = ?
                """,
                (DEFAULT_PERSON_ID,),
            ).fetchone()

        restarted_database = SQLiteDatabase(self.database_path)
        restarted_database.initialize()

        with restarted_database.connect() as connection:
            persons = connection.execute(
                """
                SELECT id, display_name, created_at
                FROM persons
                ORDER BY id
                """
            ).fetchall()

        self.assertIsNotNone(initial_row)
        self.assertEqual(persons, [initial_row])

    def test_migration_preserves_existing_data(self) -> None:
        """Migrating should not remove unrelated existing data."""
        self._create_version_one_schema()

        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE legacy_items (
                    id INTEGER PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO legacy_items (value)
                VALUES (?)
                """,
                ("preserved",),
            )

        self.database.initialize()

        with self.database.connect() as connection:
            preserved_row = connection.execute(
                """
                SELECT value
                FROM legacy_items
                """
            ).fetchone()

        self.assertEqual(preserved_row, ("preserved",))

    def test_migration_rolls_back_on_error(self) -> None:
        """A failed migration should roll back all of its changes."""
        self._create_version_one_schema()

        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY
                )
                """
            )

        with self.assertRaisesRegex(
            DatabaseError,
            "SQLite database operation failed",
        ):
            self.database.initialize()

        with self.database.connect() as connection:
            version_row = connection.execute(
                """
                SELECT version
                FROM schema_version
                WHERE id = ?
                """,
                (1,),
            ).fetchone()

        self.assertEqual(version_row, (1,))
        self.assertFalse(self._table_exists("tasks"))
        self.assertTrue(self._table_exists("memories"))
        self.assertFalse(self._table_exists("journal_entries"))

    def test_rejects_future_schema_version(self) -> None:
        """A database from a future version should be rejected."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL CHECK (version >= 1)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO schema_version (id, version)
                VALUES (?, ?)
                """,
                (
                    1,
                    CURRENT_SCHEMA_VERSION + 1,
                ),
            )

        with self.assertRaisesRegex(
            DatabaseError,
            "Unsupported database schema version",
        ):
            self.database.initialize()

    def test_rejects_malformed_technical_schema(self) -> None:
        """Malformed technical metadata should become a database error."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY
                )
                """
            )

        with self.assertRaisesRegex(
            DatabaseError,
            "SQLite database operation failed",
        ) as raised_error:
            self.database.initialize()

        self.assertIsInstance(
            raised_error.exception.__cause__,
            sqlite3.Error,
        )

    def test_rejects_incomplete_current_business_schema(self) -> None:
        """A current version without its business tables is malformed."""
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL CHECK (version >= 1)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO schema_version (id, version)
                VALUES (?, ?)
                """,
                (
                    1,
                    CURRENT_SCHEMA_VERSION,
                ),
            )

        with self.assertRaisesRegex(
            DatabaseError,
            "Malformed database business schema",
        ):
            self.database.initialize()

    def test_foreign_keys_remain_enabled_after_initialization(self) -> None:
        """Schema initialization should preserve foreign key enforcement."""
        self.database.initialize()

        with self.database.connect() as connection:
            foreign_keys_status = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()

        self.assertEqual(foreign_keys_status, (1,))


if __name__ == "__main__":
    unittest.main()
