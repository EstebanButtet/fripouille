"""Tests for the SQLite persistence infrastructure."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant_ia.memory.repository import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_DATABASE_DIRECTORY_NAME,
    DEFAULT_DATABASE_FILENAME,
    DatabaseError,
    SQLiteDatabase,
    default_database_path,
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
    """Validate technical schema initialization and versioning."""

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

    def test_initialize_creates_schema_version(self) -> None:
        """Initialization should create and record the current schema."""
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

    def test_initialize_is_idempotent(self) -> None:
        """Repeated initialization should not duplicate schema metadata."""
        self.database.initialize()
        self.database.initialize()

        with self.database.connect() as connection:
            result = connection.execute(
                "SELECT COUNT(*) FROM schema_version"
            ).fetchone()

        self.assertEqual(result, (1,))

    def test_initialize_creates_only_technical_schema(self) -> None:
        """Step 7 should not create future business tables."""
        self.database.initialize()

        with self.database.connect() as connection:
            tables = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = ?
                  AND name NOT LIKE ?
                ORDER BY name
                """,
                ("table", "sqlite_%"),
            ).fetchall()

        self.assertEqual(tables, [("schema_version",)])

    def test_rejects_unsupported_schema_version(self) -> None:
        """Databases using another schema version should be rejected."""
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
                (1, CURRENT_SCHEMA_VERSION + 1),
            )

        with self.assertRaisesRegex(
            DatabaseError,
            "Unsupported database schema version",
        ):
            self.database.initialize()

    def test_converts_invalid_schema_table_error(self) -> None:
        """Malformed technical schema should become a database error."""
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


if __name__ == "__main__":
    unittest.main()
