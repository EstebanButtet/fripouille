"""SQLite persistence infrastructure."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Iterator

DEFAULT_DATABASE_DIRECTORY_NAME = "assistant-ia"
DEFAULT_DATABASE_FILENAME = "assistant_ia.db"

_INITIAL_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = 2

_SCHEMA_VERSION_COLUMNS = (
    ("id", "INTEGER", 0, 1),
    ("version", "INTEGER", 1, 0),
)

_BUSINESS_TABLE_COLUMNS = {
    "tasks": (
        ("id", "INTEGER", 0, 1),
        ("title", "TEXT", 1, 0),
        ("due_at", "TEXT", 0, 0),
        ("status", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("completed_at", "TEXT", 0, 0),
    ),
    "memories": (
        ("id", "INTEGER", 0, 1),
        ("content", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
    ),
    "journal_entries": (
        ("id", "INTEGER", 0, 1),
        ("content", "TEXT", 1, 0),
        ("entry_date", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
    ),
}


class DatabaseError(RuntimeError):
    """Raised when the SQLite persistence layer cannot complete an operation."""


def default_database_path() -> Path:
    """Return the default user-local SQLite database path."""
    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data is None or not local_app_data.strip():
        raise DatabaseError("LOCALAPPDATA is not configured.")

    return _normalize_database_path(
        Path(local_app_data)
        / DEFAULT_DATABASE_DIRECTORY_NAME
        / DEFAULT_DATABASE_FILENAME
    )


class SQLiteDatabase:
    """Represent and manage a SQLite database stored at a local path."""

    def __init__(self, database_path: str | PathLike[str]) -> None:
        """Create a database manager with a normalized path."""
        self._path = _normalize_database_path(database_path)

    @property
    def path(self) -> Path:
        """Return the normalized absolute database path."""
        return self._path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured transactional connection and close it afterward."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DatabaseError(
                "Database directory could not be created."
            ) from error

        connection: sqlite3.Connection | None = None

        try:
            connection = sqlite3.connect(self._path)
            connection.execute("PRAGMA foreign_keys = ON")

            foreign_keys_status = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()

            if (
                foreign_keys_status is None
                or foreign_keys_status[0] != 1
            ):
                raise DatabaseError(
                    "SQLite foreign key enforcement could not be enabled."
                )

            with connection:
                yield connection
        except sqlite3.Error as error:
            raise DatabaseError(
                "SQLite database operation failed."
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        """Create, migrate and validate the current database schema."""
        with self.connect() as connection:
            connection.execute("BEGIN")

            _create_technical_schema(connection)
            schema_version = _read_schema_version(connection)

            if schema_version > CURRENT_SCHEMA_VERSION:
                raise DatabaseError(
                    "Unsupported database schema version."
                )

            while schema_version < CURRENT_SCHEMA_VERSION:
                migration = _SCHEMA_MIGRATIONS.get(schema_version)

                if migration is None:
                    raise DatabaseError(
                        "Unsupported database schema version."
                    )

                migration(connection)

                next_version = schema_version + 1
                update_cursor = connection.execute(
                    """
                    UPDATE schema_version
                    SET version = ?
                    WHERE id = ?
                      AND version = ?
                    """,
                    (
                        next_version,
                        1,
                        schema_version,
                    ),
                )

                if update_cursor.rowcount != 1:
                    raise DatabaseError(
                        "Database schema version could not be updated."
                    )

                schema_version = next_version

            _validate_current_business_schema(connection)


def _create_technical_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create and validate the technical schema metadata."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL CHECK (version >= 1)
        )
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_version (id, version)
        VALUES (?, ?)
        """,
        (
            1,
            _INITIAL_SCHEMA_VERSION,
        ),
    )

    if _table_columns(connection, "schema_version") != (
        _SCHEMA_VERSION_COLUMNS
    ):
        raise DatabaseError(
            "Malformed database schema metadata."
        )


def _read_schema_version(
    connection: sqlite3.Connection,
) -> int:
    """Return the single validated schema version."""
    version_rows = connection.execute(
        """
        SELECT id, version
        FROM schema_version
        ORDER BY id
        """
    ).fetchall()

    if len(version_rows) != 1:
        raise DatabaseError(
            "Malformed database schema metadata."
        )

    row_id, schema_version = version_rows[0]

    if (
        row_id != 1
        or not isinstance(schema_version, int)
        or schema_version < _INITIAL_SCHEMA_VERSION
    ):
        raise DatabaseError(
            "Malformed database schema metadata."
        )

    return schema_version


def _migrate_schema_1_to_2(
    connection: sqlite3.Connection,
) -> None:
    """Add the initial task, memory and journal business tables."""
    connection.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL
                CHECK (length(trim(title)) > 0),
            due_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'completed')),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            completed_at TEXT,
            CHECK (
                (
                    status = 'pending'
                    AND completed_at IS NULL
                )
                OR
                (
                    status = 'completed'
                    AND completed_at IS NOT NULL
                )
            )
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
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            entry_date TEXT NOT NULL
                CHECK (length(trim(entry_date)) > 0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0)
        )
        """
    )


def _validate_current_business_schema(
    connection: sqlite3.Connection,
) -> None:
    """Validate the required business table structures."""
    for table_name, expected_columns in _BUSINESS_TABLE_COLUMNS.items():
        if _table_columns(connection, table_name) != expected_columns:
            raise DatabaseError(
                "Malformed database business schema."
            )


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple[str, str, int, int], ...]:
    """Return the normalized column structure of an internal table."""
    column_rows = connection.execute(
        """
        SELECT
            name,
            upper(type),
            "notnull",
            pk
        FROM pragma_table_info(?)
        ORDER BY cid
        """,
        (table_name,),
    ).fetchall()

    return tuple(
        (
            column_name,
            column_type,
            not_null,
            primary_key,
        )
        for (
            column_name,
            column_type,
            not_null,
            primary_key,
        ) in column_rows
    )


_SCHEMA_MIGRATIONS = {
    1: _migrate_schema_1_to_2,
}


def _normalize_database_path(
    database_path: str | PathLike[str],
) -> Path:
    """Validate and normalize a SQLite database path."""
    try:
        raw_path = os.fspath(database_path)
    except TypeError as error:
        raise TypeError(
            "Database path must be a string or path-like object."
        ) from error

    if not isinstance(raw_path, str):
        raise TypeError(
            "Database path must be a string or path-like object."
        )

    if not raw_path.strip():
        raise ValueError("Database path cannot be empty.")

    if "\x00" in raw_path:
        raise ValueError("Database path cannot contain null bytes.")

    normalized_path = Path(raw_path).expanduser().resolve(strict=False)

    if normalized_path.exists() and normalized_path.is_dir():
        raise ValueError("Database path must reference a file.")

    return normalized_path
