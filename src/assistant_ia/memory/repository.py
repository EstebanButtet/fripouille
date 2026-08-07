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
CURRENT_SCHEMA_VERSION = 1


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
        """Create and validate the current technical database schema."""
        with self.connect() as connection:
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
                (1, CURRENT_SCHEMA_VERSION),
            )

            version_row = connection.execute(
                """
                SELECT version
                FROM schema_version
                WHERE id = ?
                """,
                (1,),
            ).fetchone()

            if (
                version_row is None
                or version_row[0] != CURRENT_SCHEMA_VERSION
            ):
                raise DatabaseError(
                    "Unsupported database schema version."
                )


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
