"""SQLite repository for persistent assistant memories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast

from assistant_ia.memory.errors import (
    MemoryNotFoundError,
    RepositoryError,
)
from assistant_ia.memory.models import (
    Memory,
    MemoryCandidate,
    MemorySource,
)
from assistant_ia.memory.repository import SQLiteDatabase

DEFAULT_MEMORY_RESULT_LIMIT = 20
MAX_MEMORY_RESULT_LIMIT = 100
DEFAULT_MEMORY_LIST_LIMIT = 500
MAX_MEMORY_LIST_LIMIT = 1000

_MEMORY_SELECT_COLUMNS = """
    SELECT
        id,
        content,
        source,
        source_text,
        confidence,
        created_at,
        updated_at
    FROM memories
"""


class MemoryRepository:
    """Save, search, update and delete memories stored in SQLite."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a memory repository with injectable persistence and time."""
        if not isinstance(database, SQLiteDatabase):
            raise TypeError(
                "Memory repository database must be a SQLiteDatabase."
            )

        if clock is not None and not callable(clock):
            raise TypeError(
                "Memory repository clock must be callable."
            )

        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def save_memory(self, content: str) -> Memory:
        """Persist and return one assistant memory."""
        normalized_content = _normalize_required_text(
            content,
            field_name="Memory content",
        )
        created_at = _normalize_datetime(
            self._clock(),
            field_name="Memory creation time",
        )
        source: MemorySource = "explicit_user"
        source_text = None
        confidence = 1.0
        updated_at = created_at

        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    content,
                    source,
                    source_text,
                    confidence,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_content,
                    source,
                    source_text,
                    confidence,
                    _serialize_datetime(created_at),
                    _serialize_datetime(updated_at),
                ),
            )

            memory_id = cursor.lastrowid

            if (
                isinstance(memory_id, bool)
                or not isinstance(memory_id, int)
                or memory_id < 1
            ):
                raise RepositoryError(
                    "Created memory identifier is invalid."
                )

            memory_row = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        return _memory_from_row(memory_row)

    def save_candidate(self, candidate: MemoryCandidate) -> Memory:
        """Persist one explicitly confirmed conversation candidate."""
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "Candidate persistence requires a MemoryCandidate."
            )

        created_at = _normalize_datetime(
            self._clock(),
            field_name="Memory creation time",
        )

        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    content,
                    source,
                    source_text,
                    confidence,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.content,
                    "conversation_analysis",
                    candidate.source_text,
                    candidate.confidence,
                    _serialize_datetime(created_at),
                    _serialize_datetime(created_at),
                ),
            )
            memory_id = cursor.lastrowid
            if (
                isinstance(memory_id, bool)
                or not isinstance(memory_id, int)
                or memory_id < 1
            ):
                raise RepositoryError(
                    "Created memory identifier is invalid."
                )

            memory_row = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        return _memory_from_row(memory_row)

    def update_memory(
        self,
        memory_id: int,
        candidate: MemoryCandidate,
    ) -> Memory:
        """Replace one memory from an explicitly confirmed candidate."""
        normalized_memory_id = _validate_identifier(memory_id)
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "Memory update requires a MemoryCandidate."
            )

        with self._database.connect() as connection:
            existing_row = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_memory_id,),
            ).fetchone()
            if existing_row is None:
                raise MemoryNotFoundError(
                    f"Memory {normalized_memory_id} does not exist."
                )

            existing_memory = _memory_from_row(existing_row)
            updated_at = _normalize_datetime(
                self._clock(),
                field_name="Memory update time",
            )
            if updated_at <= existing_memory.updated_at:
                updated_at = (
                    existing_memory.updated_at
                    + timedelta(microseconds=1)
                )

            update_cursor = connection.execute(
                """
                UPDATE memories
                SET content = ?,
                    source = ?,
                    source_text = ?,
                    confidence = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    candidate.content,
                    "conversation_analysis",
                    candidate.source_text,
                    candidate.confidence,
                    _serialize_datetime(updated_at),
                    normalized_memory_id,
                ),
            )
            if update_cursor.rowcount != 1:
                raise RepositoryError(
                    "Memory update could not be persisted."
                )

            updated_row = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_memory_id,),
            ).fetchone()
            updated_memory = _memory_from_row(updated_row)

            if updated_memory.created_at != existing_memory.created_at:
                raise RepositoryError(
                    "Memory creation time changed during update."
                )

        return updated_memory

    def find_memories(
        self,
        query: str,
        limit: int = DEFAULT_MEMORY_RESULT_LIMIT,
    ) -> tuple[Memory, ...]:
        """Return memories containing one literal text query."""
        normalized_query = _normalize_required_text(
            query,
            field_name="Memory search query",
        )
        normalized_limit = _validate_result_limit(limit)
        search_pattern = (
            f"%{_escape_like_literal(normalized_query)}%"
        )

        with self._database.connect() as connection:
            memory_rows = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE content LIKE ? ESCAPE '!'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (
                    search_pattern,
                    normalized_limit,
                ),
            ).fetchall()

        return tuple(
            _memory_from_row(memory_row)
            for memory_row in memory_rows
        )

    def list_memories(
        self,
        limit: int = DEFAULT_MEMORY_LIST_LIMIT,
    ) -> tuple[Memory, ...]:
        """Return a bounded recent memory set for local processing."""
        normalized_limit = _validate_list_limit(limit)

        with self._database.connect() as connection:
            memory_rows = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()

        return tuple(
            _memory_from_row(memory_row)
            for memory_row in memory_rows
        )

    def delete_memory(self, memory_id: int) -> Memory:
        """Delete exactly one memory selected by its identifier."""
        normalized_memory_id = _validate_identifier(memory_id)

        with self._database.connect() as connection:
            memory_row = connection.execute(
                f"""
                {_MEMORY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_memory_id,),
            ).fetchone()

            if memory_row is None:
                raise MemoryNotFoundError(
                    f"Memory {normalized_memory_id} does not exist."
                )

            memory = _memory_from_row(memory_row)

            delete_cursor = connection.execute(
                """
                DELETE FROM memories
                WHERE id = ?
                """,
                (normalized_memory_id,),
            )

            if delete_cursor.rowcount != 1:
                raise RepositoryError(
                    "Memory deletion could not be persisted."
                )

        return memory


def _memory_from_row(
    memory_row: tuple[object, ...] | None,
) -> Memory:
    """Convert one validated SQLite row into a memory model."""
    if memory_row is None or len(memory_row) != 7:
        raise RepositoryError(
            "Stored memory data is incomplete."
        )

    (
        memory_id,
        content,
        source,
        source_text,
        confidence,
        created_at,
        updated_at,
    ) = memory_row

    try:
        return Memory(
            id=cast(int, memory_id),
            content=cast(str, content),
            source=cast(MemorySource, source),
            source_text=cast(str | None, source_text),
            confidence=cast(float, confidence),
            created_at=_parse_datetime(
                created_at,
                field_name="Stored memory creation time",
            ),
            updated_at=_parse_datetime(
                updated_at,
                field_name="Stored memory update time",
            ),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            "Stored memory data is invalid."
        ) from error


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Return normalized non-empty text."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized_value


def _validate_identifier(memory_id: int) -> int:
    """Return a validated positive memory identifier."""
    if isinstance(memory_id, bool) or not isinstance(memory_id, int):
        raise TypeError(
            "Memory identifier must be an integer."
        )

    if memory_id < 1:
        raise ValueError(
            "Memory identifier must be greater than zero."
        )

    return memory_id


def _validate_result_limit(limit: int) -> int:
    """Return a bounded positive memory result limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Memory result limit must be an integer."
        )

    if limit < 1 or limit > MAX_MEMORY_RESULT_LIMIT:
        raise ValueError(
            "Memory result limit must be between 1 and 100."
        )

    return limit


def _validate_list_limit(limit: int) -> int:
    """Return a bounded positive memory listing limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Memory list limit must be an integer."
        )

    if limit < 1 or limit > MAX_MEMORY_LIST_LIMIT:
        raise ValueError(
            "Memory list limit must be between 1 and 1000."
        )

    return limit


def _escape_like_literal(value: str) -> str:
    """Escape SQLite LIKE wildcard characters for literal searching."""
    return (
        value
        .replace("!", "!!")
        .replace("%", "!%")
        .replace("_", "!_")
    )


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Return a timezone-aware datetime normalized to UTC."""
    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} must be a datetime."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field_name} must include timezone information."
        )

    return value.astimezone(timezone.utc)


def _serialize_datetime(value: datetime) -> str:
    """Serialize one normalized datetime as ISO 8601."""
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(
    value: object,
    *,
    field_name: str,
) -> datetime:
    """Parse one persisted ISO 8601 datetime."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be stored as text."
        )

    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} is not valid ISO 8601."
        ) from error


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)
