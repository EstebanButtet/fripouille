"""SQLite repository for persistent assistant tasks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

from assistant_ia.memory.errors import (
    RepositoryError,
    TaskAlreadyCompletedError,
    TaskNotFoundError,
)
from assistant_ia.memory.models import (
    ALLOWED_TASK_STATUSES,
    Task,
    TaskStatus,
)
from assistant_ia.memory.repository import SQLiteDatabase

DEFAULT_TASK_RESULT_LIMIT = 50
MAX_TASK_RESULT_LIMIT = 100

_TASK_SELECT_COLUMNS = """
    SELECT
        id,
        title,
        due_at,
        status,
        created_at,
        completed_at
    FROM tasks
"""


class TaskRepository:
    """Create, list and complete tasks stored in SQLite."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a task repository with injectable persistence and time."""
        if not isinstance(database, SQLiteDatabase):
            raise TypeError(
                "Task repository database must be a SQLiteDatabase."
            )

        if clock is not None and not callable(clock):
            raise TypeError(
                "Task repository clock must be callable."
            )

        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_task(
        self,
        title: str,
        due_at: datetime | None = None,
    ) -> Task:
        """Create and return one pending task."""
        normalized_title = _normalize_title(title)
        normalized_due_at = _normalize_optional_datetime(
            due_at,
            field_name="Task due date",
        )
        created_at = _normalize_datetime(
            self._clock(),
            field_name="Task creation time",
        )

        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks (
                    title,
                    due_at,
                    status,
                    created_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    normalized_title,
                    _serialize_optional_datetime(normalized_due_at),
                    "pending",
                    _serialize_datetime(created_at),
                    None,
                ),
            )

            task_id = cursor.lastrowid

            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id < 1
            ):
                raise RepositoryError(
                    "Created task identifier is invalid."
                )

            task_row = connection.execute(
                f"""
                {_TASK_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        return _task_from_row(task_row)

    def list_tasks(
        self,
        status: TaskStatus | None = "pending",
        limit: int = DEFAULT_TASK_RESULT_LIMIT,
    ) -> tuple[Task, ...]:
        """Return tasks in deterministic creation order."""
        normalized_status = _normalize_optional_task_status(status)
        normalized_limit = _validate_result_limit(limit)

        with self._database.connect() as connection:
            if normalized_status is None:
                task_rows = connection.execute(
                    f"""
                    {_TASK_SELECT_COLUMNS}
                    ORDER BY created_at, id
                    LIMIT ?
                    """,
                    (normalized_limit,),
                ).fetchall()
            else:
                task_rows = connection.execute(
                    f"""
                    {_TASK_SELECT_COLUMNS}
                    WHERE status = ?
                    ORDER BY created_at, id
                    LIMIT ?
                    """,
                    (
                        normalized_status,
                        normalized_limit,
                    ),
                ).fetchall()

        return tuple(
            _task_from_row(task_row)
            for task_row in task_rows
        )

    def complete_task(self, task_id: int) -> Task:
        """Mark one pending task as completed and return it."""
        normalized_task_id = _validate_identifier(task_id)
        completed_at = _normalize_datetime(
            self._clock(),
            field_name="Task completion time",
        )

        with self._database.connect() as connection:
            task_row = connection.execute(
                f"""
                {_TASK_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_task_id,),
            ).fetchone()

            if task_row is None:
                raise TaskNotFoundError(
                    f"Task {normalized_task_id} does not exist."
                )

            existing_task = _task_from_row(task_row)

            if existing_task.status == "completed":
                raise TaskAlreadyCompletedError(
                    f"Task {normalized_task_id} is already completed."
                )

            if completed_at < existing_task.created_at:
                raise RepositoryError(
                    "Task completion time cannot precede creation time."
                )

            update_cursor = connection.execute(
                """
                UPDATE tasks
                SET
                    status = ?,
                    completed_at = ?
                WHERE id = ?
                  AND status = ?
                """,
                (
                    "completed",
                    _serialize_datetime(completed_at),
                    normalized_task_id,
                    "pending",
                ),
            )

            if update_cursor.rowcount != 1:
                raise RepositoryError(
                    "Task completion could not be persisted."
                )

            completed_task_row = connection.execute(
                f"""
                {_TASK_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_task_id,),
            ).fetchone()

        return _task_from_row(completed_task_row)


def _task_from_row(
    task_row: tuple[object, ...] | None,
) -> Task:
    """Convert one validated SQLite row into a task model."""
    if task_row is None or len(task_row) != 6:
        raise RepositoryError(
            "Stored task data is incomplete."
        )

    (
        task_id,
        title,
        due_at,
        status,
        created_at,
        completed_at,
    ) = task_row

    try:
        return Task(
            id=cast(int, task_id),
            title=cast(str, title),
            due_at=_parse_optional_datetime(
                due_at,
                field_name="Stored task due date",
            ),
            status=cast(TaskStatus, status),
            created_at=_parse_datetime(
                created_at,
                field_name="Stored task creation time",
            ),
            completed_at=_parse_optional_datetime(
                completed_at,
                field_name="Stored task completion time",
            ),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            "Stored task data is invalid."
        ) from error


def _normalize_title(title: str) -> str:
    """Return a normalized non-empty task title."""
    if not isinstance(title, str):
        raise TypeError(
            "Task title must be a string."
        )

    normalized_title = title.strip()

    if not normalized_title:
        raise ValueError(
            "Task title cannot be empty."
        )

    return normalized_title


def _validate_identifier(task_id: int) -> int:
    """Return a validated positive task identifier."""
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError(
            "Task identifier must be an integer."
        )

    if task_id < 1:
        raise ValueError(
            "Task identifier must be greater than zero."
        )

    return task_id


def _validate_result_limit(limit: int) -> int:
    """Return a bounded positive task result limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Task result limit must be an integer."
        )

    if limit < 1 or limit > MAX_TASK_RESULT_LIMIT:
        raise ValueError(
            "Task result limit must be between 1 and 100."
        )

    return limit


def _normalize_optional_task_status(
    status: TaskStatus | None,
) -> TaskStatus | None:
    """Return an optional validated task status."""
    if status is None:
        return None

    if not isinstance(status, str):
        raise TypeError(
            "Task status must be a string or None."
        )

    normalized_status = status.strip()

    if normalized_status not in ALLOWED_TASK_STATUSES:
        raise ValueError(
            f"Unknown task status: {normalized_status!r}."
        )

    return cast(TaskStatus, normalized_status)


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


def _normalize_optional_datetime(
    value: datetime | None,
    *,
    field_name: str,
) -> datetime | None:
    """Normalize an optional timezone-aware datetime."""
    if value is None:
        return None

    return _normalize_datetime(
        value,
        field_name=field_name,
    )


def _serialize_datetime(value: datetime) -> str:
    """Serialize one normalized datetime as ISO 8601."""
    return value.astimezone(timezone.utc).isoformat()


def _serialize_optional_datetime(
    value: datetime | None,
) -> str | None:
    """Serialize an optional normalized datetime."""
    if value is None:
        return None

    return _serialize_datetime(value)


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


def _parse_optional_datetime(
    value: object,
    *,
    field_name: str,
) -> datetime | None:
    """Parse an optional persisted ISO 8601 datetime."""
    if value is None:
        return None

    return _parse_datetime(
        value,
        field_name=field_name,
    )


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)
