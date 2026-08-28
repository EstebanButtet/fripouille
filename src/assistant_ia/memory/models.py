"""Persistent business models for tasks, memories and journal entries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from typing import Literal, cast

TaskStatus = Literal[
    "pending",
    "completed",
]

ALLOWED_TASK_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "completed",
    }
)

MemorySource = Literal[
    "legacy_explicit",
    "explicit_user",
]

ALLOWED_MEMORY_SOURCES: frozenset[str] = frozenset(
    {
        "legacy_explicit",
        "explicit_user",
    }
)


@dataclass(frozen=True, slots=True)
class Task:
    """Represent one persisted assistant task."""

    id: int
    title: str
    due_at: datetime | None
    status: TaskStatus
    created_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        """Validate and normalize persisted task data."""
        normalized_id = _validate_identifier(self.id)
        normalized_title = _normalize_required_text(
            self.title,
            field_name="Task title",
        )
        normalized_status = _normalize_task_status(self.status)
        normalized_due_at = _normalize_optional_datetime(
            self.due_at,
            field_name="Task due date",
        )
        normalized_created_at = _normalize_datetime(
            self.created_at,
            field_name="Task creation time",
        )
        normalized_completed_at = _normalize_optional_datetime(
            self.completed_at,
            field_name="Task completion time",
        )

        if (
            normalized_status == "pending"
            and normalized_completed_at is not None
        ):
            raise ValueError(
                "Pending tasks cannot have a completion time."
            )

        if (
            normalized_status == "completed"
            and normalized_completed_at is None
        ):
            raise ValueError(
                "Completed tasks must have a completion time."
            )

        if (
            normalized_completed_at is not None
            and normalized_completed_at < normalized_created_at
        ):
            raise ValueError(
                "Task completion time cannot precede its creation time."
            )

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "title", normalized_title)
        object.__setattr__(self, "due_at", normalized_due_at)
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "created_at", normalized_created_at)
        object.__setattr__(
            self,
            "completed_at",
            normalized_completed_at,
        )


@dataclass(frozen=True, slots=True)
class Memory:
    """Represent one persisted assistant memory."""

    id: int
    content: str
    source: MemorySource
    source_text: str | None
    confidence: float
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """Validate and normalize persisted memory data."""
        normalized_created_at = _normalize_datetime(
            self.created_at,
            field_name="Memory creation time",
        )
        normalized_updated_at = _normalize_datetime(
            self.updated_at,
            field_name="Memory update time",
        )

        if normalized_updated_at < normalized_created_at:
            raise ValueError(
                "Memory update time cannot precede its creation time."
            )

        object.__setattr__(self, "id", _validate_identifier(self.id))
        object.__setattr__(
            self,
            "content",
            _normalize_required_text(
                self.content,
                field_name="Memory content",
            ),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_memory_source(self.source),
        )
        object.__setattr__(
            self,
            "source_text",
            _validate_optional_source_text(self.source_text),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_memory_confidence(self.confidence),
        )
        object.__setattr__(self, "created_at", normalized_created_at)
        object.__setattr__(self, "updated_at", normalized_updated_at)


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """Represent one persisted journal entry."""

    id: int
    content: str
    entry_date: date
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate and normalize persisted journal data."""
        if (
            isinstance(self.entry_date, datetime)
            or not isinstance(self.entry_date, date)
        ):
            raise TypeError(
                "Journal entry date must be a date."
            )

        object.__setattr__(
            self,
            "id",
            _validate_identifier(self.id),
        )
        object.__setattr__(
            self,
            "content",
            _normalize_required_text(
                self.content,
                field_name="Journal content",
            ),
        )
        object.__setattr__(
            self,
            "created_at",
            _normalize_datetime(
                self.created_at,
                field_name="Journal creation time",
            ),
        )


def _validate_identifier(identifier: int) -> int:
    """Return a validated positive persisted identifier."""
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise TypeError(
            "Persisted identifier must be an integer."
        )

    if identifier < 1:
        raise ValueError(
            "Persisted identifier must be greater than zero."
        )

    return identifier


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Return normalized non-empty business text."""
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


def _normalize_task_status(
    status: TaskStatus,
) -> TaskStatus:
    """Return a validated task status."""
    if not isinstance(status, str):
        raise TypeError(
            "Task status must be a string."
        )

    normalized_status = status.strip()

    if normalized_status not in ALLOWED_TASK_STATUSES:
        raise ValueError(
            f"Unknown task status: {normalized_status!r}."
        )

    return cast(TaskStatus, normalized_status)


def _normalize_memory_source(
    source: MemorySource,
) -> MemorySource:
    """Return a validated known memory provenance."""
    if not isinstance(source, str):
        raise TypeError(
            "Memory source must be a string."
        )

    normalized_source = source.strip()

    if not normalized_source:
        raise ValueError(
            "Memory source cannot be empty."
        )

    if normalized_source not in ALLOWED_MEMORY_SOURCES:
        raise ValueError(
            f"Unknown memory source: {normalized_source!r}."
        )

    return cast(MemorySource, normalized_source)


def _validate_optional_source_text(
    source_text: str | None,
) -> str | None:
    """Validate optional exact user evidence without rewriting it."""
    if source_text is None:
        return None

    if not isinstance(source_text, str):
        raise TypeError(
            "Memory source text must be a string or None."
        )

    if not source_text.strip():
        raise ValueError(
            "Memory source text cannot be empty."
        )

    return source_text


def _normalize_memory_confidence(
    confidence: float,
) -> float:
    """Return confidence in the validated recording mechanism."""
    if isinstance(confidence, bool) or not isinstance(
        confidence,
        (int, float),
    ):
        raise TypeError(
            "Memory confidence must be a number."
        )

    normalized_confidence = float(confidence)

    if (
        not math.isfinite(normalized_confidence)
        or normalized_confidence < 0.0
        or normalized_confidence > 1.0
    ):
        raise ValueError(
            "Memory confidence must be between 0.0 and 1.0."
        )

    return normalized_confidence


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Return one timezone-aware datetime normalized to UTC."""
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
