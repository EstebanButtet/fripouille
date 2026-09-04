"""Repository SQLite des observations sociales explicitement non confirmées."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.social_models import (
    Observation,
    ObservationCategory,
    ObservationSource,
    ObservationStatus,
)

DEFAULT_OBSERVATION_LIMIT = 50
MAX_OBSERVATION_LIMIT = 500

_SELECT = """
    SELECT id, person_id, category, content, source, source_text,
           confidence, status, created_at
    FROM person_observations
"""


class ObservationNotFoundError(RepositoryError):
    """Signaler que l'observation demandée n'existe pas."""


class ObservationRepository:
    """Créer, relire, lister et supprimer des observations par personne."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(database, SQLiteDatabase):
            raise TypeError("Observation repository database must be a SQLiteDatabase.")
        if clock is not None and not callable(clock):
            raise TypeError("Observation repository clock must be callable.")
        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_observation(
        self,
        person_id: int,
        category: ObservationCategory,
        content: str,
        *,
        source: ObservationSource = "manual_entry",
        source_text: str | None = None,
        confidence: float = 1.0,
    ) -> Observation:
        created_at = _normalized_now(self._clock())
        validated = Observation(
            id=1,
            person_id=person_id,
            category=category,
            content=content,
            source=source,
            source_text=source_text,
            confidence=confidence,
            status="unconfirmed",
            created_at=created_at,
        )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO person_observations (
                    person_id, category, content, source, source_text,
                    confidence, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.person_id,
                    validated.category,
                    validated.content,
                    validated.source,
                    validated.source_text,
                    validated.confidence,
                    validated.status,
                    validated.created_at.isoformat(),
                ),
            )
            observation_id = cursor.lastrowid
            if not isinstance(observation_id, int) or observation_id < 1:
                raise RepositoryError("Created observation identifier is invalid.")
            row = connection.execute(
                f"{_SELECT} WHERE id = ?", (observation_id,)
            ).fetchone()
        return _from_row(row)

    def get_observation(self, observation_id: int) -> Observation | None:
        normalized_id = _validate_identifier(observation_id, "Observation")
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_SELECT} WHERE id = ?", (normalized_id,)
            ).fetchone()
        return None if row is None else _from_row(row)

    def list_observations(
        self,
        person_id: int,
        limit: int = DEFAULT_OBSERVATION_LIMIT,
    ) -> tuple[Observation, ...]:
        normalized_person_id = _validate_identifier(person_id, "Person")
        normalized_limit = _validate_limit(limit)
        with self._database.connect() as connection:
            rows = connection.execute(
                f"""
                {_SELECT}
                WHERE person_id = ? AND status = 'unconfirmed'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (normalized_person_id, normalized_limit),
            ).fetchall()
        return tuple(_from_row(row) for row in rows)

    def delete_observation(self, observation_id: int) -> Observation:
        existing = self.get_observation(observation_id)
        if existing is None:
            raise ObservationNotFoundError(
                f"Observation {observation_id} does not exist."
            )
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM person_observations WHERE id = ?", (existing.id,)
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Observation deletion could not be persisted.")
        return existing


def _from_row(row: tuple[object, ...] | None) -> Observation:
    if row is None or len(row) != 9:
        raise RepositoryError("Stored observation data is incomplete.")
    try:
        return Observation(
            id=cast(int, row[0]),
            person_id=cast(int, row[1]),
            category=cast(ObservationCategory, row[2]),
            content=cast(str, row[3]),
            source=cast(ObservationSource, row[4]),
            source_text=cast(str | None, row[5]),
            confidence=cast(float, row[6]),
            status=cast(ObservationStatus, row[7]),
            created_at=_parse_datetime(row[8]),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError("Stored observation data is invalid.") from error


def _validate_identifier(value: int, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} identifier must be an integer.")
    if value < 1:
        raise ValueError(f"{subject} identifier must be greater than zero.")
    return value


def _validate_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Observation result limit must be an integer.")
    if value < 1 or value > MAX_OBSERVATION_LIMIT:
        raise ValueError("Observation result limit must be between 1 and 500.")
    return value


def _normalized_now(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Observation clock must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Observation clock must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Stored observation time must be text.")
    return datetime.fromisoformat(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
