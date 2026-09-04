"""Repository SQLite des relations conversationnelles par personne."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast

from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.social_models import (
    PersonRelationship,
    RelationshipFamiliarity,
    RelationshipInteractionStyle,
)

_SELECT = """
    SELECT person_id, familiarity, interaction_style, created_at, updated_at
    FROM person_relationships
"""


class PersonRelationshipNotFoundError(RepositoryError):
    """Signaler qu'aucune relation n'existe pour la personne demandée."""


class PersonRelationshipRepository:
    """Créer et modifier une relation optionnelle unique par personne."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(database, SQLiteDatabase):
            raise TypeError("Relationship repository database must be a SQLiteDatabase.")
        if clock is not None and not callable(clock):
            raise TypeError("Relationship repository clock must be callable.")
        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_relationship(
        self,
        person_id: int,
        familiarity: RelationshipFamiliarity = "new",
        interaction_style: RelationshipInteractionStyle = "neutral",
    ) -> PersonRelationship:
        now = _normalized_now(self._clock())
        relationship = PersonRelationship(
            person_id=person_id,
            familiarity=familiarity,
            interaction_style=interaction_style,
            created_at=now,
            updated_at=now,
        )
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO person_relationships (
                    person_id, familiarity, interaction_style,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relationship.person_id,
                    relationship.familiarity,
                    relationship.interaction_style,
                    relationship.created_at.isoformat(),
                    relationship.updated_at.isoformat(),
                ),
            )
        return relationship

    def get_relationship(self, person_id: int) -> PersonRelationship | None:
        normalized_id = _validate_identifier(person_id)
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_SELECT} WHERE person_id = ?", (normalized_id,)
            ).fetchone()
        return None if row is None else _from_row(row)

    def update_relationship(
        self,
        person_id: int,
        *,
        familiarity: RelationshipFamiliarity,
        interaction_style: RelationshipInteractionStyle,
    ) -> PersonRelationship:
        existing = self.get_relationship(person_id)
        if existing is None:
            raise PersonRelationshipNotFoundError(
                f"Relationship for person {person_id} does not exist."
            )
        updated_at = _normalized_now(self._clock())
        if updated_at <= existing.updated_at:
            updated_at = existing.updated_at + timedelta(microseconds=1)
        updated = PersonRelationship(
            person_id=existing.person_id,
            familiarity=familiarity,
            interaction_style=interaction_style,
            created_at=existing.created_at,
            updated_at=updated_at,
        )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE person_relationships
                SET familiarity = ?, interaction_style = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (
                    updated.familiarity,
                    updated.interaction_style,
                    updated.updated_at.isoformat(),
                    updated.person_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Relationship update could not be persisted.")
        return updated

    def delete_relationship(self, person_id: int) -> PersonRelationship:
        existing = self.get_relationship(person_id)
        if existing is None:
            raise PersonRelationshipNotFoundError(
                f"Relationship for person {person_id} does not exist."
            )
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM person_relationships WHERE person_id = ?",
                (existing.person_id,),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Relationship deletion could not be persisted.")
        return existing


def _from_row(row: tuple[object, ...]) -> PersonRelationship:
    if len(row) != 5:
        raise RepositoryError("Stored relationship data is incomplete.")
    try:
        return PersonRelationship(
            person_id=cast(int, row[0]),
            familiarity=cast(RelationshipFamiliarity, row[1]),
            interaction_style=cast(RelationshipInteractionStyle, row[2]),
            created_at=_parse_datetime(row[3]),
            updated_at=_parse_datetime(row[4]),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError("Stored relationship data is invalid.") from error


def _validate_identifier(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Person identifier must be an integer.")
    if value < 1:
        raise ValueError("Person identifier must be greater than zero.")
    return value


def _normalized_now(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Relationship clock must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Relationship clock must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Stored relationship time must be text.")
    return datetime.fromisoformat(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
