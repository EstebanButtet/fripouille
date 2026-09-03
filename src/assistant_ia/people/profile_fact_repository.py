"""Repository SQLite dédié aux faits de profil confirmés d'une personne."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast

from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.profile_models import (
    ALLOWED_PROFILE_FACT_SOURCES,
    ProfileFact,
    ProfileFactCandidate,
    ProfileFactCategory,
    ProfileFactSource,
)

DEFAULT_PROFILE_FACT_LIMIT = 100
MAX_PROFILE_FACT_LIMIT = 500

_PROFILE_FACT_SELECT_COLUMNS = """
    SELECT
        id,
        person_id,
        category,
        content,
        source,
        source_text,
        confidence,
        created_at,
        updated_at
    FROM profile_facts
"""


class ProfileFactNotFoundError(RepositoryError):
    """Signaler qu'un fait de profil demandé n'existe pas."""


class ProfileFactRepository:
    """Créer, relire, corriger et supprimer les faits, personne par personne."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(database, SQLiteDatabase):
            raise TypeError(
                "Profile fact repository database must be a SQLiteDatabase."
            )
        if clock is not None and not callable(clock):
            raise TypeError("Profile fact repository clock must be callable.")
        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_profile_fact(
        self,
        person_id: int,
        category: ProfileFactCategory,
        content: str,
        *,
        source: ProfileFactSource = "explicit_user",
        source_text: str | None = None,
        confidence: float = 1.0,
    ) -> ProfileFact:
        """Créer un fait explicitement confirmé pour une personne existante."""
        # Le modèle temporaire centralise les validations sans inventer d'ID.
        candidate = ProfileFactCandidate(
            person_id=person_id,
            category=category,
            content=content,
            source_text=source_text if source_text is not None else content,
            confidence=confidence,
        )
        return self._insert(
            candidate,
            source=source,
            source_text=source_text,
        )

    def save_candidate(self, candidate: ProfileFactCandidate) -> ProfileFact:
        """Persister un candidat conversationnel après confirmation explicite."""
        if not isinstance(candidate, ProfileFactCandidate):
            raise TypeError(
                "Profile candidate persistence requires a ProfileFactCandidate."
            )
        return self._insert(
            candidate,
            source="conversation_analysis",
            source_text=candidate.source_text,
        )

    def get_profile_fact(self, fact_id: int) -> ProfileFact | None:
        """Retourner un fait par son identifiant stable, ou ``None``."""
        normalized_id = _validate_identifier(fact_id, "Profile fact")
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_PROFILE_FACT_SELECT_COLUMNS} WHERE id = ?",
                (normalized_id,),
            ).fetchone()
        return None if row is None else _profile_fact_from_row(row)

    def list_profile_facts(
        self,
        person_id: int,
        limit: int = DEFAULT_PROFILE_FACT_LIMIT,
    ) -> tuple[ProfileFact, ...]:
        """Lister uniquement les faits du sujet demandé, par identifiant."""
        normalized_person_id = _validate_identifier(person_id, "Person")
        normalized_limit = _validate_limit(limit)
        with self._database.connect() as connection:
            rows = connection.execute(
                f"""
                {_PROFILE_FACT_SELECT_COLUMNS}
                WHERE person_id = ?
                ORDER BY id
                LIMIT ?
                """,
                (normalized_person_id, normalized_limit),
            ).fetchall()
        return tuple(_profile_fact_from_row(row) for row in rows)

    def update_profile_fact(
        self,
        fact_id: int,
        candidate: ProfileFactCandidate,
    ) -> ProfileFact:
        """Corriger un fait avec un candidat visant exactement le même sujet."""
        normalized_id = _validate_identifier(fact_id, "Profile fact")
        if not isinstance(candidate, ProfileFactCandidate):
            raise TypeError("Profile fact update requires a ProfileFactCandidate.")

        with self._database.connect() as connection:
            row = connection.execute(
                f"{_PROFILE_FACT_SELECT_COLUMNS} WHERE id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise ProfileFactNotFoundError(
                    f"Profile fact {normalized_id} does not exist."
                )
            existing = _profile_fact_from_row(row)
            if existing.person_id != candidate.person_id:
                raise ValueError(
                    "Profile fact correction cannot change its person."
                )

            updated_at = _normalize_datetime(self._clock())
            if updated_at <= existing.updated_at:
                updated_at = existing.updated_at + timedelta(microseconds=1)

            cursor = connection.execute(
                """
                UPDATE profile_facts
                SET category = ?, content = ?, source = ?, source_text = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ? AND person_id = ?
                """,
                (
                    candidate.category,
                    candidate.content,
                    "conversation_analysis",
                    candidate.source_text,
                    candidate.confidence,
                    updated_at.isoformat(),
                    normalized_id,
                    candidate.person_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Profile fact update could not be persisted.")
            updated_row = connection.execute(
                f"{_PROFILE_FACT_SELECT_COLUMNS} WHERE id = ?",
                (normalized_id,),
            ).fetchone()

        updated = _profile_fact_from_row(updated_row)
        if updated.created_at != existing.created_at:
            raise RepositoryError("Profile fact creation time changed during update.")
        return updated

    def delete_profile_fact(self, fact_id: int) -> ProfileFact:
        """Supprimer un fait identifié et retourner son dernier état."""
        normalized_id = _validate_identifier(fact_id, "Profile fact")
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_PROFILE_FACT_SELECT_COLUMNS} WHERE id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise ProfileFactNotFoundError(
                    f"Profile fact {normalized_id} does not exist."
                )
            fact = _profile_fact_from_row(row)
            cursor = connection.execute(
                "DELETE FROM profile_facts WHERE id = ?",
                (normalized_id,),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("Profile fact deletion could not be persisted.")
        return fact

    def _insert(
        self,
        candidate: ProfileFactCandidate,
        *,
        source: ProfileFactSource,
        source_text: str | None,
    ) -> ProfileFact:
        if not isinstance(source, str):
            raise TypeError("Profile fact source must be a string.")
        if source not in ALLOWED_PROFILE_FACT_SOURCES:
            raise ValueError(f"Unknown profile fact source: {source!r}.")
        if source == "conversation_analysis" and source_text is None:
            raise ValueError(
                "Analyzed profile facts require their exact source text."
            )
        created_at = _normalize_datetime(self._clock())
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO profile_facts (
                    person_id, category, content, source, source_text,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.person_id,
                    candidate.category,
                    candidate.content,
                    source,
                    source_text,
                    candidate.confidence,
                    created_at.isoformat(),
                    created_at.isoformat(),
                ),
            )
            fact_id = cursor.lastrowid
            if isinstance(fact_id, bool) or not isinstance(fact_id, int):
                raise RepositoryError("Created profile fact identifier is invalid.")
            row = connection.execute(
                f"{_PROFILE_FACT_SELECT_COLUMNS} WHERE id = ?",
                (fact_id,),
            ).fetchone()
        return _profile_fact_from_row(row)


def _profile_fact_from_row(row: tuple[object, ...] | None) -> ProfileFact:
    if row is None or len(row) != 9:
        raise RepositoryError("Stored profile fact data is incomplete.")
    try:
        return ProfileFact(
            id=cast(int, row[0]),
            person_id=cast(int, row[1]),
            category=cast(ProfileFactCategory, row[2]),
            content=cast(str, row[3]),
            source=cast(ProfileFactSource, row[4]),
            source_text=cast(str | None, row[5]),
            confidence=cast(float, row[6]),
            created_at=_parse_datetime(row[7]),
            updated_at=_parse_datetime(row[8]),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError("Stored profile fact data is invalid.") from error


def _validate_identifier(value: int, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} identifier must be an integer.")
    if value < 1:
        raise ValueError(f"{subject} identifier must be greater than zero.")
    return value


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("Profile fact limit must be an integer.")
    if limit < 1 or limit > MAX_PROFILE_FACT_LIMIT:
        raise ValueError("Profile fact limit must be between 1 and 500.")
    return limit


def _normalize_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Profile fact time must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Profile fact time must include timezone information.")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Stored profile fact time must be text.")
    return _normalize_datetime(datetime.fromisoformat(value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
