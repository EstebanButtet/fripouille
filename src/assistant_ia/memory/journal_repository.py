"""Repository SQLite des entrées persistantes du journal.

Ce module traduit l'écriture d'une note datée en transaction SQLite et
reconstruit un :class:`JournalEntry` validé. Il ne décide pas si une intention
est autorisée : cette validation se situe dans les actions et leur registre.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import cast

from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.models import JournalEntry
from assistant_ia.memory.repository import SQLiteDatabase

_JOURNAL_ENTRY_SELECT_COLUMNS = """
    SELECT
        id,
        content,
        entry_date,
        created_at
    FROM journal_entries
"""


class JournalRepository:
    """Écrire des entrées de journal dans SQLite.

    La date courante et l'horloge sont injectables pour rendre les tests
    indépendants du calendrier réel. Une entrée possède une date métier et un
    instant technique de création distincts.
    """

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Créer le repository avec base, calendrier et horloge injectables."""
        if not isinstance(database, SQLiteDatabase):
            raise TypeError(
                "Journal repository database must be a SQLiteDatabase."
            )

        if clock is not None and not callable(clock):
            raise TypeError(
                "Journal repository clock must be callable."
            )

        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def write_journal_entry(
        self,
        content: str,
        entry_date: date,
    ) -> JournalEntry:
        """Valider, persister puis relire une entrée de journal."""
        normalized_content = _normalize_required_text(
            content,
            field_name="Journal content",
        )
        normalized_entry_date = _validate_entry_date(entry_date)
        created_at = _normalize_datetime(
            self._clock(),
            field_name="Journal creation time",
        )

        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO journal_entries (
                    content,
                    entry_date,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    normalized_content,
                    normalized_entry_date.isoformat(),
                    _serialize_datetime(created_at),
                ),
            )

            journal_entry_id = cursor.lastrowid

            if (
                isinstance(journal_entry_id, bool)
                or not isinstance(journal_entry_id, int)
                or journal_entry_id < 1
            ):
                raise RepositoryError(
                    "Created journal entry identifier is invalid."
                )

            journal_entry_row = connection.execute(
                f"""
                {_JOURNAL_ENTRY_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (journal_entry_id,),
            ).fetchone()

        return _journal_entry_from_row(journal_entry_row)


def _journal_entry_from_row(
    journal_entry_row: tuple[object, ...] | None,
) -> JournalEntry:
    """Convertir une ligne SQLite complète en modèle de journal validé."""
    if journal_entry_row is None or len(journal_entry_row) != 4:
        raise RepositoryError(
            "Stored journal entry data is incomplete."
        )

    (
        journal_entry_id,
        content,
        entry_date,
        created_at,
    ) = journal_entry_row

    try:
        return JournalEntry(
            id=cast(int, journal_entry_id),
            content=cast(str, content),
            entry_date=_parse_date(
                entry_date,
                field_name="Stored journal entry date",
            ),
            created_at=_parse_datetime(
                created_at,
                field_name="Stored journal creation time",
            ),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            "Stored journal entry data is invalid."
        ) from error


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Retourner un texte non vide après normalisation extérieure."""
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


def _validate_entry_date(value: date) -> date:
    """Valider et retourner une date sans information horaire."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(
            "Journal entry date must be a date."
        )

    return value


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Retourner une date-heure avec fuseau normalisée en UTC."""
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
    """Sérialiser une date-heure normalisée au format ISO 8601."""
    return value.astimezone(timezone.utc).isoformat()


def _parse_date(
    value: object,
    *,
    field_name: str,
) -> date:
    """Reconstruire une date persistée au format ISO 8601."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be stored as text."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} is not valid ISO 8601."
        ) from error


def _parse_datetime(
    value: object,
    *,
    field_name: str,
) -> datetime:
    """Reconstruire une date-heure persistée au format ISO 8601."""
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
    """Retourner l'instant courant en UTC avec information de fuseau."""
    return datetime.now(timezone.utc)
