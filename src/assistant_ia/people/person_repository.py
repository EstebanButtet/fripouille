"""Repository SQLite du registre persistant minimal des personnes.

Le repository crée et relit des identités stables sans leur attribuer de
profil détaillé, relation, rôle, alias ou souvenir. Le nom affiché reste une
donnée de la personne et n'est jamais utilisé comme clé persistante.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.models import Person

DEFAULT_PERSON_RESULT_LIMIT = 50
MAX_PERSON_RESULT_LIMIT = 100

_PERSON_SELECT_COLUMNS = """
    SELECT
        id,
        display_name,
        created_at
    FROM persons
"""


class PersonRepository:
    """Créer, retrouver et lister les personnes stockées dans SQLite."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Créer le repository avec une base et une horloge injectables."""
        if not isinstance(database, SQLiteDatabase):
            raise TypeError(
                "Person repository database must be a SQLiteDatabase."
            )

        if clock is not None and not callable(clock):
            raise TypeError(
                "Person repository clock must be callable."
            )

        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_person(self, display_name: str) -> Person:
        """Créer une identité persistante puis relire sa ligne complète."""
        normalized_display_name = _normalize_display_name(display_name)
        created_at = _normalize_datetime(self._clock())

        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO persons (
                    display_name,
                    created_at
                )
                VALUES (?, ?)
                """,
                (
                    normalized_display_name,
                    created_at.isoformat(),
                ),
            )

            person_id = cursor.lastrowid

            if (
                isinstance(person_id, bool)
                or not isinstance(person_id, int)
                or person_id < 1
            ):
                raise RepositoryError(
                    "Created person identifier is invalid."
                )

            person_row = connection.execute(
                f"""
                {_PERSON_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (person_id,),
            ).fetchone()

        return _person_from_row(person_row)

    def get_person(self, person_id: int) -> Person | None:
        """Retourner une personne par identifiant, ou ``None`` si absente."""
        normalized_person_id = _validate_identifier(person_id)

        with self._database.connect() as connection:
            person_row = connection.execute(
                f"""
                {_PERSON_SELECT_COLUMNS}
                WHERE id = ?
                """,
                (normalized_person_id,),
            ).fetchone()

        if person_row is None:
            return None

        return _person_from_row(person_row)

    def find_persons_by_display_name(
        self,
        display_name: str,
    ) -> tuple[Person, ...]:
        """Trouver toutes les correspondances exactes après normalisation.

        La comparaison retire uniquement les espaces extérieurs, normalise
        Unicode en NFC puis applique ``casefold``. Elle ne rapproche jamais
        deux orthographes différentes et ne choisit pas entre des homonymes.
        """
        matching_name = _normalize_display_name_for_matching(display_name)

        with self._database.connect() as connection:
            person_rows = connection.execute(
                f"""
                {_PERSON_SELECT_COLUMNS}
                ORDER BY id
                """
            ).fetchall()

        persons = tuple(
            _person_from_row(person_row)
            for person_row in person_rows
        )

        return tuple(
            person
            for person in persons
            if _normalize_display_name_for_matching(
                person.display_name
            ) == matching_name
        )

    def list_persons(
        self,
        limit: int = DEFAULT_PERSON_RESULT_LIMIT,
    ) -> tuple[Person, ...]:
        """Lister les personnes par identifiant dans un ordre déterministe."""
        normalized_limit = _validate_result_limit(limit)

        with self._database.connect() as connection:
            person_rows = connection.execute(
                f"""
                {_PERSON_SELECT_COLUMNS}
                ORDER BY id
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()

        return tuple(
            _person_from_row(person_row)
            for person_row in person_rows
        )


def _person_from_row(
    person_row: tuple[object, ...] | None,
) -> Person:
    """Convertir une ligne SQLite complète en modèle ``Person`` validé."""
    if person_row is None or len(person_row) != 3:
        raise RepositoryError(
            "Stored person data is incomplete."
        )

    person_id, display_name, created_at = person_row

    try:
        return Person(
            id=cast(int, person_id),
            display_name=cast(str, display_name),
            created_at=_parse_datetime(created_at),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            "Stored person data is invalid."
        ) from error


def _normalize_display_name(display_name: str) -> str:
    """Retourner un nom affiché non vide après normalisation extérieure."""
    if not isinstance(display_name, str):
        raise TypeError(
            "Person display name must be a string."
        )

    normalized_display_name = unicodedata.normalize(
        "NFC",
        display_name,
    ).strip()

    if not normalized_display_name:
        raise ValueError(
            "Person display name cannot be empty."
        )

    return normalized_display_name


def _normalize_display_name_for_matching(display_name: str) -> str:
    """Construire la clé exacte NFC et insensible à la casse d'un nom."""
    return _normalize_display_name(display_name).casefold()


def _validate_identifier(person_id: int) -> int:
    """Retourner un identifiant de personne entier strictement positif."""
    if isinstance(person_id, bool) or not isinstance(person_id, int):
        raise TypeError(
            "Person identifier must be an integer."
        )

    if person_id < 1:
        raise ValueError(
            "Person identifier must be greater than zero."
        )

    return person_id


def _validate_result_limit(limit: int) -> int:
    """Valider une limite positive et bornée de résultats."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Person result limit must be an integer."
        )

    if limit < 1 or limit > MAX_PERSON_RESULT_LIMIT:
        raise ValueError(
            "Person result limit must be between 1 and 100."
        )

    return limit


def _normalize_datetime(value: datetime) -> datetime:
    """Retourner une date-heure consciente normalisée en UTC."""
    if not isinstance(value, datetime):
        raise TypeError(
            "Person creation time must be a datetime."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Person creation time must include timezone information."
        )

    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime:
    """Reconstruire une date-heure persistée au format ISO 8601."""
    if not isinstance(value, str):
        raise TypeError(
            "Stored person creation time must be text."
        )

    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "Stored person creation time is not valid ISO 8601."
        ) from error


def _utc_now() -> datetime:
    """Retourner l'instant courant en UTC avec information de fuseau."""
    return datetime.now(timezone.utc)
