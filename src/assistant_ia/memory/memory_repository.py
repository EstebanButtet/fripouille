"""Repository SQLite des souvenirs persistants de Fripouille.

Un repository traduit les opérations métier (« enregistrer », « chercher »,
« corriger », « supprimer ») en SQL paramétré, puis reconstruit des modèles
:class:`Memory` validés. Il est la seule couche de ce domaine autorisée à
connaître les lignes SQLite ; le coeur et la promotion manipulent des objets.
"""

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
    """Enregistrer, rechercher, corriger et supprimer des souvenirs SQLite.

    L'horloge injectable rend les dates déterministes dans les tests. Chaque
    méthode ouvre sa transaction par ``SQLiteDatabase.connect`` et retourne le
    modèle relu depuis la base, pas seulement les valeurs demandées.
    """

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Créer le repository avec une base et une horloge injectables."""
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
        """Persister un souvenir demandé explicitement et le retourner.

        Cette voie marque la provenance ``explicit_user`` avec une confiance
        de mécanisme égale à 1 ; elle est distincte de l'analyse automatique.
        """
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

        # L'insertion et la relecture appartiennent à la même transaction : le
        # modèle retourné correspond exactement à la ligne créée.
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
        """Persister un candidat conversationnel explicitement confirmé.

        La preuve et la confiance calculées pendant l'analyse sont conservées
        avec la provenance ``conversation_analysis``.
        """
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
        """Remplacer le contenu d'un souvenir par un candidat confirmé.

        L'identifiant et la date de création sont conservés. Si l'horloge ne
        progresse pas, une microseconde est ajoutée pour garantir que
        ``updated_at`` matérialise réellement la correction.
        """
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
        """Chercher littéralement un texte dans les souvenirs.

        Les caractères joker de ``LIKE`` sont échappés : ``%`` et ``_`` saisis
        par l'utilisateur restent des caractères à chercher, pas du SQL.
        """
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
        """Retourner une vue récente bornée destinée aux traitements locaux."""
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
        """Supprimer exactement un souvenir identifié et retourner son état.

        La ligne est lue avant suppression dans la même transaction, ce qui
        permet à l'action de décrire ce qui a réellement été retiré.
        """
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
    """Convertir une ligne SQLite complète en modèle ``Memory`` validé."""
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


def _validate_identifier(memory_id: int) -> int:
    """Retourner un identifiant mémoire entier strictement positif."""
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
    """Valider une limite positive pour les résultats de recherche."""
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
    """Valider une limite positive pour la vue locale des souvenirs."""
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
    """Échapper les jokers ``LIKE`` afin d'effectuer une recherche littérale."""
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
