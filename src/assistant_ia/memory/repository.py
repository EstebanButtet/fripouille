"""Infrastructure SQLite partagée par les repositories métier.

``SQLiteDatabase`` possède le chemin, ouvre des connexions transactionnelles,
active les clés étrangères et gère les migrations de schéma. Les repositories
de tâches, mémoire, journal, personnes, faits de profil et associations
utilisent cette infrastructure mais gardent leur SQL métier dans leurs propres
modules.

Une transaction couvre chaque bloc ``with database.connect()`` : une sortie
normale valide les écritures, tandis qu'une exception provoque leur annulation
par le gestionnaire de contexte SQLite avant fermeture.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Iterator

from assistant_ia.people.defaults import (
    DEFAULT_PERSON_ID,
    DEFAULT_PERSON_NAME,
)

DEFAULT_DATABASE_DIRECTORY_NAME = "assistant-ia"
DEFAULT_DATABASE_FILENAME = "assistant_ia.db"

_INITIAL_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = 7

_SCHEMA_VERSION_COLUMNS = (
    ("id", "INTEGER", 0, 1),
    ("version", "INTEGER", 1, 0),
)

_BUSINESS_TABLE_COLUMNS = {
    "tasks": (
        ("id", "INTEGER", 0, 1),
        ("title", "TEXT", 1, 0),
        ("due_at", "TEXT", 0, 0),
        ("status", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("completed_at", "TEXT", 0, 0),
    ),
    "memories": (
        ("id", "INTEGER", 0, 1),
        ("content", "TEXT", 1, 0),
        ("source", "TEXT", 1, 0),
        ("source_text", "TEXT", 0, 0),
        ("confidence", "REAL", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    ),
    "journal_entries": (
        ("id", "INTEGER", 0, 1),
        ("content", "TEXT", 1, 0),
        ("entry_date", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
    ),
    "persons": (
        ("id", "INTEGER", 0, 1),
        ("display_name", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
    ),
    "profile_facts": (
        ("id", "INTEGER", 0, 1),
        ("person_id", "INTEGER", 1, 0),
        ("category", "TEXT", 1, 0),
        ("content", "TEXT", 1, 0),
        ("source", "TEXT", 1, 0),
        ("source_text", "TEXT", 0, 0),
        ("confidence", "REAL", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    ),
    "memory_people": (
        ("memory_id", "INTEGER", 1, 1),
        ("person_id", "INTEGER", 1, 2),
        ("role", "TEXT", 1, 3),
    ),
    "person_relationships": (
        ("person_id", "INTEGER", 0, 1),
        ("familiarity", "TEXT", 1, 0),
        ("interaction_style", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    ),
    "person_observations": (
        ("id", "INTEGER", 0, 1),
        ("person_id", "INTEGER", 1, 0),
        ("category", "TEXT", 1, 0),
        ("content", "TEXT", 1, 0),
        ("source", "TEXT", 1, 0),
        ("source_text", "TEXT", 0, 0),
        ("confidence", "REAL", 1, 0),
        ("status", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
    ),
}


class DatabaseError(RuntimeError):
    """Traduire un échec technique SQLite en erreur stable de persistance."""


def default_database_path() -> Path:
    """Construire le chemin SQLite local propre à l'utilisateur Windows.

    La variable ``LOCALAPPDATA`` choisit l'emplacement ; le dossier n'est créé
    qu'à l'ouverture d'une connexion.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data is None or not local_app_data.strip():
        raise DatabaseError("LOCALAPPDATA is not configured.")

    return _normalize_database_path(
        Path(local_app_data)
        / DEFAULT_DATABASE_DIRECTORY_NAME
        / DEFAULT_DATABASE_FILENAME
    )


class SQLiteDatabase:
    """Gérer le chemin, les connexions et le schéma d'une base SQLite locale.

    L'objet est léger et peut être partagé par plusieurs repositories. Il ne
    garde pas une connexion ouverte entre les opérations.
    """

    def __init__(self, database_path: str | PathLike[str]) -> None:
        """Créer le gestionnaire après normalisation du chemin de fichier."""
        self._path = _normalize_database_path(database_path)

    @property
    def path(self) -> Path:
        """Retourner le chemin absolu normalisé de la base."""
        return self._path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Ouvrir une connexion transactionnelle configurée puis la fermer.

        La valeur produite par ``yield`` est disponible dans le bloc ``with``.
        Les erreurs SQLite sont enveloppées dans ``DatabaseError`` et la
        fermeture est garantie par ``finally``, même en cas d'exception.
        """
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

            # Le gestionnaire de contexte natif valide ou annule la transaction
            # sans transférer cette responsabilité aux repositories appelants.
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
        """Créer, migrer puis valider le schéma dans une même transaction.

        Les migrations sont appliquées une par une selon la version enregistrée.
        Une base plus récente que le code ou une structure inattendue est
        refusée afin d'éviter une utilisation partielle des données.
        """
        with self.connect() as connection:
            connection.execute("BEGIN")

            _create_technical_schema(connection)
            schema_version = _read_schema_version(connection)

            if schema_version > CURRENT_SCHEMA_VERSION:
                raise DatabaseError(
                    "Unsupported database schema version."
                )

            # Chaque migration ne modifie la version qu'après avoir réussi.
            # Toute exception annule donc aussi bien le SQL que le compteur.
            while schema_version < CURRENT_SCHEMA_VERSION:
                migration = _SCHEMA_MIGRATIONS.get(schema_version)

                if migration is None:
                    raise DatabaseError(
                        "Unsupported database schema version."
                    )

                migration(connection)

                next_version = schema_version + 1
                update_cursor = connection.execute(
                    """
                    UPDATE schema_version
                    SET version = ?
                    WHERE id = ?
                      AND version = ?
                    """,
                    (
                        next_version,
                        1,
                        schema_version,
                    ),
                )

                if update_cursor.rowcount != 1:
                    raise DatabaseError(
                        "Database schema version could not be updated."
                    )

                schema_version = next_version

            _validate_current_business_schema(connection)
            _ensure_default_person(connection)


def _create_technical_schema(
    connection: sqlite3.Connection,
) -> None:
    """Créer puis valider la table technique de version du schéma."""
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
        (
            1,
            _INITIAL_SCHEMA_VERSION,
        ),
    )

    if _table_columns(connection, "schema_version") != (
        _SCHEMA_VERSION_COLUMNS
    ):
        raise DatabaseError(
            "Malformed database schema metadata."
        )


def _read_schema_version(
    connection: sqlite3.Connection,
) -> int:
    """Lire l'unique ligne de version après validation de sa forme."""
    version_rows = connection.execute(
        """
        SELECT id, version
        FROM schema_version
        ORDER BY id
        """
    ).fetchall()

    if len(version_rows) != 1:
        raise DatabaseError(
            "Malformed database schema metadata."
        )

    row_id, schema_version = version_rows[0]

    if (
        row_id != 1
        or not isinstance(schema_version, int)
        or schema_version < _INITIAL_SCHEMA_VERSION
    ):
        raise DatabaseError(
            "Malformed database schema metadata."
        )

    return schema_version


def _migrate_schema_1_to_2(
    connection: sqlite3.Connection,
) -> None:
    """Ajouter les premières tables métier de tâches, mémoire et journal."""
    connection.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL
                CHECK (length(trim(title)) > 0),
            due_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'completed')),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            completed_at TEXT,
            CHECK (
                (
                    status = 'pending'
                    AND completed_at IS NULL
                )
                OR
                (
                    status = 'completed'
                    AND completed_at IS NOT NULL
                )
            )
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE journal_entries (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            entry_date TEXT NOT NULL
                CHECK (length(trim(entry_date)) > 0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0)
        )
        """
    )


def _migrate_schema_2_to_3(
    connection: sqlite3.Connection,
) -> None:
    """Ajouter une provenance inspectable à chaque souvenir existant.

    SQLite ne permettant pas ici toutes les contraintes via de simples ajouts
    de colonnes, la table est reconstruite. Un comptage avant/après garantit
    que la migration n'a perdu aucune ligne.
    """
    source_count_row = connection.execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()

    if source_count_row is None:
        raise DatabaseError(
            "Existing memories could not be counted."
        )

    source_count = source_count_row[0]

    connection.execute(
        "ALTER TABLE memories RENAME TO memories_v2"
    )
    connection.execute(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            source TEXT NOT NULL
                CHECK (length(trim(source)) > 0),
            source_text TEXT
                CHECK (
                    source_text IS NULL
                    OR length(trim(source_text)) > 0
                ),
            confidence REAL NOT NULL
                CHECK (confidence >= 0.0 AND confidence <= 1.0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            updated_at TEXT NOT NULL
                CHECK (length(trim(updated_at)) > 0)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO memories (
            id,
            content,
            source,
            source_text,
            confidence,
            created_at,
            updated_at
        )
        SELECT
            id,
            content,
            'legacy_explicit',
            NULL,
            1.0,
            created_at,
            created_at
        FROM memories_v2
        """
    )

    target_count_row = connection.execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()

    if (
        target_count_row is None
        or target_count_row[0] != source_count
    ):
        raise DatabaseError(
            "Existing memories were not fully preserved."
        )

    connection.execute(
        "DROP TABLE memories_v2"
    )


def _migrate_schema_3_to_4(
    connection: sqlite3.Connection,
) -> None:
    """Ajouter le registre minimal des personnes sans liaison sociale."""
    connection.execute(
        """
        CREATE TABLE persons (
            id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL
                CHECK (length(trim(display_name)) > 0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0)
        )
        """
    )


def _migrate_schema_4_to_5(
    connection: sqlite3.Connection,
) -> None:
    """Ajouter les faits de profil sans relier les souvenirs aux personnes."""
    connection.execute(
        """
        CREATE TABLE profile_facts (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL
                REFERENCES persons(id),
            category TEXT NOT NULL
                CHECK (category IN (
                    'preference',
                    'communication_preference',
                    'interest',
                    'habit',
                    'personal_fact'
                )),
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            source TEXT NOT NULL
                CHECK (source IN (
                    'explicit_user',
                    'conversation_analysis'
                )),
            source_text TEXT
                CHECK (
                    source_text IS NULL
                    OR length(trim(source_text)) > 0
                ),
            confidence REAL NOT NULL
                CHECK (confidence >= 0.0 AND confidence <= 1.0),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            updated_at TEXT NOT NULL
                CHECK (length(trim(updated_at)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_profile_facts_person_category
        ON profile_facts (person_id, category, id)
        """
    )


def _migrate_schema_5_to_6(
    connection: sqlite3.Connection,
) -> None:
    """Relier explicitement mémoires et personnes sans inventer de sujet."""
    connection.execute(
        """
        CREATE TABLE memory_people (
            memory_id INTEGER NOT NULL
                REFERENCES memories(id) ON DELETE CASCADE,
            person_id INTEGER NOT NULL
                REFERENCES persons(id) ON DELETE CASCADE,
            role TEXT NOT NULL
                CHECK (role IN ('subject')),
            PRIMARY KEY (memory_id, person_id, role)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_memory_people_person_id
        ON memory_people (person_id, role, memory_id)
        """
    )


def _migrate_schema_6_to_7(
    connection: sqlite3.Connection,
) -> None:
    """Ajouter relations et observations sans inventer de données sociales."""
    connection.execute(
        """
        CREATE TABLE person_relationships (
            person_id INTEGER PRIMARY KEY
                REFERENCES persons(id) ON DELETE CASCADE,
            familiarity TEXT NOT NULL
                CHECK (familiarity IN ('new', 'known', 'familiar', 'close')),
            interaction_style TEXT NOT NULL
                CHECK (interaction_style IN (
                    'neutral', 'direct', 'warm', 'playful', 'formal'
                )),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            updated_at TEXT NOT NULL
                CHECK (length(trim(updated_at)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE person_observations (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL
                REFERENCES persons(id) ON DELETE CASCADE,
            category TEXT NOT NULL
                CHECK (category IN (
                    'communication', 'preference', 'habit', 'behavior', 'context'
                )),
            content TEXT NOT NULL
                CHECK (length(trim(content)) > 0),
            source TEXT NOT NULL
                CHECK (source IN ('manual_entry', 'conversation_analysis')),
            source_text TEXT
                CHECK (source_text IS NULL OR length(trim(source_text)) > 0),
            confidence REAL NOT NULL
                CHECK (confidence >= 0.0 AND confidence <= 1.0),
            status TEXT NOT NULL
                CHECK (status IN ('unconfirmed')),
            created_at TEXT NOT NULL
                CHECK (length(trim(created_at)) > 0),
            CHECK (source != 'conversation_analysis' OR source_text IS NOT NULL)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_person_observations_person_created
        ON person_observations (person_id, created_at DESC, id DESC)
        """
    )


def _ensure_default_person(
    connection: sqlite3.Connection,
) -> None:
    """Garantir l'identité persistante par défaut sans doublon de lancement."""
    connection.execute(
        """
        INSERT OR IGNORE INTO persons (
            id,
            display_name,
            created_at
        )
        VALUES (
            ?,
            ?,
            strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
        )
        """,
        (
            DEFAULT_PERSON_ID,
            DEFAULT_PERSON_NAME,
        ),
    )

    default_person_row = connection.execute(
        """
        SELECT display_name
        FROM persons
        WHERE id = ?
        """,
        (DEFAULT_PERSON_ID,),
    ).fetchone()

    if default_person_row != (DEFAULT_PERSON_NAME,):
        raise DatabaseError(
            "Default person registry entry is invalid."
        )


def _validate_current_business_schema(
    connection: sqlite3.Connection,
) -> None:
    """Comparer les structures des tables métier à la forme attendue."""
    for table_name, expected_columns in _BUSINESS_TABLE_COLUMNS.items():
        if _table_columns(connection, table_name) != expected_columns:
            raise DatabaseError(
                "Malformed database business schema."
            )


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple[str, str, int, int], ...]:
    """Retourner la structure normalisée des colonnes d'une table interne."""
    column_rows = connection.execute(
        """
        SELECT
            name,
            upper(type),
            "notnull",
            pk
        FROM pragma_table_info(?)
        ORDER BY cid
        """,
        (table_name,),
    ).fetchall()

    return tuple(
        (
            column_name,
            column_type,
            not_null,
            primary_key,
        )
        for (
            column_name,
            column_type,
            not_null,
            primary_key,
        ) in column_rows
    )


_SCHEMA_MIGRATIONS = {
    1: _migrate_schema_1_to_2,
    2: _migrate_schema_2_to_3,
    3: _migrate_schema_3_to_4,
    4: _migrate_schema_4_to_5,
    5: _migrate_schema_5_to_6,
    6: _migrate_schema_6_to_7,
}


def _normalize_database_path(
    database_path: str | PathLike[str],
) -> Path:
    """Valider puis résoudre un chemin destiné à un fichier SQLite."""
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
