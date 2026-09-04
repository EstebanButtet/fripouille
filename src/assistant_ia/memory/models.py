"""Modèles persistants des tâches, souvenirs, liens et entrées de journal.

Ces dataclasses immuables représentent les lignes SQLite après validation.
Elles centralisent les invariants afin qu'un repository ne puisse pas remettre
au reste de l'application une donnée incomplète ou incohérente. Elles ne
lisent et n'écrivent jamais la base elles-mêmes.
"""

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
    "conversation_analysis",
]

MemoryPersonRole = Literal[
    "subject",
]

ALLOWED_MEMORY_PERSON_ROLES: frozenset[str] = frozenset(
    {
        "subject",
    }
)

ALLOWED_MEMORY_SOURCES: frozenset[str] = frozenset(
    {
        "legacy_explicit",
        "explicit_user",
        "conversation_analysis",
    }
)


@dataclass(frozen=True, slots=True)
class Task:
    """Représenter une tâche persistée et son cycle de vie.

    ``due_at`` est une échéance facultative. ``created_at`` ne change jamais ;
    ``completed_at`` doit être présent exactement lorsque ``status`` vaut
    ``completed``. Toutes les dates-heures sont normalisées en UTC.
    """

    id: int
    title: str
    due_at: datetime | None
    status: TaskStatus
    created_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        """Valider, normaliser et figer les données persistées de la tâche."""
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
    """Représenter un souvenir durable avec sa provenance inspectable.

    ``source`` indique le mécanisme d'enregistrement ; ``source_text`` conserve
    éventuellement la preuve utilisateur exacte. ``confidence`` mesure la
    confiance dans l'extraction/enregistrement, pas la vérité du souvenir.
    ``updated_at`` suit les corrections sans effacer ``created_at``.
    """

    id: int
    content: str
    source: MemorySource
    source_text: str | None
    confidence: float
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """Valider la provenance, les dates et le contenu du souvenir."""
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
class MemoryCandidate:
    """Représenter une proposition validée qui n'est pas encore persistée.

    ``content`` est la formulation candidate, ``source_text`` est un extrait
    exact obligatoire du message utilisateur et ``confidence`` décrit la
    fidélité de l'extraction. Aucun identifiant n'existe avant promotion.
    """

    content: str
    source_text: str
    confidence: float

    def __post_init__(self) -> None:
        """Valider le contenu, la preuve textuelle exacte et la confiance."""
        object.__setattr__(
            self,
            "content",
            _normalize_required_text(
                self.content,
                field_name="Memory candidate content",
            ),
        )
        validated_source_text = _validate_optional_source_text(
            self.source_text
        )

        if validated_source_text is None:
            raise ValueError(
                "Memory candidate source text is required."
            )

        object.__setattr__(
            self,
            "source_text",
            validated_source_text,
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_memory_confidence(self.confidence),
        )


@dataclass(frozen=True, slots=True)
class MemoryPersonLink:
    """Relier explicitement un souvenir à une personne qui en est le sujet.

    Le rôle unique ``subject`` suffit à FRP-IA-04D. Plusieurs personnes
    peuvent être sujets du même souvenir ; aucun lien n'est déduit d'un nom.
    """

    memory_id: int
    person_id: int
    role: MemoryPersonRole = "subject"

    def __post_init__(self) -> None:
        """Valider les deux identifiants et le rôle borné de l'association."""
        object.__setattr__(
            self,
            "memory_id",
            _validate_identifier(self.memory_id),
        )
        object.__setattr__(
            self,
            "person_id",
            _validate_identifier(self.person_id),
        )
        object.__setattr__(
            self,
            "role",
            _normalize_memory_person_role(self.role),
        )


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """Représenter une entrée de journal persistée.

    ``entry_date`` est la date racontée par l'entrée, sans heure ;
    ``created_at`` est l'instant technique UTC de son enregistrement.
    """

    id: int
    content: str
    entry_date: date
    created_at: datetime

    def __post_init__(self) -> None:
        """Valider et normaliser les données persistées du journal."""
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
    """Retourner un identifiant persistant entier strictement positif."""
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
    """Retourner un texte métier non vide après retrait des espaces externes."""
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
    """Retourner un statut de tâche appartenant à la liste fermée."""
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
    """Retourner une provenance mémoire connue et normalisée."""
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


def _normalize_memory_person_role(
    role: MemoryPersonRole,
) -> MemoryPersonRole:
    """Retourner un rôle d'association appartenant à la liste fermée."""
    if not isinstance(role, str):
        raise TypeError("Memory person role must be a string.")

    normalized_role = role.strip()
    if normalized_role not in ALLOWED_MEMORY_PERSON_ROLES:
        raise ValueError(
            f"Unknown memory person role: {normalized_role!r}."
        )

    return cast(MemoryPersonRole, normalized_role)


def _validate_optional_source_text(
    source_text: str | None,
) -> str | None:
    """Valider une preuve utilisateur facultative sans la réécrire.

    Les espaces sont contrôlés mais la chaîne originale est conservée pour que
    ``source_text`` reste un extrait inspectable du message source.
    """
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
    """Retourner la confiance bornée dans le mécanisme d'enregistrement."""
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
    """Retourner une date-heure consciente de son fuseau et normalisée en UTC."""
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
    """Normaliser une date-heure facultative consciente de son fuseau."""
    if value is None:
        return None

    return _normalize_datetime(
        value,
        field_name=field_name,
    )
