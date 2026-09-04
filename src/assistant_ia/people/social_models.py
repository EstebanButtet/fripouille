"""Modèles sociaux persistants distincts de l'identité et des faits confirmés."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal

RelationshipFamiliarity = Literal["new", "known", "familiar", "close"]
RelationshipInteractionStyle = Literal[
    "neutral", "direct", "warm", "playful", "formal"
]
ObservationCategory = Literal[
    "communication", "preference", "habit", "behavior", "context"
]
ObservationSource = Literal["manual_entry", "conversation_analysis"]
ObservationStatus = Literal["unconfirmed"]

ALLOWED_RELATIONSHIP_FAMILIARITIES = frozenset(
    {"new", "known", "familiar", "close"}
)
ALLOWED_RELATIONSHIP_INTERACTION_STYLES = frozenset(
    {"neutral", "direct", "warm", "playful", "formal"}
)
ALLOWED_OBSERVATION_CATEGORIES = frozenset(
    {"communication", "preference", "habit", "behavior", "context"}
)
ALLOWED_OBSERVATION_SOURCES = frozenset(
    {"manual_entry", "conversation_analysis"}
)
ALLOWED_OBSERVATION_STATUSES = frozenset({"unconfirmed"})


@dataclass(frozen=True, slots=True)
class PersonRelationship:
    """Décrire la relation conversationnelle avec une personne persistante.

    Ces dimensions guident uniquement la conversation. Elles ne représentent
    ni l'identité de Fripouille, ni une autorité, ni un niveau de sécurité.
    """

    person_id: int
    familiarity: RelationshipFamiliarity
    interaction_style: RelationshipInteractionStyle
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        created_at = _normalize_datetime(
            self.created_at, field_name="Relationship creation time"
        )
        updated_at = _normalize_datetime(
            self.updated_at, field_name="Relationship update time"
        )
        if updated_at < created_at:
            raise ValueError(
                "Relationship update time cannot precede creation time."
            )
        object.__setattr__(self, "person_id", _validate_identifier(self.person_id))
        object.__setattr__(
            self,
            "familiarity",
            _normalize_choice(
                self.familiarity,
                ALLOWED_RELATIONSHIP_FAMILIARITIES,
                field_name="relationship familiarity",
            ),
        )
        object.__setattr__(
            self,
            "interaction_style",
            _normalize_choice(
                self.interaction_style,
                ALLOWED_RELATIONSHIP_INTERACTION_STYLES,
                field_name="relationship interaction style",
            ),
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


@dataclass(frozen=True, slots=True)
class Observation:
    """Représenter un signal social non confirmé concernant une personne."""

    id: int
    person_id: int
    category: ObservationCategory
    content: str
    source: ObservationSource
    source_text: str | None
    confidence: float
    status: ObservationStatus
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _validate_identifier(self.id))
        object.__setattr__(self, "person_id", _validate_identifier(self.person_id))
        object.__setattr__(
            self,
            "category",
            _normalize_choice(
                self.category,
                ALLOWED_OBSERVATION_CATEGORIES,
                field_name="observation category",
            ),
        )
        object.__setattr__(
            self, "content", _normalize_required_text(self.content, "Observation content")
        )
        source = _normalize_choice(
            self.source,
            ALLOWED_OBSERVATION_SOURCES,
            field_name="observation source",
        )
        source_text = _normalize_optional_text(self.source_text)
        if source == "conversation_analysis" and source_text is None:
            raise ValueError("Analyzed observations require exact source text.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "confidence", _normalize_confidence(self.confidence))
        object.__setattr__(
            self,
            "status",
            _normalize_choice(
                self.status,
                ALLOWED_OBSERVATION_STATUSES,
                field_name="observation status",
            ),
        )
        object.__setattr__(
            self,
            "created_at",
            _normalize_datetime(
                self.created_at, field_name="Observation creation time"
            ),
        )


def _validate_identifier(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Persistent identifier must be an integer.")
    if value < 1:
        raise ValueError("Persistent identifier must be greater than zero.")
    return value


def _normalize_choice(value: object, allowed: frozenset[str], *, field_name: str):
    if not isinstance(value, str):
        raise TypeError(f"{field_name.capitalize()} must be a string.")
    normalized = value.strip()
    if normalized not in allowed:
        raise ValueError(f"Unknown {field_name}: {normalized!r}.")
    return normalized


def _normalize_required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _normalize_required_text(value, "Observation source text")


def _normalize_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Observation confidence must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("Observation confidence must be between zero and one.")
    return normalized


def _normalize_datetime(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)
