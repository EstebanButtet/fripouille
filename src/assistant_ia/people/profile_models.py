"""Modèles distincts des faits de profil confirmés et de leurs candidats.

Une ``Person`` identifie un sujet persistant. Un ``ProfileFactCandidate`` est
une proposition non confirmée, rattachée par l'application à ce sujet. Seul
un ``ProfileFact`` représente une donnée de profil effectivement persistée.
Ces objets ne sont ni des souvenirs généraux, ni des relations sociales.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal, cast

ProfileFactCategory = Literal[
    "preference",
    "communication_preference",
    "interest",
    "habit",
    "personal_fact",
]

ALLOWED_PROFILE_FACT_CATEGORIES: frozenset[str] = frozenset(
    {
        "preference",
        "communication_preference",
        "interest",
        "habit",
        "personal_fact",
    }
)

ProfileFactSource = Literal[
    "explicit_user",
    "conversation_analysis",
]

ALLOWED_PROFILE_FACT_SOURCES: frozenset[str] = frozenset(
    {
        "explicit_user",
        "conversation_analysis",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileFact:
    """Représenter un fait de profil confirmé, traçable et corrigeable."""

    id: int
    person_id: int
    category: ProfileFactCategory
    content: str
    source: ProfileFactSource
    source_text: str | None
    confidence: float
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """Valider les identifiants, la catégorie, la preuve et les dates."""
        created_at = _normalize_datetime(
            self.created_at,
            field_name="Profile fact creation time",
        )
        updated_at = _normalize_datetime(
            self.updated_at,
            field_name="Profile fact update time",
        )
        if updated_at < created_at:
            raise ValueError(
                "Profile fact update time cannot precede creation time."
            )

        object.__setattr__(self, "id", _validate_identifier(self.id, "Fact"))
        object.__setattr__(
            self,
            "person_id",
            _validate_identifier(self.person_id, "Person"),
        )
        object.__setattr__(self, "category", _normalize_category(self.category))
        object.__setattr__(
            self,
            "content",
            _normalize_required_text(self.content, "Profile fact content"),
        )
        normalized_source = _normalize_source(self.source)
        normalized_source_text = _normalize_optional_text(self.source_text)
        if (
            normalized_source == "conversation_analysis"
            and normalized_source_text is None
        ):
            raise ValueError(
                "Analyzed profile facts require their exact source text."
            )
        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(
            self,
            "source_text",
            normalized_source_text,
        )
        object.__setattr__(self, "confidence", _normalize_confidence(self.confidence))
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


@dataclass(frozen=True, slots=True)
class ProfileFactCandidate:
    """Porter une proposition non confirmée visant une personne résolue.

    ``confidence`` mesure uniquement la fidélité et l'admissibilité de
    l'extraction. Une confiance élevée ne transforme jamais la proposition en
    vérité et n'autorise aucune écriture.
    """

    person_id: int
    category: ProfileFactCategory
    content: str
    source_text: str
    confidence: float

    def __post_init__(self) -> None:
        """Valider le sujet applicatif et les données proposées."""
        object.__setattr__(
            self,
            "person_id",
            _validate_identifier(self.person_id, "Person"),
        )
        object.__setattr__(self, "category", _normalize_category(self.category))
        object.__setattr__(
            self,
            "content",
            _normalize_required_text(self.content, "Profile candidate content"),
        )
        object.__setattr__(
            self,
            "source_text",
            _normalize_required_text(
                self.source_text,
                "Profile candidate source text",
            ),
        )
        object.__setattr__(self, "confidence", _normalize_confidence(self.confidence))


def _validate_identifier(value: int, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} identifier must be an integer.")
    if value < 1:
        raise ValueError(f"{subject} identifier must be greater than zero.")
    return value


def _normalize_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_required_text(value, "Profile fact source text")


def _normalize_category(value: ProfileFactCategory) -> ProfileFactCategory:
    if not isinstance(value, str):
        raise TypeError("Profile fact category must be a string.")
    normalized = value.strip()
    if normalized not in ALLOWED_PROFILE_FACT_CATEGORIES:
        raise ValueError(f"Unknown profile fact category: {normalized!r}.")
    return cast(ProfileFactCategory, normalized)


def _normalize_source(value: ProfileFactSource) -> ProfileFactSource:
    if not isinstance(value, str):
        raise TypeError("Profile fact source must be a string.")
    normalized = value.strip()
    if normalized not in ALLOWED_PROFILE_FACT_SOURCES:
        raise ValueError(f"Unknown profile fact source: {normalized!r}.")
    return cast(ProfileFactSource, normalized)


def _normalize_confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Profile fact confidence must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("Profile fact confidence must be between zero and one.")
    return normalized


def _normalize_datetime(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)
