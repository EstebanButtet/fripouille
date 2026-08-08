"""Structured assistant identity domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


BehaviorLevel = Literal[
    "low",
    "moderate",
    "high",
]

GrammaticalGender = Literal[
    "masculine",
    "feminine",
    "neutral",
]

ALLOWED_BEHAVIOR_LEVELS: frozenset[str] = frozenset(
    {
        "low",
        "moderate",
        "high",
    }
)

ALLOWED_GRAMMATICAL_GENDERS: frozenset[str] = frozenset(
    {
        "masculine",
        "feminine",
        "neutral",
    }
)


@dataclass(frozen=True, slots=True)
class AssistantIdentity:
    """Represent the stable identity of the assistant."""

    name: str
    role: str
    relationship_to_user: str
    grammatical_gender: GrammaticalGender
    traits: tuple[str, ...]
    communication_style: tuple[str, ...]
    humor_level: BehaviorLevel
    initiative_level: BehaviorLevel
    curiosity_level: BehaviorLevel
    behavioral_rules: tuple[str, ...]
    boundaries: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the identity configuration."""
        object.__setattr__(
            self,
            "name",
            _normalize_text(
                self.name,
                field_name="name",
            ),
        )
        object.__setattr__(
            self,
            "role",
            _normalize_text(
                self.role,
                field_name="role",
            ),
        )
        object.__setattr__(
            self,
            "relationship_to_user",
            _normalize_text(
                self.relationship_to_user,
                field_name="relationship_to_user",
            ),
        )
        object.__setattr__(
            self,
            "grammatical_gender",
            _normalize_grammatical_gender(
                self.grammatical_gender,
            ),
        )

        for field_name in (
            "traits",
            "communication_style",
            "behavioral_rules",
            "boundaries",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        for field_name in (
            "humor_level",
            "initiative_level",
            "curiosity_level",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_behavior_level(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )


def _normalize_text(
    value: object,
    *,
    field_name: str,
) -> str:
    """Validate and normalize one required identity text value."""
    if not isinstance(value, str):
        raise TypeError(
            f"Assistant identity {field_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"Assistant identity {field_name} cannot be empty."
        )

    return normalized_value


def _normalize_grammatical_gender(
    value: object,
) -> GrammaticalGender:
    """Validate and normalize the assistant grammatical gender."""
    if not isinstance(value, str):
        raise TypeError(
            "Assistant identity grammatical_gender must be a string."
        )

    normalized_value = value.strip()

    if normalized_value not in ALLOWED_GRAMMATICAL_GENDERS:
        raise ValueError(
            "Unknown assistant identity grammatical_gender: "
            f"{normalized_value!r}."
        )

    return cast(GrammaticalGender, normalized_value)


def _normalize_text_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate and normalize one immutable identity text collection."""
    if not isinstance(value, tuple):
        raise TypeError(
            f"Assistant identity {field_name} must be a tuple."
        )

    if not value:
        raise ValueError(
            f"Assistant identity {field_name} cannot be empty."
        )

    normalized_values: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise TypeError(
                f"Assistant identity {field_name} items must be strings."
            )

        normalized_item = item.strip()

        if not normalized_item:
            raise ValueError(
                f"Assistant identity {field_name} items cannot be empty."
            )

        normalized_values.append(normalized_item)

    return tuple(normalized_values)


def _normalize_behavior_level(
    value: object,
    *,
    field_name: str,
) -> BehaviorLevel:
    """Validate and normalize one categorical behavior level."""
    if not isinstance(value, str):
        raise TypeError(
            f"Assistant identity {field_name} must be a string."
        )

    normalized_value = value.strip()

    if normalized_value not in ALLOWED_BEHAVIOR_LEVELS:
        raise ValueError(
            f"Unknown assistant identity {field_name}: "
            f"{normalized_value!r}."
        )

    return cast(BehaviorLevel, normalized_value)
