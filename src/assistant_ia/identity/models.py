"""Modèle métier structuré de l'identité stable de l'assistant.

``AssistantIdentity`` contient les repères déclaratifs qui donnent à
Fripouille son nom, son rôle et son style. Ce modèle est immuable et n'est
jamais mis à jour depuis une réponse du LLM. Il ne représente ni des souvenirs,
ni une relation sociale évolutive, ni un apprentissage comportemental.
"""

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
    """Représenter la configuration stable de Fripouille.

    Les tuples décrivent des collections ordonnées de traits ou règles. Les
    niveaux ``humor_level``, ``initiative_level`` et ``curiosity_level`` sont
    des catégories de configuration, pas des mesures d'un état interne vivant.
    ``grammatical_gender`` sert au rendu linguistique de l'assistant.
    """

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
        """Valider et normaliser chaque partie de la configuration immuable."""
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
    """Valider et normaliser un texte obligatoire de l'identité."""
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
    """Valider le genre grammatical contre la liste fermée."""
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
    """Valider une collection non vide de textes d'identité."""
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
    """Valider un niveau comportemental appartenant aux catégories connues."""
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
