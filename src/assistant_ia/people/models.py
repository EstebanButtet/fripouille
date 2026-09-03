"""Modèles minimaux d'une personne persistante et de sa vue de session.

``Person`` porte l'identité stable du registre FRP-IA-04A. ``PersonProfile``
reste la projection légère utilisée par la conversation courante ; il ne
constitue ni un profil social détaillé ni une relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Person:
    """Représenter l'identité minimale d'une personne persistée."""

    id: int
    display_name: str
    created_at: datetime

    def __post_init__(self) -> None:
        """Valider et normaliser les données issues du registre."""
        if isinstance(self.id, bool) or not isinstance(self.id, int):
            raise TypeError("Person identifier must be an integer.")

        if self.id < 1:
            raise ValueError(
                "Person identifier must be greater than zero."
            )

        if not isinstance(self.display_name, str):
            raise TypeError("Person display name must be a string.")

        normalized_display_name = self.display_name.strip()

        if not normalized_display_name:
            raise ValueError("Person display name must not be empty.")

        if not isinstance(self.created_at, datetime):
            raise TypeError("Person creation time must be a datetime.")

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError(
                "Person creation time must include timezone information."
            )

        object.__setattr__(
            self,
            "display_name",
            normalized_display_name,
        )
        object.__setattr__(
            self,
            "created_at",
            self.created_at.astimezone(timezone.utc),
        )


@dataclass(frozen=True, slots=True)
class PersonProfile:
    """Porter le nom de session de la personne qui interagit actuellement."""

    name: str

    def __post_init__(self) -> None:
        """Valider le type et normaliser les espaces extérieurs du nom."""
        if not isinstance(self.name, str):
            raise TypeError("Person name must be a string.")

        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Person name must not be empty.")

        object.__setattr__(
            self,
            "name",
            normalized_name,
        )
