"""Modèle minimal d'une personne reconnue dans la session courante.

Le terme ``Profile`` désigne ici uniquement un nom non vide en mémoire. Aucun
profil relationnel persistant n'est implémenté à ce stade (FRP-IA-04 reste
futur).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonProfile:
    """Porter le nom minimal de la personne qui interagit avec l'assistant."""

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
