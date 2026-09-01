"""Personne supposée présente par défaut au début d'une session.

Cette valeur locale n'est pas un profil relationnel appris ; elle sert
seulement à initialiser :class:`ActivePersonContext`.
"""

from __future__ import annotations

from assistant_ia.people.models import PersonProfile


DEFAULT_PERSON_NAME = "Este"


def build_default_person() -> PersonProfile:
    """Construire la personne supposée utiliser l'assistant par défaut."""
    return PersonProfile(
        name=DEFAULT_PERSON_NAME,
    )
