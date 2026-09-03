"""Personne supposée présente par défaut au début d'une session.

L'identifiant réservé est la clé stable du registre persistant. Le profil
local reste sans données relationnelles et sert à initialiser
:class:`ActivePersonContext`.
"""

from __future__ import annotations

from assistant_ia.people.models import PersonProfile


DEFAULT_PERSON_ID = 1
DEFAULT_PERSON_NAME = "Este"


def build_default_person() -> PersonProfile:
    """Construire la personne supposée utiliser l'assistant par défaut."""
    return PersonProfile(
        name=DEFAULT_PERSON_NAME,
    )
