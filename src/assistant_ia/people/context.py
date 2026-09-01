"""Contexte de la personne active pendant une session conversationnelle.

Le contexte commence avec une personne par défaut, peut basculer après une
présentation explicite détectée par l'application, puis revient à la valeur par
défaut lors d'une réinitialisation. Il n'effectue aucune persistance et refuse
que le nom réservé de Fripouille devienne celui de l'utilisateur.
"""

from __future__ import annotations

from assistant_ia.people.models import PersonProfile


class ActivePersonContext:
    """Suivre le locuteur courant sans créer de profil social durable."""

    def __init__(
        self,
        *,
        assistant_name: str,
        default_person: PersonProfile,
    ) -> None:
        """Créer le contexte avec le nom réservé et la personne par défaut."""
        if not isinstance(assistant_name, str):
            raise TypeError("Assistant name must be a string.")

        normalized_assistant_name = assistant_name.strip()

        if not normalized_assistant_name:
            raise ValueError("Assistant name must not be empty.")

        if not isinstance(default_person, PersonProfile):
            raise TypeError(
                "Default person must be a PersonProfile."
            )

        self._assistant_name = normalized_assistant_name
        self._default_person = default_person
        self._validate_person_name(default_person)
        self._active_person = default_person

    @property
    def assistant_name(self) -> str:
        """Retourner le nom réservé exclusivement à l'assistant."""
        return self._assistant_name

    @property
    def default_person(self) -> PersonProfile:
        """Retourner la personne restaurée au début d'une conversation."""
        return self._default_person

    @property
    def active_person(self) -> PersonProfile:
        """Retourner la personne actuellement considérée comme locuteur."""
        return self._active_person

    def set_active_person(
        self,
        person: PersonProfile,
    ) -> None:
        """Changer la personne active après une présentation validée."""
        if not isinstance(person, PersonProfile):
            raise TypeError(
                "Active person must be a PersonProfile."
            )

        self._validate_person_name(person)
        self._active_person = person

    def reset(self) -> None:
        """Restaurer la personne par défaut pour une nouvelle conversation."""
        self._active_person = self._default_person

    def _validate_person_name(
        self,
        person: PersonProfile,
    ) -> None:
        """Interdire qu'un profil de personne emprunte le nom de l'assistant."""
        if (
            person.name.casefold()
            == self._assistant_name.casefold()
        ):
            raise ValueError(
                "The assistant name is reserved exclusively "
                "for the assistant."
            )
