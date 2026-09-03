"""Contexte de la personne active pendant une session conversationnelle.

Le contexte commence avec une personne par défaut, peut être relié à une
identité persistante résolue par l'application, puis revient à la valeur par
défaut lors d'une réinitialisation. Il n'effectue lui-même aucune persistance
et refuse que le nom réservé de Fripouille devienne celui de l'utilisateur.
"""

from __future__ import annotations

from assistant_ia.people.models import Person, PersonProfile


class ActivePersonContext:
    """Suivre le locuteur courant sans créer de profil social durable."""

    def __init__(
        self,
        *,
        assistant_name: str,
        default_person: Person | PersonProfile,
    ) -> None:
        """Créer le contexte avec le nom réservé et la personne par défaut."""
        if not isinstance(assistant_name, str):
            raise TypeError("Assistant name must be a string.")

        normalized_assistant_name = assistant_name.strip()

        if not normalized_assistant_name:
            raise ValueError("Assistant name must not be empty.")

        if isinstance(default_person, Person):
            resolved_default_person = PersonProfile(
                name=default_person.display_name
            )
            resolved_default_person_id = default_person.id
        elif isinstance(default_person, PersonProfile):
            resolved_default_person = default_person
            resolved_default_person_id = None
        else:
            raise TypeError(
                "Default person must be a Person or PersonProfile."
            )

        self._assistant_name = normalized_assistant_name
        self._default_person = resolved_default_person
        self._default_person_id = resolved_default_person_id
        self._validate_person_name(resolved_default_person)
        self._active_person = resolved_default_person
        self._active_person_id = self._default_person_id

    @property
    def assistant_name(self) -> str:
        """Retourner le nom réservé exclusivement à l'assistant."""
        return self._assistant_name

    @property
    def default_person(self) -> PersonProfile:
        """Retourner la personne restaurée au début d'une conversation."""
        return self._default_person

    @property
    def default_person_id(self) -> int | None:
        """Retourner l'identifiant persistant par défaut lorsqu'il est connu."""
        return self._default_person_id

    @property
    def active_person(self) -> PersonProfile:
        """Retourner la personne actuellement considérée comme locuteur."""
        return self._active_person

    @property
    def active_person_id(self) -> int | None:
        """Retourner l'identifiant persistant actif lorsqu'il est résolu."""
        return self._active_person_id

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
        self._active_person_id = None

    def set_active_persistent_person(
        self,
        person: Person,
    ) -> None:
        """Lier la personne active à une identité persistante validée."""
        if not isinstance(person, Person):
            raise TypeError(
                "Persistent active person must be a Person."
            )

        profile = PersonProfile(name=person.display_name)
        self._validate_person_name(profile)
        self._active_person = profile
        self._active_person_id = person.id

    def reset(self) -> None:
        """Restaurer la personne par défaut pour une nouvelle conversation."""
        self._active_person = self._default_person
        self._active_person_id = self._default_person_id

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
