"""Active person context for one conversational session."""

from __future__ import annotations

from assistant_ia.people.models import PersonProfile


class ActivePersonContext:
    """Track who is currently interacting with the assistant."""

    def __init__(
        self,
        *,
        assistant_name: str,
        default_person: PersonProfile,
    ) -> None:
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
        """Return the assistant's reserved name."""
        return self._assistant_name

    @property
    def default_person(self) -> PersonProfile:
        """Return the person restored for a new conversation."""
        return self._default_person

    @property
    def active_person(self) -> PersonProfile:
        """Return the person currently interacting with the assistant."""
        return self._active_person

    def set_active_person(
        self,
        person: PersonProfile,
    ) -> None:
        """Switch the active conversational user."""
        if not isinstance(person, PersonProfile):
            raise TypeError(
                "Active person must be a PersonProfile."
            )

        self._validate_person_name(person)
        self._active_person = person

    def reset(self) -> None:
        """Restore the default person for a new conversation."""
        self._active_person = self._default_person

    def _validate_person_name(
        self,
        person: PersonProfile,
    ) -> None:
        if (
            person.name.casefold()
            == self._assistant_name.casefold()
        ):
            raise ValueError(
                "The assistant name is reserved exclusively "
                "for the assistant."
            )
