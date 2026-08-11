"""Default person configuration."""

from __future__ import annotations

from assistant_ia.people.models import PersonProfile


DEFAULT_PERSON_NAME = "Este"


def build_default_person() -> PersonProfile:
    """Build the person assumed to be using the assistant by default."""
    return PersonProfile(
        name=DEFAULT_PERSON_NAME,
    )
