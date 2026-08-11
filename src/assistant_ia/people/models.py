"""Structured person profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonProfile:
    """Minimal identity of a person interacting with the assistant."""

    name: str

    def __post_init__(self) -> None:
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
