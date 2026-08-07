"""Structured assistant intent definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

IntentName = Literal[
    "conversation",
    "unknown",
    "create_task",
    "list_tasks",
    "complete_task",
    "save_memory",
    "find_memory",
    "delete_memory",
    "write_journal",
    "launch_application",
]

ALLOWED_INTENT_NAMES: frozenset[str] = frozenset(
    {
        "conversation",
        "unknown",
        "create_task",
        "list_tasks",
        "complete_task",
        "save_memory",
        "find_memory",
        "delete_memory",
        "write_journal",
        "launch_application",
    }
)


@dataclass(frozen=True, slots=True)
class Intent:
    """Represent a validated intent identified in a user request."""

    name: IntentName
    parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the intent fields."""
        if not isinstance(self.name, str):
            raise TypeError("Intent name must be a string.")

        normalized_name = self.name.strip()

        if normalized_name not in ALLOWED_INTENT_NAMES:
            raise ValueError(f"Unknown intent name: {normalized_name!r}.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("Intent parameters must be a mapping.")

        normalized_parameters: dict[str, str] = {}

        for key, value in self.parameters.items():
            if not isinstance(key, str):
                raise TypeError("Intent parameter names must be strings.")

            if not isinstance(value, str):
                raise TypeError("Intent parameter values must be strings.")

            normalized_key = key.strip()
            normalized_value = value.strip()

            if not normalized_key:
                raise ValueError("Intent parameter names cannot be empty.")

            if not normalized_value:
                raise ValueError("Intent parameter values cannot be empty.")

            if normalized_key in normalized_parameters:
                raise ValueError(
                    f"Duplicate intent parameter: {normalized_key!r}."
                )

            normalized_parameters[normalized_key] = normalized_value

        object.__setattr__(
            self,
            "name",
            cast(IntentName, normalized_name),
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(normalized_parameters),
        )
        