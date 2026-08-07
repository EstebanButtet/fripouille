"""Immutable registry for executable assistant actions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from assistant_ia.actions.action import (
    Action,
    ActionError,
)
from assistant_ia.intelligence.intent import Intent


class ActionRegistryError(ActionError):
    """Base error raised by the action registry."""


class ActionNotRegisteredError(ActionRegistryError):
    """Raised when no executable action matches an intent."""


class ActionAlreadyRegisteredError(ActionRegistryError):
    """Raised when an action name is registered more than once."""


class ActionRegistry:
    """Store and select immutable executable action definitions."""

    def __init__(
        self,
        actions: Iterable[Action] = (),
    ) -> None:
        """Create a registry from a collection of validated actions."""
        registered_actions: dict[str, Action] = {}

        for action in actions:
            if not isinstance(action, Action):
                raise TypeError(
                    "Action registry entries must be Action instances."
                )

            if action.name in registered_actions:
                raise ActionAlreadyRegisteredError(
                    "Une action est déjà enregistrée pour "
                    f"l’intention {action.name!r}."
                )

            registered_actions[action.name] = action

        self._actions: Mapping[str, Action] = MappingProxyType(
            registered_actions
        )

    @property
    def action_names(self) -> frozenset[str]:
        """Return the immutable set of registered action names."""
        return frozenset(self._actions)

    @property
    def action_count(self) -> int:
        """Return the number of registered executable actions."""
        return len(self._actions)

    def has_action(self, action_name: str) -> bool:
        """Return whether one normalized action name is registered."""
        if not isinstance(action_name, str):
            raise TypeError("Action name must be a string.")

        normalized_name = action_name.strip()

        if not normalized_name:
            raise ValueError("Action name cannot be empty.")

        return normalized_name in self._actions

    def execute(self, intent: Intent) -> str:
        """Execute the action registered for one structured intent."""
        if not isinstance(intent, Intent):
            raise TypeError(
                "Action registry execution requires an Intent."
            )

        action = self._actions.get(intent.name)

        if action is None:
            raise ActionNotRegisteredError(
                "Aucune action exécutable n’est enregistrée pour "
                f"l’intention {intent.name!r}."
            )

        return action.execute(intent)
