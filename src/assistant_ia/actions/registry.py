"""Registre immuable des actions exécutables disponibles.

Le registre associe chaque nom autorisé à une unique :class:`Action`. Il est
consulté par ``AssistantCore`` après interprétation et représente la frontière
d'autorité : une intention sans entrée enregistrée ne peut pas s'exécuter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from assistant_ia.actions.action import (
    Action,
    ActionError,
)
from assistant_ia.intelligence.intent import Intent


class ActionRegistryError(ActionError):
    """Base des erreurs propres au registre d'actions."""


class ActionNotRegisteredError(ActionRegistryError):
    """Signaler qu'aucune action enregistrée ne correspond à l'intention."""


class ActionAlreadyRegisteredError(ActionRegistryError):
    """Signaler que deux handlers revendiquent le même nom d'action."""


class ActionRegistry:
    """Stocker et sélectionner des définitions d'action immuables."""

    def __init__(
        self,
        actions: Iterable[Action] = (),
    ) -> None:
        """Construire une table en lecture seule depuis des actions validées."""
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

        # MappingProxyType expose un dictionnaire non modifiable après
        # l'assemblage ; les capacités annoncées restent donc cohérentes.
        self._actions: Mapping[str, Action] = MappingProxyType(
            registered_actions
        )

    @property
    def action_names(self) -> frozenset[str]:
        """Retourner l'ensemble immuable des noms enregistrés."""
        return frozenset(self._actions)

    @property
    def action_count(self) -> int:
        """Retourner le nombre d'actions réellement exécutables."""
        return len(self._actions)

    def has_action(self, action_name: str) -> bool:
        """Indiquer si un nom normalisé possède une action enregistrée."""
        if not isinstance(action_name, str):
            raise TypeError("Action name must be a string.")

        normalized_name = action_name.strip()

        if not normalized_name:
            raise ValueError("Action name cannot be empty.")

        return normalized_name in self._actions

    def execute(self, intent: Intent) -> str:
        """Sélectionner puis exécuter l'action d'une intention structurée."""
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
