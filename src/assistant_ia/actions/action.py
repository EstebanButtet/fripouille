"""Définition validée d'une action exécutable par l'assistant.

Une :class:`Action` associe un nom d'intention appartenant à la liste fermée à
un handler applicatif. Avant l'appel, elle contrôle exactement les paramètres
requis et autorisés. Après l'appel, elle contrôle aussi que le résultat est un
texte présentable. Le LLM ne reçoit jamais le handler lui-même.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from assistant_ia.intelligence.intent import (
    INTENT_PARAMETER_SPECIFICATIONS,
    Intent,
)
from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import DatabaseError

ExecutableIntentName = Literal[
    "create_task",
    "list_tasks",
    "complete_task",
    "save_memory",
    "find_memory",
    "delete_memory",
    "write_journal",
    "launch_application",
]

EXECUTABLE_INTENT_NAMES: frozenset[str] = frozenset(
    {
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

ActionHandler = Callable[[Mapping[str, str]], str]


class ActionError(RuntimeError):
    """Base commune des erreurs contrôlées d'action."""


class ActionValidationError(ActionError):
    """Signaler des paramètres ou une demande d'action invalides."""


class ActionExecutionError(ActionError):
    """Signaler qu'une action validée n'a pas pu s'exécuter sûrement."""


@dataclass(frozen=True, slots=True)
class Action:
    """Relier un contrat d'intention validé à un handler d'exécution.

    ``handler`` reçoit une vue de paramètres textuels et retourne un texte.
    La dataclass est immuable : le registre ne peut pas voir son exécuteur
    remplacé après l'enregistrement.
    """

    name: ExecutableIntentName
    handler: ActionHandler

    def __post_init__(self) -> None:
        """Valider le nom fermé et le handler appelable de l'action."""
        if not isinstance(self.name, str):
            raise TypeError("Action name must be a string.")

        normalized_name = self.name.strip()

        if normalized_name not in EXECUTABLE_INTENT_NAMES:
            raise ValueError(
                f"Unsupported executable action: {normalized_name!r}."
            )

        if not callable(self.handler):
            raise TypeError("Action handler must be callable.")

        object.__setattr__(
            self,
            "name",
            cast(ExecutableIntentName, normalized_name),
        )

    def execute(self, intent: Intent) -> str:
        """Valider une intention puis appeler le handler enregistré.

        Les erreurs de paramètres et de repository sont traduites en familles
        d'erreurs distinctes. Une ``ActionError`` déjà explicite est conservée.
        """
        if not isinstance(intent, Intent):
            raise TypeError("Action execution requires an Intent.")

        if intent.name != self.name:
            raise ActionValidationError(
                "L’intention ne correspond pas à l’action demandée."
            )

        specification = INTENT_PARAMETER_SPECIFICATIONS[self.name]
        parameter_names = frozenset(intent.parameters)

        missing_parameters = (
            specification.required - parameter_names
        )
        unexpected_parameters = parameter_names - (
            specification.required | specification.optional
        )

        if missing_parameters:
            missing_names = ", ".join(sorted(missing_parameters))
            raise ActionValidationError(
                "Paramètre requis manquant : "
                f"{missing_names}."
            )

        if unexpected_parameters:
            unexpected_names = ", ".join(
                sorted(unexpected_parameters)
            )
            raise ActionValidationError(
                "Paramètre non autorisé : "
                f"{unexpected_names}."
            )

        # L'appel n'arrive qu'après validation exacte du contrat de paramètres.
        # Le handler ne reçoit donc pas les champs arbitraires du JSON Ollama.
        try:
            result = self.handler(intent.parameters)
        except ActionError:
            raise
        except (TypeError, ValueError) as error:
            raise ActionValidationError(
                "Les paramètres de l’action sont invalides."
            ) from error
        except (RepositoryError, DatabaseError) as error:
            raise ActionExecutionError(
                 "L’action n’a pas pu être exécutée."
            ) from error

        if not isinstance(result, str):
            raise ActionExecutionError(
                "Le résultat de l’action doit être un texte."
            )

        normalized_result = result.strip()

        if not normalized_result:
            raise ActionExecutionError(
                "Le résultat de l’action ne peut pas être vide."
            )

        return normalized_result
