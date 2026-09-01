"""Définition et validation des intentions structurées de l'assistant.

Une intention est la proposition normalisée qui relie l'interprétation du
message à une action connue de l'application. La liste fermée des noms et des
paramètres constitue une frontière de sécurité : Ollama ne peut pas inventer
un nom d'action ou transmettre des champs arbitraires au registre.
"""

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


@dataclass(frozen=True, slots=True)
class IntentParameterSpecification:
    """Décrire les paramètres obligatoires et facultatifs d'une intention.

    Les deux ensembles sont immuables et ne doivent pas se chevaucher. Cette
    classe décrit le contrat ; la validation métier des valeurs appartient
    ensuite à chaque action.
    """

    required: frozenset[str] = field(default_factory=frozenset)
    optional: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Vérifier que le contrat de paramètres est cohérent et normalisé."""
        if not isinstance(self.required, frozenset):
            raise TypeError(
                "Required intent parameters must be a frozenset."
            )

        if not isinstance(self.optional, frozenset):
            raise TypeError(
                "Optional intent parameters must be a frozenset."
            )

        for parameter_name in self.required | self.optional:
            if not isinstance(parameter_name, str):
                raise TypeError(
                    "Intent parameter specification names must be strings."
                )

            if not parameter_name.strip():
                raise ValueError(
                    "Intent parameter specification names cannot be empty."
                )

            if parameter_name != parameter_name.strip():
                raise ValueError(
                    "Intent parameter specification names must be normalized."
                )

        overlapping_parameters = self.required & self.optional

        if overlapping_parameters:
            raise ValueError(
                "Intent parameters cannot be both required and optional."
            )


INTENT_PARAMETER_SPECIFICATIONS: Mapping[
    IntentName,
    IntentParameterSpecification,
] = MappingProxyType(
    {
        "conversation": IntentParameterSpecification(),
        "unknown": IntentParameterSpecification(),
        "create_task": IntentParameterSpecification(
            required=frozenset(
                {
                    "title",
                }
            ),
            optional=frozenset(
                {
                    "due_at",
                }
            ),
        ),
        "list_tasks": IntentParameterSpecification(
            optional=frozenset(
                {
                    "status",
                }
            ),
        ),
        "complete_task": IntentParameterSpecification(
            required=frozenset(
                {
                    "task_id",
                }
            ),
        ),
        "save_memory": IntentParameterSpecification(
            required=frozenset(
                {
                    "content",
                }
            ),
        ),
        "find_memory": IntentParameterSpecification(
            required=frozenset(
                {
                    "query",
                }
            ),
        ),
        "delete_memory": IntentParameterSpecification(
            required=frozenset(
                {
                    "memory_id",
                }
            ),
        ),
        "write_journal": IntentParameterSpecification(
            required=frozenset(
                {
                    "content",
                }
            ),
            optional=frozenset(
                {
                    "entry_date",
                }
            ),
        ),
        "launch_application": IntentParameterSpecification(
            required=frozenset(
                {
                    "application",
                }
            ),
        ),
    }
)

ALLOWED_INTENT_NAMES: frozenset[str] = frozenset(
    INTENT_PARAMETER_SPECIFICATIONS
)


@dataclass(frozen=True, slots=True)
class Intent:
    """Représenter une intention validée issue d'une demande utilisateur.

    ``name`` choisit un chemin autorisé. ``parameters`` ne contient que des
    chaînes normalisées et devient un ``MappingProxyType`` en lecture seule :
    une action ne peut donc pas recevoir un dictionnaire modifié après coup.
    """

    name: IntentName
    parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valider les champs et figer une copie normalisée des paramètres."""
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
