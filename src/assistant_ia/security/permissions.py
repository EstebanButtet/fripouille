"""Politique déterministe de permission des actions.

La politique reçoit un nom d'action normalisé et retourne une décision fermée :
autoriser, demander confirmation ou refuser. Elle ne présente pas elle-même la
confirmation et n'exécute aucun effet système.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

PermissionDecision = Literal[
    "allowed",
    "confirmation_required",
    "denied",
]

ALLOWED_PERMISSION_DECISIONS: frozenset[str] = frozenset(
    {
        "allowed",
        "confirmation_required",
        "denied",
    }
)


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """Résoudre une décision explicite pour chaque action.

    ``decisions`` contient les exceptions nominatives ; ``default_decision``
    s'applique à tout nom absent. Le défaut est le refus, choix sûr pour une
    action nouvelle qui n'aurait pas encore été configurée.
    """

    decisions: Mapping[str, PermissionDecision] = field(
        default_factory=dict
    )
    default_decision: PermissionDecision = "denied"

    def __post_init__(self) -> None:
        """Valider puis figer une copie des décisions configurées."""
        if not isinstance(self.decisions, Mapping):
            raise TypeError(
                "Permission decisions must be a mapping."
            )

        if self.default_decision not in ALLOWED_PERMISSION_DECISIONS:
            raise ValueError(
                "Unknown default permission decision: "
                f"{self.default_decision!r}."
            )

        normalized_decisions: dict[str, PermissionDecision] = {}

        for action_name, decision in self.decisions.items():
            if not isinstance(action_name, str):
                raise TypeError(
                    "Permission action names must be strings."
                )

            normalized_name = action_name.strip()

            if not normalized_name:
                raise ValueError(
                    "Permission action names cannot be empty."
                )

            if normalized_name != action_name:
                raise ValueError(
                    "Permission action names must be normalized."
                )

            if decision not in ALLOWED_PERMISSION_DECISIONS:
                raise ValueError(
                    "Unknown permission decision for "
                    f"{normalized_name!r}: {decision!r}."
                )

            normalized_decisions[normalized_name] = cast(
                PermissionDecision,
                decision,
            )

        object.__setattr__(
            self,
            "decisions",
            MappingProxyType(normalized_decisions),
        )
        object.__setattr__(
            self,
            "default_decision",
            cast(
                PermissionDecision,
                self.default_decision,
            ),
        )

    def decision_for(
        self,
        action_name: str,
    ) -> PermissionDecision:
        """Retourner la décision nominative ou la décision par défaut."""
        if not isinstance(action_name, str):
            raise TypeError(
                "Permission action name must be a string."
            )

        normalized_name = action_name.strip()

        if not normalized_name:
            raise ValueError(
                "Permission action name cannot be empty."
            )

        return self.decisions.get(
            normalized_name,
            self.default_decision,
        )


def build_default_permission_policy() -> PermissionPolicy:
    """Construire la politique par défaut des actions système.

    Le lancement d'application exige actuellement une confirmation explicite ;
    les autres actions non listées héritent du refus par défaut.
    """
    return PermissionPolicy(
        decisions={
            "launch_application": "confirmation_required",
        }
    )
