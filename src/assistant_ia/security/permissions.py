"""Action permission policy definitions."""

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
    """Resolve explicit permission decisions for assistant actions."""

    decisions: Mapping[str, PermissionDecision] = field(
        default_factory=dict
    )
    default_decision: PermissionDecision = "denied"

    def __post_init__(self) -> None:
        """Validate and freeze the configured permission decisions."""
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
        """Return the permission decision for one normalized action."""
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
    """Build the default permission policy for system actions."""
    return PermissionPolicy(
        decisions={
            "launch_application": "confirmation_required",
        }
    )
