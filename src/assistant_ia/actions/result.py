"""Résultat structuré et applicatif d'une exécution d'action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from assistant_ia.intelligence.intent import ALLOWED_INTENT_NAMES

ActionResultStatus = Literal["success", "cancelled", "error"]
ActionErrorKind = Literal["validation", "execution"]

ALLOWED_ACTION_RESULT_STATUSES = frozenset({"success", "cancelled", "error"})
ALLOWED_ACTION_ERROR_KINDS = frozenset({"validation", "execution"})
_EXECUTABLE_NAMES = ALLOWED_INTENT_NAMES - {"conversation", "unknown"}


@dataclass(frozen=True, slots=True)
class ActionExecutionResult:
    """Décrire ce que l'application sait réellement d'une action.

    ``attempted`` signifie que le chemin d'exécution a dépassé les refus de
    validation et a pu tenter un effet. Cela ne prouve ni que l'effet externe
    a eu lieu, ni que le résultat constitue une leçon.
    """

    action_name: str
    status: ActionResultStatus
    message: str
    attempted: bool
    error_kind: ActionErrorKind | None = None

    def __post_init__(self) -> None:
        action_name = _normalize_text(self.action_name, "Action result name")
        if action_name not in _EXECUTABLE_NAMES:
            raise ValueError(f"Unknown executable action: {action_name!r}.")
        status = _normalize_choice(
            self.status,
            ALLOWED_ACTION_RESULT_STATUSES,
            "action result status",
        )
        if not isinstance(self.attempted, bool):
            raise TypeError("Action result attempted flag must be boolean.")
        error_kind = self.error_kind
        if error_kind is not None:
            error_kind = cast(
                ActionErrorKind,
                _normalize_choice(
                    error_kind,
                    ALLOWED_ACTION_ERROR_KINDS,
                    "action error kind",
                ),
            )

        if status == "success":
            if not self.attempted or error_kind is not None:
                raise ValueError(
                    "Successful actions must be attempted and have no error."
                )
        elif status == "cancelled":
            if self.attempted or error_kind is not None:
                raise ValueError(
                    "Cancelled actions must be unattempted and have no error."
                )
        else:
            if error_kind is None:
                raise ValueError("Action errors require an error kind.")
            if self.attempted != (error_kind == "execution"):
                raise ValueError(
                    "Only execution errors represent an attempted handler."
                )

        object.__setattr__(self, "action_name", action_name)
        object.__setattr__(self, "status", cast(ActionResultStatus, status))
        object.__setattr__(self, "message", _normalize_text(self.message, "Action result message"))
        object.__setattr__(self, "error_kind", error_kind)


def _normalize_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_choice(value: object, allowed: frozenset[str], field_name: str) -> str:
    normalized = _normalize_text(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"Unknown {field_name}: {normalized!r}.")
    return normalized
