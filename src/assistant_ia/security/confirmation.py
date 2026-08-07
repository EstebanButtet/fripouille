"""Explicit user confirmation boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """Describe one assistant action requiring user confirmation."""

    action_name: str
    description: str

    def __post_init__(self) -> None:
        """Validate and normalize the confirmation request."""
        if not isinstance(self.action_name, str):
            raise TypeError(
                "Confirmation action name must be a string."
            )

        if not isinstance(self.description, str):
            raise TypeError(
                "Confirmation description must be a string."
            )

        normalized_action_name = self.action_name.strip()
        normalized_description = self.description.strip()

        if not normalized_action_name:
            raise ValueError(
                "Confirmation action name cannot be empty."
            )

        if not normalized_description:
            raise ValueError(
                "Confirmation description cannot be empty."
            )

        object.__setattr__(
            self,
            "action_name",
            normalized_action_name,
        )
        object.__setattr__(
            self,
            "description",
            normalized_description,
        )


ConfirmationHandler = Callable[[ConfirmationRequest], bool]


def request_confirmation(
    handler: ConfirmationHandler,
    request: ConfirmationRequest,
) -> bool:
    """Request one explicit confirmation through an injected handler."""
    if not callable(handler):
        raise TypeError(
            "Confirmation handler must be callable."
        )

    if not isinstance(request, ConfirmationRequest):
        raise TypeError(
            "Confirmation requires a ConfirmationRequest."
        )

    result = handler(request)

    if not isinstance(result, bool):
        raise TypeError(
            "Confirmation handler must return a boolean."
        )

    return result


def deny_confirmation(
    request: ConfirmationRequest,
) -> bool:
    """Deny confirmation when no interactive boundary is available."""
    if not isinstance(request, ConfirmationRequest):
        raise TypeError(
            "Confirmation requires a ConfirmationRequest."
        )

    return False
