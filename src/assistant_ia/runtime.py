"""Application runtime for coordinating assistant interactions."""

from __future__ import annotations

from typing import Protocol

from assistant_ia.core.assistant import AssistantCore


class ResponsePresenter(Protocol):
    """Present one final assistant response outside the conversational core."""

    def present(self, response: str) -> None:
        """Present one final assistant response."""


class AssistantRuntime:
    """Coordinate the assistant core with optional external presentation."""

    def __init__(
        self,
        assistant: AssistantCore,
        presenter: ResponsePresenter | None = None,
    ) -> None:
        """Create the runtime around an assembled assistant core."""
        if not isinstance(assistant, AssistantCore):
            raise TypeError(
                "Assistant runtime requires an AssistantCore."
            )

        self._assistant = assistant
        self._presenter = presenter

    @property
    def assistant(self) -> AssistantCore:
        """Return the conversational core owned by this runtime."""
        return self._assistant

    def process_message(
        self,
        user_message: str,
    ) -> str:
        """Process one turn and present the final resolved response."""
        response = self._assistant.process_message(
            user_message
        )

        if self._presenter is not None:
            self._presenter.present(
                response
            )

        return response

    def reset_conversation(self) -> None:
        """Reset the conversational state owned by the assistant core."""
        self._assistant.reset_conversation()
