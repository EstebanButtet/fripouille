"""Core orchestration for the personal AI assistant."""

from __future__ import annotations

from assistant_ia.core.context import ConversationContext
from assistant_ia.intelligence.model_client import (
    ModelClient,
    ModelClientError,
    OllamaModelClient,
)


class AssistantCoreError(RuntimeError):
    """Raised when the assistant cannot complete a requested operation."""


class AssistantCore:
    """Coordinate user messages, model responses and conversation context."""

    def __init__(
        self,
        model_client: ModelClient | None = None,
        context: ConversationContext | None = None,
    ) -> None:
        """Create the assistant core with optional injected dependencies."""
        self._model_client = (
            model_client if model_client is not None else OllamaModelClient()
        )
        self._context = context if context is not None else ConversationContext()

    @property
    def context(self) -> ConversationContext:
        """Return the conversation context managed by the assistant."""
        return self._context

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the generated model response."""
        self._context.add_user_message(user_message)

        try:
            model_response = self._model_client.generate_response(
                self._context.messages,
            )
        except ModelClientError as error:
            raise AssistantCoreError(
                "The local language model could not produce a response."
            ) from error

        self._context.add_assistant_message(model_response.content)

        return model_response.content

    def reset_conversation(self) -> None:
        """Clear the current conversation context."""
        self._context.clear()
