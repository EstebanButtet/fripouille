"""Core orchestration for the personal AI assistant."""

from __future__ import annotations

from assistant_ia.core.context import ConversationContext
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.model_client import (
    ModelClient,
    ModelClientError,
    OllamaModelClient,
)

ACTION_UNAVAILABLE_MESSAGE = (
    "J’ai identifié votre demande, mais aucune action n’a été exécutée. "
    "Les fonctionnalités d’action ne sont pas encore disponibles."
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
        self._last_intent: Intent | None = None

    @property
    def context(self) -> ConversationContext:
        """Return the conversation context managed by the assistant."""
        return self._context

    @property
    def last_intent(self) -> Intent | None:
        """Return the last successfully identified intent."""
        return self._last_intent

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

        self._last_intent = model_response.intent

        if model_response.intent.name == "conversation":
            assistant_content = model_response.content
        else:
            assistant_content = ACTION_UNAVAILABLE_MESSAGE

        self._context.add_assistant_message(assistant_content)

        return assistant_content

    def reset_conversation(self) -> None:
        """Clear the current conversation context and identified intent."""
        self._context.clear()
        self._last_intent = None
