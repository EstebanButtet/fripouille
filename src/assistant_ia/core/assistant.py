"""Core orchestration for the personal AI assistant."""

from __future__ import annotations

from assistant_ia.actions.action import (
    EXECUTABLE_INTENT_NAMES,
    ActionExecutionError,
    ActionValidationError,
)
from assistant_ia.actions.registry import (
    ActionNotRegisteredError,
    ActionRegistry,
)
from assistant_ia.core.context import ConversationContext
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.model_client import (
    ModelClient,
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.response import ModelResponse

ACTION_UNAVAILABLE_MESSAGE = (
    "J’ai identifié votre demande, mais aucune action n’a été exécutée. "
    "Cette action n’est pas disponible."
)

ACTION_EXECUTION_ERROR_MESSAGE = (
    "L’action n’a pas pu être confirmée. "
    "Vérifiez l’état enregistré avant de réessayer."
)

JOURNAL_CONTENT_SEPARATOR = " : "


class AssistantCoreError(RuntimeError):
    """Raised when the assistant cannot complete a requested operation."""


class AssistantCore:
    """Coordinate messages, model responses and executable actions."""

    def __init__(
        self,
        model_client: ModelClient | None = None,
        context: ConversationContext | None = None,
        action_registry: ActionRegistry | None = None,
    ) -> None:
        """Create the assistant core with optional injected dependencies."""
        if (
            action_registry is not None
            and not isinstance(action_registry, ActionRegistry)
        ):
            raise TypeError(
                "Assistant action registry must be an ActionRegistry."
            )

        self._model_client = (
            model_client
            if model_client is not None
            else OllamaModelClient()
        )
        self._context = (
            context
            if context is not None
            else ConversationContext()
        )
        self._action_registry = (
            action_registry
            if action_registry is not None
            else ActionRegistry()
        )
        self._last_intent: Intent | None = None

    @property
    def context(self) -> ConversationContext:
        """Return the conversation context managed by the assistant."""
        return self._context

    @property
    def action_registry(self) -> ActionRegistry:
        """Return the executable action registry used by the assistant."""
        return self._action_registry

    @property
    def last_intent(self) -> Intent | None:
        """Return the last successfully identified intent."""
        return self._last_intent

    def process_message(self, user_message: str) -> str:
        """Process one user message and return the assistant response."""
        self._context.add_user_message(user_message)

        try:
            model_response = self._model_client.generate_response(
                self._context.messages,
            )
        except ModelClientError as error:
            raise AssistantCoreError(
                "The local language model could not produce a response."
            ) from error

        resolved_intent = _recover_missing_journal_content(
            model_response.intent,
            user_message,
        )
        self._last_intent = resolved_intent

        assistant_content = self._resolve_assistant_content(
            model_response=model_response,
            intent=resolved_intent,
        )

        self._context.add_assistant_message(
            assistant_content
        )

        return assistant_content

    def reset_conversation(self) -> None:
        """Clear the current conversation context and identified intent."""
        self._context.clear()
        self._last_intent = None

    def _resolve_assistant_content(
        self,
        *,
        model_response: ModelResponse,
        intent: Intent,
    ) -> str:
        """Resolve safe visible content from a structured response."""
        if intent.name == "conversation":
            return model_response.content

        if intent.name not in EXECUTABLE_INTENT_NAMES:
            return ACTION_UNAVAILABLE_MESSAGE

        try:
            return self._action_registry.execute(intent)
        except ActionNotRegisteredError:
            return ACTION_UNAVAILABLE_MESSAGE
        except ActionValidationError as error:
            return str(error)
        except ActionExecutionError:
            return ACTION_EXECUTION_ERROR_MESSAGE


def _recover_missing_journal_content(
    intent: Intent,
    user_message: str,
) -> Intent:
    """Recover explicit journal content omitted by the model."""
    if intent.name != "write_journal":
        return intent

    if "content" in intent.parameters:
        return intent

    separator_index = user_message.find(
        JOURNAL_CONTENT_SEPARATOR
    )

    if separator_index < 0:
        return intent

    content = user_message[
        separator_index + len(JOURNAL_CONTENT_SEPARATOR):
    ].strip()

    if not content:
        return intent

    parameters = dict(intent.parameters)
    parameters["content"] = content

    return Intent(
        name=intent.name,
        parameters=parameters,
    )
