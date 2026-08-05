"""Core orchestration for the personal AI assistant."""

from __future__ import annotations

from assistant_ia.core.context import ConversationContext


class AssistantCore:
    """Coordinate user messages, responses and conversation context."""

    def __init__(
        self,
        context: ConversationContext | None = None,
    ) -> None:
        """Create the assistant core with an optional conversation context."""
        self._context = context if context is not None else ConversationContext()

    @property
    def context(self) -> ConversationContext:
        """Return the conversation context managed by the assistant."""
        return self._context

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the assistant response."""
        stored_user_message = self._context.add_user_message(user_message)

        response = (
            f"Message reçu : {stored_user_message.content!r}. "
            "Le noyau de l'assistant fonctionne, "
            "mais aucun modèle d'IA n'est encore connecté."
        )

        self._context.add_assistant_message(response)

        return response

    def reset_conversation(self) -> None:
        """Clear the current conversation context."""
        self._context.clear()
