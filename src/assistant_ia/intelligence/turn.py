"""Preparation of one conversational turn from ordered messages."""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.core.context import ConversationMessage


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Separate previous conversation history from the current user message."""

    history: tuple[ConversationMessage, ...]
    current_user_message: ConversationMessage


def build_conversation_turn(
    messages: tuple[ConversationMessage, ...],
) -> ConversationTurn:
    """Build one turn from an ordered conversation ending with the user."""
    if not isinstance(messages, tuple):
        raise TypeError(
            "Conversation messages must be provided as a tuple."
        )

    if not messages:
        raise ValueError(
            "At least one conversation message is required."
        )

    current_user_message = messages[-1]

    if current_user_message.role != "user":
        raise ValueError(
            "The latest conversation message must be from the user."
        )

    return ConversationTurn(
        history=messages[:-1],
        current_user_message=current_user_message,
    )
