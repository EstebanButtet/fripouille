"""Preparation of one conversational turn from ordered messages."""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.core.context import ConversationMessage

# Initial empirical limits: keep them centralized so real usage can be
# measured before any future adjustment.
MAX_PROJECTED_HISTORY_MESSAGES = 8
MAX_PROJECTED_HISTORY_CHARACTERS = 3000


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
        history=project_conversation_history(
            messages[:-1]
        ),
        current_user_message=current_user_message,
    )


def project_conversation_history(
    history: tuple[ConversationMessage, ...],
) -> tuple[ConversationMessage, ...]:
    """Return a recent bounded suffix without truncating messages."""
    if not isinstance(history, tuple):
        raise TypeError(
            "Conversation history must be provided as a tuple."
        )

    groups = _group_conversation_history(history)
    selected_groups: list[tuple[ConversationMessage, ...]] = []
    selected_message_count = 0
    selected_character_count = 0

    for group in reversed(groups):
        group_message_count = len(group)
        group_character_count = sum(
            len(message.content)
            for message in group
        )

        if (
            selected_message_count + group_message_count
            > MAX_PROJECTED_HISTORY_MESSAGES
            or selected_character_count + group_character_count
            > MAX_PROJECTED_HISTORY_CHARACTERS
        ):
            break

        selected_groups.append(group)
        selected_message_count += group_message_count
        selected_character_count += group_character_count

    return tuple(
        message
        for group in reversed(selected_groups)
        for message in group
    )


def _group_conversation_history(
    history: tuple[ConversationMessage, ...],
) -> tuple[tuple[ConversationMessage, ...], ...]:
    """Group normal user/assistant exchanges while tolerating odd history."""
    groups: list[tuple[ConversationMessage, ...]] = []
    index = 0

    while index < len(history):
        message = history[index]

        if (
            message.role == "user"
            and index + 1 < len(history)
            and history[index + 1].role == "assistant"
        ):
            groups.append(
                (
                    message,
                    history[index + 1],
                )
            )
            index += 2
            continue

        groups.append((message,))
        index += 1

    return tuple(groups)
