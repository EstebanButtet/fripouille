"""In-memory conversation context management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """Represent one message stored in the conversation history."""

    role: MessageRole
    content: str


class ConversationContext:
    """Store conversation messages in memory and preserve their order."""

    def __init__(self) -> None:
        """Create an empty conversation context."""
        self._messages: list[ConversationMessage] = []

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        """Return the conversation messages as an immutable sequence."""
        return tuple(self._messages)

    @property
    def message_count(self) -> int:
        """Return the number of messages stored in the context."""
        return len(self._messages)

    def add_user_message(self, content: str) -> ConversationMessage:
        """Add a user message to the conversation."""
        return self._add_message(role="user", content=content)

    def add_assistant_message(self, content: str) -> ConversationMessage:
        """Add an assistant message to the conversation."""
        return self._add_message(role="assistant", content=content)

    def clear(self) -> None:
        """Remove every message from the conversation."""
        self._messages.clear()

    def _add_message(
        self,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage:
        """Validate and store one conversation message."""
        if not isinstance(content, str):
            raise TypeError("Message content must be a string.")

        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError("Conversation messages cannot be empty.")

        message = ConversationMessage(
            role=role,
            content=normalized_content,
        )
        self._messages.append(message)

        return message
