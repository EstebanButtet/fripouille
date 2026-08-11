"""Validated internal results of turn interpretation."""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.intelligence.conversation import (
    ConversationDirectiveProposal,
    ConversationMode,
)
from assistant_ia.intelligence.intent import Intent


@dataclass(frozen=True, slots=True)
class TurnInterpretation:
    """Combine intent and conversation-generation metadata."""

    intent: Intent
    conversation_directive_proposal: ConversationDirectiveProposal
    model: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.intent,
            Intent,
        ):
            raise TypeError(
                "Turn interpretation intent must be an Intent."
            )

        if not isinstance(
            self.conversation_directive_proposal,
            ConversationDirectiveProposal,
        ):
            raise TypeError(
                "Turn interpretation conversation directive "
                "proposal must be a ConversationDirectiveProposal."
            )

        if not isinstance(
            self.model,
            str,
        ):
            raise TypeError(
                "Turn interpretation model must be a string."
            )

        normalized_model = self.model.strip()

        if not normalized_model:
            raise ValueError(
                "Turn interpretation model cannot be empty."
            )

        if (
            self.intent.name != "conversation"
            and self.conversation_directive_proposal.mode
            is not ConversationMode.STANDARD
        ):
            raise ValueError(
                "Non-conversation intents require a standard "
                "conversation directive proposal."
            )

        object.__setattr__(
            self,
            "model",
            normalized_model,
        )
