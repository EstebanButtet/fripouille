"""Application runtime for coordinating assistant interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.interfaces.presentation import (
    build_user_facing_response,
)
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.promotion import MemoryPromotionProposal


@dataclass(frozen=True, slots=True)
class TurnDiagnostics:
    """Keep technical turn details outside the visible conversation."""

    user_message: str
    raw_response: str
    intent: Intent | None
    memory_candidates: tuple[MemoryCandidate, ...]
    memory_promotion_proposal: MemoryPromotionProposal | None


class DiagnosticReporter(Protocol):
    """Report technical details for one completed turn."""

    def report(self, diagnostics: TurnDiagnostics) -> None:
        """Report diagnostics without changing the user-facing response."""


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
        diagnostic_reporter: DiagnosticReporter | None = None,
    ) -> None:
        """Create the runtime around an assembled assistant core."""
        if not isinstance(assistant, AssistantCore):
            raise TypeError(
                "Assistant runtime requires an AssistantCore."
            )

        self._assistant = assistant
        self._presenter = presenter
        self._diagnostic_reporter = diagnostic_reporter

    @property
    def assistant(self) -> AssistantCore:
        """Return the conversational core owned by this runtime."""
        return self._assistant

    def process_message(
        self,
        user_message: str,
    ) -> str:
        """Process one turn and present the final resolved response."""
        raw_response = self._assistant.process_message(
            user_message
        )
        diagnostics = TurnDiagnostics(
            user_message=user_message,
            raw_response=raw_response,
            intent=self._assistant.last_intent,
            memory_candidates=(
                self._assistant.last_memory_candidates
            ),
            memory_promotion_proposal=(
                self._assistant.last_memory_promotion_proposal
            ),
        )

        if self._diagnostic_reporter is not None:
            self._diagnostic_reporter.report(diagnostics)

        response = build_user_facing_response(
            raw_response,
            intent=diagnostics.intent,
            memory_proposal=(
                diagnostics.memory_promotion_proposal
            ),
            awaiting_memory_confirmation=(
                self._assistant.pending_memory_promotion is not None
            ),
        )

        if self._presenter is not None:
            self._presenter.present(
                response
            )

        return response

    def reset_conversation(self) -> None:
        """Reset the conversational state owned by the assistant core."""
        self._assistant.reset_conversation()
