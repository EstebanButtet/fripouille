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
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.memory_candidates import (
    MemoryCandidateAnalysisError,
    MemoryCandidateAnalyzer,
)
from assistant_ia.intelligence.model_client import (
    ModelClient,
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person
from assistant_ia.people.presentation import detect_presented_person

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
        person_context: ActivePersonContext | None = None,
        memory_candidate_analyzer: (
            MemoryCandidateAnalyzer | None
        ) = None,
    ) -> None:
        """Create the assistant core with optional injected dependencies."""
        if (
            action_registry is not None
            and not isinstance(action_registry, ActionRegistry)
        ):
            raise TypeError(
                "Assistant action registry must be an ActionRegistry."
            )

        if (
            person_context is not None
            and not isinstance(person_context, ActivePersonContext)
        ):
            raise TypeError(
                "Assistant person context must be an "
                "ActivePersonContext."
            )

        default_identity = build_default_identity()

        self._person_context = (
            person_context
            if person_context is not None
            else ActivePersonContext(
                assistant_name=default_identity.name,
                default_person=build_default_person(),
            )
        )

        self._model_client = (
            model_client
            if model_client is not None
            else OllamaModelClient(
                identity=default_identity,
                person_context=self._person_context,
            )
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
        self._memory_candidate_analyzer = memory_candidate_analyzer
        self._last_memory_candidates: tuple[
            MemoryCandidate, ...
        ] = ()

    @property
    def context(self) -> ConversationContext:
        """Return the conversation context managed by the assistant."""
        return self._context

    @property
    def person_context(self) -> ActivePersonContext:
        """Return the current conversational person context."""
        return self._person_context

    @property
    def action_registry(self) -> ActionRegistry:
        """Return the executable action registry used by the assistant."""
        return self._action_registry

    @property
    def last_intent(self) -> Intent | None:
        """Return the last successfully identified intent."""
        return self._last_intent

    @property
    def last_memory_candidates(
        self,
    ) -> tuple[MemoryCandidate, ...]:
        """Return validated non-persistent candidates from the last turn."""
        return self._last_memory_candidates

    def process_message(self, user_message: str) -> str:
        """Process one user message and return the assistant response."""
        self._last_memory_candidates = ()
        current_user_message = self._context.add_user_message(user_message)

        presented_person = detect_presented_person(
            user_message
        )

        if presented_person is not None:
            try:
                self._person_context.set_active_person(
                    presented_person
                )
            except ValueError:
                # The assistant's reserved name can never become
                # the active user's identity.
                pass

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

        if (
            resolved_intent.name == "conversation"
            and self._memory_candidate_analyzer is not None
        ):
            try:
                self._last_memory_candidates = (
                    self._memory_candidate_analyzer.analyze(
                        current_user_message.content
                    )
                )
            except MemoryCandidateAnalysisError:
                self._last_memory_candidates = ()

        return assistant_content

    def reset_conversation(self) -> None:
        """Clear the current conversation context and identified intent."""
        self._context.clear()
        self._person_context.reset()
        self._last_intent = None
        self._last_memory_candidates = ()

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
