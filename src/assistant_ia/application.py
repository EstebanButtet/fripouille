"""Application assembly for the local personal assistant."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from assistant_ia.actions.defaults import (
    build_default_action_registry,
)
from assistant_ia.capabilities.context import build_capability_context
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationContext
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.identity.models import AssistantIdentity
from assistant_ia.intelligence.model_client import (
    ModelClient,
    OllamaModelClient,
)
from assistant_ia.intelligence.memory_candidates import (
    OllamaMemoryCandidateAnalyzer,
)
from assistant_ia.memory.journal_repository import JournalRepository
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.retrieval import ContextualMemoryRetriever
from assistant_ia.memory.repository import (
    DatabaseError,
    SQLiteDatabase,
    default_database_path,
)
from assistant_ia.memory.task_repository import TaskRepository
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person
from assistant_ia.security.confirmation import ConfirmationHandler
from assistant_ia.security.permissions import PermissionPolicy
from assistant_ia.runtime import AssistantRuntime, ResponsePresenter
from assistant_ia.system.windows import WindowsApplicationLauncher


class ApplicationInitializationError(RuntimeError):
    """Raised when the assistant application cannot be initialized."""


def build_default_assistant(
    database: SQLiteDatabase | None = None,
    model_client: ModelClient | None = None,
    context: ConversationContext | None = None,
    current_date: Callable[[], date] | None = None,
    permission_policy: PermissionPolicy | None = None,
    confirmation_handler: ConfirmationHandler | None = None,
    windows_launcher: WindowsApplicationLauncher | None = None,
    identity: AssistantIdentity | None = None,
    person_context: ActivePersonContext | None = None,
) -> AssistantCore:
    """Build a fully initialized assistant with available actions."""
    if database is not None and not isinstance(
        database,
        SQLiteDatabase,
    ):
        raise TypeError(
            "Assistant database must be a SQLiteDatabase."
        )

    if (
        identity is not None
        and not isinstance(identity, AssistantIdentity)
    ):
        raise TypeError(
            "Assistant identity must be an AssistantIdentity."
        )

    if identity is not None and model_client is not None:
        raise ValueError(
            "Assistant identity cannot be combined with "
            "an explicit model client."
        )

    if (
        person_context is not None
        and not isinstance(person_context, ActivePersonContext)
    ):
        raise TypeError(
            "Assistant person context must be an "
            "ActivePersonContext."
        )

    if (
        windows_launcher is not None
        and not isinstance(
            windows_launcher,
            WindowsApplicationLauncher,
        )
    ):
        raise TypeError(
            "Assistant Windows launcher must be a "
            "WindowsApplicationLauncher."
        )

    try:
        resolved_database = (
            database
            if database is not None
            else SQLiteDatabase(
                default_database_path()
            )
        )
        resolved_database.initialize()
    except DatabaseError as error:
        raise ApplicationInitializationError(
            "The assistant database could not be initialized."
        ) from error

    resolved_windows_launcher = (
        windows_launcher
        if windows_launcher is not None
        else WindowsApplicationLauncher()
    )

    memory_repository = MemoryRepository(
        resolved_database
    )

    action_registry = build_default_action_registry(
        task_repository=TaskRepository(
            resolved_database
        ),
        memory_repository=memory_repository,
        journal_repository=JournalRepository(
            resolved_database
        ),
        current_date=current_date,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        windows_launcher=resolved_windows_launcher,
    )

    capability_context = build_capability_context(
        action_registry,
        automatic_memory_retrieval=model_client is None,
    )

    resolved_identity = (
        identity
        if identity is not None
        else build_default_identity()
    )

    resolved_person_context = (
        person_context
        if person_context is not None
        else ActivePersonContext(
            assistant_name=resolved_identity.name,
            default_person=build_default_person(),
        )
    )

    if (
        resolved_person_context.assistant_name.casefold()
        != resolved_identity.name.casefold()
    ):
        raise ValueError(
            "Assistant person context name must match "
            "the assistant identity."
        )

    resolved_model_client = (
        model_client
        if model_client is not None
        else OllamaModelClient(
            identity=resolved_identity,
            person_context=resolved_person_context,
            capability_context=capability_context,
            contextual_memory_retriever=(
                ContextualMemoryRetriever(
                    memory_repository
                )
            ),
        )
    )

    return AssistantCore(
        model_client=resolved_model_client,
        context=context,
        action_registry=action_registry,
        person_context=resolved_person_context,
        memory_candidate_analyzer=(
            OllamaMemoryCandidateAnalyzer()
            if model_client is None
            else None
        ),
    )


def build_default_runtime(
    database: SQLiteDatabase | None = None,
    model_client: ModelClient | None = None,
    context: ConversationContext | None = None,
    current_date: Callable[[], date] | None = None,
    permission_policy: PermissionPolicy | None = None,
    confirmation_handler: ConfirmationHandler | None = None,
    windows_launcher: WindowsApplicationLauncher | None = None,
    identity: AssistantIdentity | None = None,
    person_context: ActivePersonContext | None = None,
    presenter: ResponsePresenter | None = None,
) -> AssistantRuntime:
    """Build the default assistant runtime with optional presentation."""
    assistant = build_default_assistant(
        database=database,
        model_client=model_client,
        context=context,
        current_date=current_date,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        windows_launcher=windows_launcher,
        identity=identity,
        person_context=person_context,
    )

    return AssistantRuntime(
        assistant=assistant,
        presenter=presenter,
    )
