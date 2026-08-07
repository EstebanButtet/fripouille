"""Application assembly for the local personal assistant."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from assistant_ia.actions.defaults import (
    build_default_action_registry,
)
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationContext
from assistant_ia.intelligence.model_client import ModelClient
from assistant_ia.memory.journal_repository import JournalRepository
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import (
    DatabaseError,
    SQLiteDatabase,
    default_database_path,
)
from assistant_ia.memory.task_repository import TaskRepository
from assistant_ia.security.confirmation import ConfirmationHandler
from assistant_ia.security.permissions import PermissionPolicy
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

    action_registry = build_default_action_registry(
        task_repository=TaskRepository(
            resolved_database
        ),
        memory_repository=MemoryRepository(
            resolved_database
        ),
        journal_repository=JournalRepository(
            resolved_database
        ),
        current_date=current_date,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        windows_launcher=resolved_windows_launcher,
    )

    return AssistantCore(
        model_client=model_client,
        context=context,
        action_registry=action_registry,
    )
