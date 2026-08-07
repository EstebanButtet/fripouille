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
)
from assistant_ia.memory.task_repository import TaskRepository


class ApplicationInitializationError(RuntimeError):
    """Raised when the assistant application cannot be initialized."""


def build_default_assistant(
    database: SQLiteDatabase | None = None,
    model_client: ModelClient | None = None,
    context: ConversationContext | None = None,
    current_date: Callable[[], date] | None = None,
) -> AssistantCore:
    """Build a fully initialized assistant with persistent actions."""
    if database is not None and not isinstance(
        database,
        SQLiteDatabase,
    ):
        raise TypeError(
            "Assistant database must be a SQLiteDatabase."
        )

    try:
        resolved_database = (
            database
            if database is not None
            else SQLiteDatabase()
        )
        resolved_database.initialize()
    except DatabaseError as error:
        raise ApplicationInitializationError(
            "The assistant database could not be initialized."
        ) from error

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
    )

    return AssistantCore(
        model_client=model_client,
        context=context,
        action_registry=action_registry,
    )
