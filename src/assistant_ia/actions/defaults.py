"""Default persistent actions for the local assistant."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import cast

from assistant_ia.actions.action import (
    Action,
    ActionValidationError,
)
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.memory.errors import (
    MemoryNotFoundError,
    TaskAlreadyCompletedError,
    TaskNotFoundError,
)
from assistant_ia.memory.journal_repository import JournalRepository
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import (
    ALLOWED_TASK_STATUSES,
    Task,
    TaskStatus,
)
from assistant_ia.memory.task_repository import TaskRepository


class _BusinessActionHandlers:
    """Execute validated persistent assistant actions."""

    def __init__(
        self,
        task_repository: TaskRepository,
        memory_repository: MemoryRepository,
        journal_repository: JournalRepository,
        current_date: Callable[[], date],
    ) -> None:
        """Store the repositories used by the action handlers."""
        self._task_repository = task_repository
        self._memory_repository = memory_repository
        self._journal_repository = journal_repository
        self._current_date = current_date

    def create_task(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Create one persistent task."""
        due_at = _parse_optional_due_at(
            parameters.get("due_at")
        )
        task = self._task_repository.create_task(
            parameters["title"],
            due_at=due_at,
        )

        result = f"Tâche créée : [#{task.id}] {task.title}"

        if task.due_at is not None:
            result += (
                " — échéance : "
                f"{_format_datetime(task.due_at)}"
            )

        return f"{result}."

    def list_tasks(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """List persistent tasks using an optional status filter."""
        status = _parse_task_status(
            parameters.get("status", "pending")
        )
        tasks = self._task_repository.list_tasks(
            status=status,
        )

        return _format_task_list(
            tasks,
            status=status,
        )

    def complete_task(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Complete one task selected by its stable identifier."""
        task_id = _parse_positive_identifier(
            parameters["task_id"],
            field_name="L’identifiant de tâche",
        )

        try:
            task = self._task_repository.complete_task(task_id)
        except TaskNotFoundError as error:
            raise ActionValidationError(
                f"La tâche #{task_id} n’existe pas."
            ) from error
        except TaskAlreadyCompletedError as error:
            raise ActionValidationError(
                f"La tâche #{task_id} est déjà terminée."
            ) from error

        return (
            f"Tâche terminée : [#{task.id}] "
            f"{task.title}."
        )

    def save_memory(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Save one persistent assistant memory."""
        memory = self._memory_repository.save_memory(
            parameters["content"]
        )

        return (
            f"Souvenir enregistré : [#{memory.id}] "
            f"{memory.content}"
        )

    def find_memory(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Find memories matching one literal query."""
        memories = self._memory_repository.find_memories(
            parameters["query"]
        )

        if not memories:
            return "Aucun souvenir trouvé."

        lines = [
            "Souvenirs trouvés :",
            *(
                f"• [#{memory.id}] {memory.content}"
                for memory in memories
            ),
        ]

        return "\n".join(lines)

    def delete_memory(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Delete one memory selected by its stable identifier."""
        memory_id = _parse_positive_identifier(
            parameters["memory_id"],
            field_name="L’identifiant de souvenir",
        )

        try:
            memory = self._memory_repository.delete_memory(
                memory_id
            )
        except MemoryNotFoundError as error:
            raise ActionValidationError(
                f"Le souvenir #{memory_id} n’existe pas."
            ) from error

        return (
            f"Souvenir supprimé : [#{memory.id}] "
            f"{memory.content}"
        )

    def write_journal(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Write one persistent journal entry."""
        raw_entry_date = parameters.get("entry_date")

        if raw_entry_date is None:
            entry_date = _resolve_current_date(
                self._current_date
            )
        else:
            entry_date = _parse_entry_date(
                raw_entry_date
            )

        entry = self._journal_repository.write_journal_entry(
            parameters["content"],
            entry_date=entry_date,
        )

        return (
            "Entrée de journal enregistrée pour le "
            f"{entry.entry_date.isoformat()} : "
            f"[#{entry.id}] {entry.content}"
        )


def build_default_action_registry(
    task_repository: TaskRepository,
    memory_repository: MemoryRepository,
    journal_repository: JournalRepository,
    current_date: Callable[[], date] | None = None,
) -> ActionRegistry:
    """Build the seven persistent actions supported by the assistant."""
    if not isinstance(task_repository, TaskRepository):
        raise TypeError(
            "Default actions require a TaskRepository."
        )

    if not isinstance(memory_repository, MemoryRepository):
        raise TypeError(
            "Default actions require a MemoryRepository."
        )

    if not isinstance(journal_repository, JournalRepository):
        raise TypeError(
            "Default actions require a JournalRepository."
        )

    if current_date is not None and not callable(current_date):
        raise TypeError(
            "Default action current date provider must be callable."
        )

    handlers = _BusinessActionHandlers(
        task_repository=task_repository,
        memory_repository=memory_repository,
        journal_repository=journal_repository,
        current_date=(
            current_date
            if current_date is not None
            else _local_today
        ),
    )

    return ActionRegistry(
        actions=(
            Action(
                name="create_task",
                handler=handlers.create_task,
            ),
            Action(
                name="list_tasks",
                handler=handlers.list_tasks,
            ),
            Action(
                name="complete_task",
                handler=handlers.complete_task,
            ),
            Action(
                name="save_memory",
                handler=handlers.save_memory,
            ),
            Action(
                name="find_memory",
                handler=handlers.find_memory,
            ),
            Action(
                name="delete_memory",
                handler=handlers.delete_memory,
            ),
            Action(
                name="write_journal",
                handler=handlers.write_journal,
            ),
        )
    )


def _parse_optional_due_at(
    value: str | None,
) -> datetime | None:
    """Parse one unambiguous timezone-aware ISO task deadline."""
    if value is None:
        return None

    normalized_value = value

    if normalized_value.endswith(("Z", "z")):
        normalized_value = (
            normalized_value[:-1] + "+00:00"
        )

    try:
        due_at = datetime.fromisoformat(normalized_value)
    except ValueError as error:
        raise ActionValidationError(
            "La date d’échéance doit être un datetime "
            "ISO 8601 avec fuseau horaire."
        ) from error

    if due_at.tzinfo is None or due_at.utcoffset() is None:
        raise ActionValidationError(
            "La date d’échéance doit inclure un fuseau horaire."
        )

    return due_at


def _parse_task_status(
    value: str,
) -> TaskStatus | None:
    """Parse one task list status parameter."""
    normalized_value = value.lower()

    if normalized_value == "all":
        return None

    if normalized_value not in ALLOWED_TASK_STATUSES:
        raise ActionValidationError(
            "Le statut des tâches doit être "
            "pending, completed ou all."
        )

    return cast(TaskStatus, normalized_value)


def _parse_positive_identifier(
    value: str,
    *,
    field_name: str,
) -> int:
    """Parse a strictly positive identifier using safe notation."""
    normalized_value = value.strip()
    normalized_casefolded = normalized_value.casefold()

    for prefix in (
        "numéro",
        "numero",
        "n°",
    ):
        if normalized_casefolded.startswith(prefix):
            normalized_value = normalized_value[
                len(prefix):
            ].strip()
            break

    if normalized_value.startswith("#"):
        normalized_value = normalized_value[1:].strip()

    if normalized_value.endswith("."):
        normalized_value = normalized_value[:-1].strip()

    if (
        not normalized_value
        or any(
            character not in "0123456789"
            for character in normalized_value
        )
    ):
        raise ActionValidationError(
            f"{field_name} doit être un entier positif."
        )

    identifier = int(normalized_value)

    if identifier < 1:
        raise ActionValidationError(
            f"{field_name} doit être supérieur à zéro."
        )

    return identifier


def _parse_entry_date(value: str) -> date:
    """Parse one strict ISO 8601 journal date."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ActionValidationError(
            "La date du journal doit utiliser le format "
            "YYYY-MM-DD."
        ) from error


def _resolve_current_date(
    current_date: Callable[[], date],
) -> date:
    """Return a validated application-local current date."""
    resolved_date = current_date()

    if (
        isinstance(resolved_date, datetime)
        or not isinstance(resolved_date, date)
    ):
        raise ActionValidationError(
            "La date courante de l’application est invalide."
        )

    return resolved_date


def _format_datetime(value: datetime) -> str:
    """Format one persisted UTC datetime for terminal display."""
    return value.strftime(
        "%d.%m.%Y à %H:%M UTC"
    )


def _format_task_list(
    tasks: tuple[Task, ...],
    *,
    status: TaskStatus | None,
) -> str:
    """Format one deterministic task list for terminal display."""
    if not tasks:
        if status == "pending":
            return "Aucune tâche en attente."

        if status == "completed":
            return "Aucune tâche terminée."

        return "Aucune tâche enregistrée."

    if status == "pending":
        heading = "Tâches en attente :"
    elif status == "completed":
        heading = "Tâches terminées :"
    else:
        heading = "Toutes les tâches :"

    lines = [heading]

    for task in tasks:
        marker = "✓" if task.status == "completed" else "•"
        line = f"{marker} [#{task.id}] {task.title}"

        if task.due_at is not None:
            line += (
                " — échéance : "
                f"{_format_datetime(task.due_at)}"
            )

        lines.append(line)

    return "\n".join(lines)


def _local_today() -> date:
    """Return the date in the computer's configured local timezone."""
    return datetime.now().astimezone().date()
