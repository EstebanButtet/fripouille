"""Implémentations par défaut des actions locales de Fripouille.

Les handlers métier traduisent des paramètres textuels déjà autorisés en
appels de repositories. Le handler système ajoute permission, confirmation et
liste blanche avant le lanceur Windows. :func:`build_default_action_registry`
assemble ensuite les seules actions effectivement exposées au coeur.

Le LLM ne connaît aucun de ces objets : il produit une intention, puis
``Action`` et ``ActionRegistry`` valident le contrat avant d'atteindre ici.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import cast

from assistant_ia.actions.action import (
    Action,
    ActionExecutionError,
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
from assistant_ia.security.confirmation import (
    ConfirmationHandler,
    ConfirmationRequest,
    deny_confirmation,
    request_confirmation,
)
from assistant_ia.security.permissions import (
    PermissionPolicy,
    build_default_permission_policy,
)
from assistant_ia.system.windows import (
    WindowsApplicationLaunchError,
    WindowsApplicationLauncher,
    WindowsApplicationNotAllowedError,
)


class _BusinessActionHandlers:
    """Exécuter les opérations persistantes après validation de l'action.

    Cette classe privée regroupe les dépendances pour fournir des méthodes
    liées utilisables comme handlers. Elle ne décide pas quels noms d'intention
    sont autorisés : cette décision appartient au registre.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        memory_repository: MemoryRepository,
        journal_repository: JournalRepository,
        current_date: Callable[[], date],
    ) -> None:
        """Conserver les repositories et le fournisseur de date locale."""
        self._task_repository = task_repository
        self._memory_repository = memory_repository
        self._journal_repository = journal_repository
        self._current_date = current_date

    def create_task(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Créer une tâche persistante et formater son accusé de réception."""
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
        """Lister les tâches persistantes avec un filtre de statut facultatif."""
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
        """Terminer la tâche choisie par son identifiant stable."""
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
        """Enregistrer un souvenir explicitement demandé par l'utilisateur."""
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
        """Chercher les souvenirs correspondant à une requête littérale."""
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
        """Supprimer le souvenir choisi par son identifiant stable."""
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
        """Écrire une entrée de journal à la date demandée ou courante."""
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


class _SystemActionHandlers:
    """Exécuter les actions système derrière les contrôles de sécurité.

    Le lanceur résout d'abord le nom dans sa liste blanche. La politique décide
    ensuite si l'action est refusée, autorisée ou soumise au handler de
    confirmation fourni par l'interface.
    """

    def __init__(
        self,
        permission_policy: PermissionPolicy,
        confirmation_handler: ConfirmationHandler,
        windows_launcher: WindowsApplicationLauncher,
    ) -> None:
        """Conserver politique, frontière de confirmation et lanceur Windows."""
        self._permission_policy = permission_policy
        self._confirmation_handler = confirmation_handler
        self._windows_launcher = windows_launcher

    def launch_application(
        self,
        parameters: Mapping[str, str],
    ) -> str:
        """Lancer une application explicitement présente dans la liste blanche.

        Une annulation produit une réponse normale sans effet. Une cible non
        autorisée est une erreur de validation ; un échec après autorisation
        devient une erreur d'exécution.
        """
        try:
            application = self._windows_launcher.resolve_application(
                parameters["application"]
            )
        except WindowsApplicationNotAllowedError as error:
            raise ActionValidationError(
                "Cette application n’est pas autorisée."
            ) from error

        # La résolution de la cible précède la confirmation : l'utilisateur
        # confirme ainsi un nom canonique et connu, pas une chaîne arbitraire.
        decision = self._permission_policy.decision_for(
            "launch_application"
        )

        if decision == "denied":
            raise ActionValidationError(
                "Le lancement d’applications n’est pas autorisé."
            )

        if decision == "confirmation_required":
            request = ConfirmationRequest(
                action_name="launch_application",
                description=(
                    f"lancer {application.display_name}"
                ),
            )

            try:
                confirmed = request_confirmation(
                    self._confirmation_handler,
                    request,
                )
            except TypeError as error:
                raise ActionExecutionError(
                    "La confirmation de l’action a échoué."
                ) from error

            if not confirmed:
                return (
                    f"Lancement annulé : "
                    f"{application.display_name}."
                )

        try:
            launched_application = self._windows_launcher.launch(
                application.name
            )
        except WindowsApplicationLaunchError as error:
            raise ActionExecutionError(
                "L’application n’a pas pu être lancée."
            ) from error

        return (
            f"Application lancée : "
            f"{launched_application.display_name}."
        )


def build_default_action_registry(
    task_repository: TaskRepository,
    memory_repository: MemoryRepository,
    journal_repository: JournalRepository,
    current_date: Callable[[], date] | None = None,
    permission_policy: PermissionPolicy | None = None,
    confirmation_handler: ConfirmationHandler | None = None,
    windows_launcher: WindowsApplicationLauncher | None = None,
) -> ActionRegistry:
    """Construire le registre des actions explicitement disponibles.

    Les actions persistantes sont toujours ajoutées avec leurs repositories.
    ``launch_application`` n'existe dans le registre que lorsqu'un lanceur
    Windows est fourni. Sans handler interactif, sa confirmation est refusée
    par défaut plutôt qu'accordée implicitement.
    """
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

    if (
        permission_policy is not None
        and not isinstance(permission_policy, PermissionPolicy)
    ):
        raise TypeError(
            "Default actions require a PermissionPolicy."
        )

    if (
        confirmation_handler is not None
        and not callable(confirmation_handler)
    ):
        raise TypeError(
            "Default action confirmation handler must be callable."
        )

    if (
        windows_launcher is not None
        and not isinstance(
            windows_launcher,
            WindowsApplicationLauncher,
        )
    ):
        raise TypeError(
            "Default actions require a WindowsApplicationLauncher."
        )

    business_handlers = _BusinessActionHandlers(
        task_repository=task_repository,
        memory_repository=memory_repository,
        journal_repository=journal_repository,
        current_date=(
            current_date
            if current_date is not None
            else _local_today
        ),
    )

    # Cette liste est la source concrète des capacités d'action annoncées au
    # modèle. Connaître un IntentName ne suffit jamais pour exécuter l'action.
    actions = [
        Action(
            name="create_task",
            handler=business_handlers.create_task,
        ),
        Action(
            name="list_tasks",
            handler=business_handlers.list_tasks,
        ),
        Action(
            name="complete_task",
            handler=business_handlers.complete_task,
        ),
        Action(
            name="save_memory",
            handler=business_handlers.save_memory,
        ),
        Action(
            name="find_memory",
            handler=business_handlers.find_memory,
        ),
        Action(
            name="delete_memory",
            handler=business_handlers.delete_memory,
        ),
        Action(
            name="write_journal",
            handler=business_handlers.write_journal,
        ),
    ]

    if windows_launcher is not None:
        system_handlers = _SystemActionHandlers(
            permission_policy=(
                permission_policy
                if permission_policy is not None
                else build_default_permission_policy()
            ),
            confirmation_handler=(
                confirmation_handler
                if confirmation_handler is not None
                else deny_confirmation
            ),
            windows_launcher=windows_launcher,
        )

        actions.append(
            Action(
                name="launch_application",
                handler=system_handlers.launch_application,
            )
        )

    return ActionRegistry(
        actions=actions
    )


def _parse_optional_due_at(
    value: str | None,
) -> datetime | None:
    """Interpréter une échéance ISO non ambiguë incluant son fuseau."""
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
    """Interpréter le filtre de statut d'une liste de tâches."""
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
    """Interpréter un identifiant positif dans les notations admises.

    Le parseur accepte quelques décorations utilisateur contrôlées mais refuse
    tout signe, exposant ou caractère non décimal avant d'appeler ``int``.
    """
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
    """Interpréter une date de journal strictement au format ISO 8601."""
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
    """Obtenir et valider la date locale fournie par l'application."""
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
    """Formater une date-heure UTC persistée pour l'affichage."""
    return value.strftime(
        "%d.%m.%Y à %H:%M UTC"
    )


def _format_task_list(
    tasks: tuple[Task, ...],
    *,
    status: TaskStatus | None,
) -> str:
    """Formater une liste de tâches déterministe avec statut et échéance."""
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
    """Retourner la date du fuseau local configuré sur l'ordinateur."""
    return datetime.now().astimezone().date()
