"""Assemblage des composants de l'application locale Fripouille.

Ce module est la *racine de composition* : il crée les objets concrets, relie
leurs dépendances et rend un :class:`AssistantCore` ou un
:class:`AssistantRuntime` prêt à l'emploi. Il reçoit éventuellement des
doubles de test ou des composants déjà construits ; sinon il choisit les
implémentations locales par défaut (SQLite, Ollama et actions Windows).

Il ne traite aucun message lui-même. Son rôle est uniquement de câbler le
flux suivant sans mélanger les responsabilités::

    interface -> AssistantRuntime -> AssistantCore
                                      |-> client Ollama
                                      |-> registre d'actions
                                      `-> services de mémoire
"""

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
from assistant_ia.memory.promotion import MemoryPromotionService
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
from assistant_ia.runtime import (
    AssistantRuntime,
    DiagnosticReporter,
    ResponsePresenter,
)
from assistant_ia.system.windows import WindowsApplicationLauncher


class ApplicationInitializationError(RuntimeError):
    """Signaler qu'une dépendance indispensable n'a pas pu être initialisée.

    L'exception masque ici les détails SQLite derrière une erreur de niveau
    application, tout en conservant l'erreur d'origine comme cause Python.
    """


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
    """Construire le coeur avec ses actions, sa mémoire et son modèle.

    Les paramètres optionnels réalisent une injection de dépendances : les
    tests peuvent fournir un client de modèle, une base ou une politique
    contrôlés, tandis que l'application normale reçoit les objets par défaut.
    La fonction initialise SQLite et peut lever
    :class:`ApplicationInitializationError` si cette étape échoue.

    Lorsque ``model_client`` est fourni, les automatismes spécifiques à
    Ollama (rappel et promotion de mémoire) ne sont pas ajoutés implicitement.
    Cela maintient un client injecté maître de son comportement de test.
    """
    # Les validations précoces rendent les erreurs d'assemblage explicites.
    # Elles ne constituent pas des validations de messages utilisateur.
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
        # Une même connexion logique est partagée par tous les repositories.
        # Ils restent ensuite responsables de traduire leur domaine en SQL.
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

    # Le registre est la frontière applicative qui autorise et valide les
    # actions. Le modèle ne reçoit jamais directement les repositories ni le
    # lanceur Windows.
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
        memory_promotion_service=(
            MemoryPromotionService(memory_repository)
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
    diagnostic_reporter: DiagnosticReporter | None = None,
) -> AssistantRuntime:
    """Envelopper un coeur assemblé dans la frontière d'interface runtime.

    Le ``presenter`` reçoit la réponse destinée à l'utilisateur ; le
    ``diagnostic_reporter`` reçoit séparément les informations techniques du
    tour. Aucun des deux ne participe à la décision métier du coeur.
    """
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
        diagnostic_reporter=diagnostic_reporter,
    )
