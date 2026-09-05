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
                                      |-> résolution de personne
                                      |-> services de mémoire
                                      |-> contexte social borné
                                      `-> apprentissage explicite
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from dataclasses import replace

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
from assistant_ia.intelligence.profile_fact_candidates import (
    OllamaProfileFactCandidateAnalyzer,
)
from assistant_ia.learning.repository import BehavioralLearningRepository
from assistant_ia.learning.service import BehavioralLearningService
from assistant_ia.memory.errors import RepositoryError
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
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.observation_repository import ObservationRepository
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.profile_promotion import ProfileFactPromotionService
from assistant_ia.people.relationship_repository import (
    PersonRelationshipRepository,
)
from assistant_ia.people.resolution import PersonResolutionService
from assistant_ia.people.social_context import PersonSocialContextProvider
from assistant_ia.security.confirmation import ConfirmationHandler
from assistant_ia.security.permissions import PermissionPolicy
from assistant_ia.runtime import (
    AssistantRuntime,
    DiagnosticReporter,
    ResponsePresenter,
)
from assistant_ia.system.windows import WindowsApplicationLauncher
from assistant_ia.roles import RoleService
from assistant_ia.internal_state import InternalStateService
from assistant_ia.social_vision import SocialVisionService
from assistant_ia.cognitive_context import CognitiveContextProvider


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
    audio_input: bool = False,
    social_vision: SocialVisionService | None = None,
) -> AssistantCore:
    """Construire le coeur avec ses actions, sa mémoire et son modèle.

    Les paramètres optionnels réalisent une injection de dépendances : les
    tests peuvent fournir un client de modèle, une base ou une politique
    contrôlés, tandis que l'application normale reçoit les objets par défaut.
    La fonction initialise SQLite et peut lever
    :class:`ApplicationInitializationError` si cette étape échoue.

    Lorsque ``model_client`` est fourni, les automatismes spécifiques à
    Ollama (rappel, mémoire et candidats de profil) ne sont pas ajoutés
    implicitement.
    Cela maintient un client injecté maître de son comportement de test.
    """
    # Les validations précoces rendent les erreurs d'assemblage explicites.
    # Elles ne constituent pas des validations de messages utilisateur.
    if not isinstance(audio_input, bool):
        raise TypeError("Audio input capability must be boolean.")
    if social_vision is not None and not isinstance(social_vision, SocialVisionService):
        raise TypeError("Social vision must be an application service.")
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

        person_repository = PersonRepository(resolved_database)
        default_persistent_person = person_repository.get_person(
            DEFAULT_PERSON_ID
        )

        if default_persistent_person is None:
            raise RepositoryError(
                "Default persistent person does not exist."
            )
    except (DatabaseError, RepositoryError) as error:
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
    profile_fact_repository = ProfileFactRepository(resolved_database)

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
    capability_context = replace(capability_context, audio_input=audio_input)

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
            default_person=default_persistent_person,
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

    internal_state = InternalStateService()
    vision = social_vision if social_vision is not None else SocialVisionService()
    roles = RoleService(lambda: action_registry.action_names | {"conversation"}
                        | ({"social_vision"} if vision.snapshot.status in {"present", "absent"} else set()))
    learning_repository = BehavioralLearningRepository(resolved_database)
    cognition = CognitiveContextProvider(internal_state, roles, vision, learning_repository)
    resolved_model_client = (
        model_client
        if model_client is not None
        else OllamaModelClient(
            cognitive_context_provider=cognition,
            identity=resolved_identity,
            person_context=resolved_person_context,
            capability_context=capability_context,
            contextual_memory_retriever=(
                ContextualMemoryRetriever(
                    memory_repository
                )
            ),
            social_context_provider=PersonSocialContextProvider(
                profile_fact_repository,
                PersonRelationshipRepository(resolved_database),
                ObservationRepository(resolved_database),
            ),
        )
    )

    return AssistantCore(
        internal_state=internal_state,
        social_vision=vision,
        roles=roles,
        model_client=resolved_model_client,
        context=context,
        action_registry=action_registry,
        person_context=resolved_person_context,
        person_resolution_service=PersonResolutionService(
            person_repository,
            confirmation_handler=confirmation_handler,
        ),
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
        profile_fact_candidate_analyzer=(
            OllamaProfileFactCandidateAnalyzer()
            if model_client is None
            else None
        ),
        profile_fact_promotion_service=(
            ProfileFactPromotionService(profile_fact_repository)
            if model_client is None
            else None
        ),
        behavioral_learning_service=BehavioralLearningService(
            learning_repository,
            resolved_person_context,
            role_id_provider=lambda: roles.active_id,
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
    audio_input: bool = False,
    social_vision: SocialVisionService | None = None,
) -> AssistantRuntime:
    """Envelopper un coeur assemblé dans la frontière d'interface runtime.

    Le ``presenter`` reçoit la réponse destinée à l'utilisateur ; le
    ``diagnostic_reporter`` reçoit séparément les informations techniques du
    tour. Aucun des deux ne participe à la décision métier du coeur.
    """
    assistant = build_default_assistant(
        audio_input=audio_input,
        social_vision=social_vision,
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
