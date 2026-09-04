"""Orchestration métier d'un tour de conversation de Fripouille.

``AssistantCore`` reçoit un message déjà acquis par une interface. Il gère
l'historique temporaire, résout les présentations explicites, demande une
réponse structurée au client de modèle, fait exécuter les intentions
autorisées par ``ActionRegistry`` et orchestre l'analyse puis les promotions
éventuelles de candidats mémoire ou profil.

Frontière d'autorité essentielle::

    message -> ModelClient (proposition d'intention)
                -> AssistantCore (choix du chemin)
                    -> ActionRegistry (validation et permission)
                    -> réponse visible

Le modèle ne reçoit donc jamais un accès direct aux actions, à SQLite ou au
hardware. Ce module ne présente pas non plus la réponse dans une interface :
cette dernière étape appartient à :mod:`assistant_ia.runtime`.
"""

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
from assistant_ia.intelligence.profile_fact_candidates import (
    ProfileFactCandidateAnalysisError,
    ProfileFactCandidateAnalyzer,
)
from assistant_ia.intelligence.model_client import (
    ModelClient,
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.promotion import (
    MemoryPromotionProposal,
    MemoryPromotionService,
    normalize_memory_equivalence,
)
from assistant_ia.memory.repository import DatabaseError
from assistant_ia.memory.subjects import candidate_targets_active_person
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person
from assistant_ia.people.models import PersonProfile
from assistant_ia.people.profile_models import ProfileFactCandidate
from assistant_ia.people.profile_promotion import (
    ProfileFactPromotionProposal,
    ProfileFactPromotionService,
    normalize_profile_fact_equivalence,
)
from assistant_ia.people.presentation import detect_presented_person
from assistant_ia.people.resolution import (
    PersonResolution,
    PersonResolutionError,
    PersonResolutionService,
)

ACTION_UNAVAILABLE_MESSAGE = (
    "J’ai identifié votre demande, mais aucune action n’a été exécutée. "
    "Cette action n’est pas disponible."
)

ACTION_EXECUTION_ERROR_MESSAGE = (
    "L’action n’a pas pu être confirmée. "
    "Vérifiez l’état enregistré avant de réessayer."
)

JOURNAL_CONTENT_SEPARATOR = " : "

MEMORY_PROMOTION_ERROR_MESSAGE = (
    "La mémoire n'a pas pu être modifiée. "
    "Vérifiez les souvenirs enregistrés avant de réessayer."
)

PROFILE_PROMOTION_ERROR_MESSAGE = (
    "Le profil n'a pas pu être modifié. "
    "Vérifiez les faits enregistrés avant de réessayer."
)

PERSON_RESOLUTION_ERROR_MESSAGE = (
    "La personne n'a pas pu être résolue. "
    "L'interlocuteur actif n'a pas changé."
)

_MEMORY_CONFIRMATION_ACCEPTED = frozenset(
    {"d accord", "ok", "oui", "vas y", "yes"}
)
_MEMORY_CONFIRMATION_REFUSED = frozenset(
    {
        "annule",
        "n enregistre pas",
        "ne la garde pas",
        "no",
        "non",
        "refuse",
    }
)

_PROFILE_CONFIRMATION_ACCEPTED = _MEMORY_CONFIRMATION_ACCEPTED
_PROFILE_CONFIRMATION_REFUSED = _MEMORY_CONFIRMATION_REFUSED


class AssistantCoreError(RuntimeError):
    """Traduire un échec d'orchestration en erreur stable du coeur."""


class AssistantCore:
    """Coordonner messages, propositions, actions, mémoire et profil.

    Les dépendances sont injectables pour isoler chaque responsabilité dans
    les tests. L'instance conserve le contexte et les résultats du dernier
    tour ; elle doit donc vivre pendant toute une session d'interface.
    """

    def __init__(
        self,
        model_client: ModelClient | None = None,
        context: ConversationContext | None = None,
        action_registry: ActionRegistry | None = None,
        person_context: ActivePersonContext | None = None,
        memory_candidate_analyzer: (
            MemoryCandidateAnalyzer | None
        ) = None,
        memory_promotion_service: (
            MemoryPromotionService | None
        ) = None,
        profile_fact_candidate_analyzer: (
            ProfileFactCandidateAnalyzer | None
        ) = None,
        profile_fact_promotion_service: (
            ProfileFactPromotionService | None
        ) = None,
        person_resolution_service: (
            PersonResolutionService | None
        ) = None,
    ) -> None:
        """Créer le coeur avec les dépendances fournies ou leurs valeurs par défaut.

        Les champs ``last_*`` sont des diagnostics du dernier tour. Le champ
        ``_pending_memory_promotion`` est différent : il porte un état métier
        temporaire qui attend une réponse explicite oui/non de l'utilisateur.
        """
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

        if (
            person_resolution_service is not None
            and not isinstance(
                person_resolution_service,
                PersonResolutionService,
            )
        ):
            raise TypeError(
                "Assistant person resolution service must be a "
                "PersonResolutionService."
            )

        # Cette identité est stable et sert seulement aux valeurs par défaut.
        # Aucune réponse du modèle ne peut la modifier.
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
        # Les objets d'analyse et de promotion sont séparés : le premier ne
        # persiste rien, le second applique les règles applicatives de mémoire.
        self._last_intent: Intent | None = None
        self._person_resolution_service = person_resolution_service
        self._last_person_resolution: PersonResolution | None = None
        self._memory_candidate_analyzer = memory_candidate_analyzer
        self._memory_promotion_service = memory_promotion_service
        self._last_memory_candidates: tuple[
            MemoryCandidate, ...
        ] = ()
        self._last_memory_promotion_proposal: (
            MemoryPromotionProposal | None
        ) = None
        self._pending_memory_promotion: (
            MemoryPromotionProposal | None
        ) = None
        self._profile_fact_candidate_analyzer = profile_fact_candidate_analyzer
        self._profile_fact_promotion_service = profile_fact_promotion_service
        self._last_profile_fact_candidates: tuple[
            ProfileFactCandidate, ...
        ] = ()
        self._last_profile_fact_promotion_proposal: (
            ProfileFactPromotionProposal | None
        ) = None
        self._pending_profile_fact_promotion: (
            ProfileFactPromotionProposal | None
        ) = None

    @property
    def context(self) -> ConversationContext:
        """Retourner l'historique temporaire géré par le coeur."""
        return self._context

    @property
    def person_context(self) -> ActivePersonContext:
        """Retourner la personne actuellement active dans la conversation."""
        return self._person_context

    @property
    def action_registry(self) -> ActionRegistry:
        """Retourner le registre contrôlé des actions exécutables."""
        return self._action_registry

    @property
    def last_intent(self) -> Intent | None:
        """Retourner la dernière intention structurée identifiée."""
        return self._last_intent

    @property
    def last_person_resolution(self) -> PersonResolution | None:
        """Retourner le dernier résultat déterministe de présentation."""
        return self._last_person_resolution

    @property
    def last_memory_candidates(
        self,
    ) -> tuple[MemoryCandidate, ...]:
        """Retourner les candidats mémoire validés mais non persistés du tour."""
        return self._last_memory_candidates

    @property
    def last_memory_promotion_proposal(
        self,
    ) -> MemoryPromotionProposal | None:
        """Retourner la dernière proposition de promotion créée par l'application."""
        return self._last_memory_promotion_proposal

    @property
    def pending_memory_promotion(
        self,
    ) -> MemoryPromotionProposal | None:
        """Retourner l'unique proposition qui attend un consentement explicite."""
        return self._pending_memory_promotion

    @property
    def last_profile_fact_candidates(
        self,
    ) -> tuple[ProfileFactCandidate, ...]:
        """Retourner les candidats de profil non persistés du dernier tour."""
        return self._last_profile_fact_candidates

    @property
    def last_profile_fact_promotion_proposal(
        self,
    ) -> ProfileFactPromotionProposal | None:
        """Retourner la dernière proposition applicative de profil."""
        return self._last_profile_fact_promotion_proposal

    @property
    def pending_profile_fact_promotion(
        self,
    ) -> ProfileFactPromotionProposal | None:
        """Retourner la proposition précise qui attend un oui ou un non."""
        return self._pending_profile_fact_promotion

    def process_message(self, user_message: str) -> str:
        """Traiter un message et retourner le texte brut résolu du coeur.

        La méthode peut écrire via une action autorisée ou une promotion de
        mémoire confirmée. Les erreurs Ollama sont converties en
        :class:`AssistantCoreError`; les erreurs contrôlées d'action et de
        mémoire deviennent des réponses stables pour l'utilisateur.
        """
        # 1. Une confirmation profil ou mémoire en attente a priorité sur
        # Ollama et les actions. Un « oui » ne doit jamais être réinterprété
        # comme un nouvel ordre général par le modèle.
        pending_response = self._resolve_pending_profile_confirmation(
            user_message
        )
        if pending_response is not None:
            return pending_response

        pending_response = self._resolve_pending_memory_confirmation(
            user_message
        )
        if pending_response is not None:
            return pending_response

        # 2. Sans confirmation reconnue, le message commence un tour normal.
        # Les diagnostics du tour précédent sont remis à zéro avant l'analyse.
        self._pending_memory_promotion = None
        self._pending_profile_fact_promotion = None
        self._last_intent = None
        self._last_person_resolution = None
        self._last_memory_candidates = ()
        self._last_memory_promotion_proposal = None
        self._last_profile_fact_candidates = ()
        self._last_profile_fact_promotion_proposal = None
        current_user_message = self._context.add_user_message(user_message)

        # La détection de présentation est déterministe et limitée au message
        # explicite ; elle n'autorise pas le modèle à inventer une identité.
        presented_person = detect_presented_person(
            user_message
        )

        if (
            presented_person is not None
            and presented_person.name.casefold()
            != self._person_context.assistant_name.casefold()
        ):
            resolution_response = self._resolve_person_presentation(
                presented_person
            )

            if resolution_response is not None:
                self._context.add_assistant_message(
                    resolution_response
                )
                return resolution_response

        # 3. Le client produit à la fois un texte et une intention structurée.
        # Le coeur ne fait confiance qu'aux objets déjà validés par ce client.
        try:
            model_response = self._model_client.generate_response(
                self._context.messages,
            )
        except ModelClientError as error:
            raise AssistantCoreError(
                "The local language model could not produce a response."
            ) from error

        # Cette récupération ciblée conserve un contenu de journal explicite
        # lorsque le JSON du modèle a omis ce paramètre, sans élargir l'intent.
        resolved_intent = _recover_missing_journal_content(
            model_response.intent,
            user_message,
        )
        self._last_intent = resolved_intent

        # 4. Une conversation utilise le texte généré ; une intention d'action
        # passe obligatoirement par le registre applicatif.
        assistant_content = self._resolve_assistant_content(
            model_response=model_response,
            intent=resolved_intent,
        )

        # 5. Un candidat de profil exige un sujet persistant résolu par
        # l'application. Le modèle d'analyse ne reçoit jamais cet ID dans son
        # schéma de sortie et aucune proposition n'écrit directement SQLite.
        active_person_id = self._person_context.active_person_id
        if (
            resolved_intent.name == "conversation"
            and active_person_id is not None
            and self._profile_fact_candidate_analyzer is not None
        ):
            try:
                self._last_profile_fact_candidates = (
                    self._profile_fact_candidate_analyzer.analyze(
                        current_user_message.content,
                        person_id=active_person_id,
                    )
                )
            except ProfileFactCandidateAnalysisError:
                self._last_profile_fact_candidates = ()

        if (
            self._last_profile_fact_candidates
            and self._profile_fact_promotion_service is not None
        ):
            try:
                profile_proposal = self._select_profile_fact_proposal(
                    self._last_profile_fact_candidates
                )
            except (DatabaseError, RepositoryError):
                profile_proposal = None
            if profile_proposal is not None:
                self._last_profile_fact_promotion_proposal = profile_proposal
                if profile_proposal.requires_confirmation:
                    self._pending_profile_fact_promotion = profile_proposal
                assistant_content = (
                    assistant_content
                    + "\n\n"
                    + _render_profile_fact_proposal(profile_proposal)
                )

        # 6. Une information reconnue comme profil ne devient pas en parallèle
        # un souvenir général. L'analyse mémoire conserve sinon son ancien
        # comportement, uniquement pour les échanges conversationnels.
        if (
            resolved_intent.name == "conversation"
            and not self._last_profile_fact_candidates
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

        # 7. Un candidat n'est pas encore un souvenir. Le service applicatif
        # décide s'il est nouveau, connu, similaire ou contradictoire, puis
        # formule au plus une proposition contrôlée.
        if (
            self._last_memory_candidates
            and self._memory_promotion_service is not None
        ):
            try:
                proposal = self._select_memory_promotion_proposal(
                    self._last_memory_candidates
                )
            except (DatabaseError, RepositoryError):
                proposal = None
            if proposal is not None:
                self._last_memory_promotion_proposal = proposal
                if proposal.requires_confirmation:
                    self._pending_memory_promotion = proposal
                assistant_content = (
                    assistant_content
                    + "\n\n"
                    + _render_memory_promotion_proposal(proposal)
                )

        # 8. Seul le contenu effectivement renvoyé est ajouté à l'historique.
        # Les diagnostics internes restent hors du contexte envoyé à Ollama.
        self._context.add_assistant_message(
            assistant_content
        )

        return assistant_content

    def reset_conversation(self) -> None:
        """Réinitialiser la session temporaire sans effacer SQLite.

        La personne active, les diagnostics et une confirmation en attente
        sont oubliés. Les souvenirs, tâches et entrées de journal persistent.
        """
        self._context.clear()
        self._person_context.reset()
        self._last_intent = None
        self._last_person_resolution = None
        self._last_memory_candidates = ()
        self._last_memory_promotion_proposal = None
        self._pending_memory_promotion = None
        self._last_profile_fact_candidates = ()
        self._last_profile_fact_promotion_proposal = None
        self._pending_profile_fact_promotion = None

    def _resolve_person_presentation(
        self,
        presented_person: PersonProfile,
    ) -> str | None:
        """Résoudre puis appliquer une présentation avant l'appel modèle.

        ``None`` autorise la poursuite normale du tour. Un texte retourné est
        une réponse applicative finale : aucun LLM n'est alors consulté.
        """
        if self._person_resolution_service is None:
            return None

        try:
            resolution = (
                self._person_resolution_service.resolve_presentation(
                    presented_person
                )
            )
        except (
            DatabaseError,
            RepositoryError,
            PersonResolutionError,
        ):
            return PERSON_RESOLUTION_ERROR_MESSAGE

        self._last_person_resolution = resolution

        if resolution.person is not None:
            try:
                self._person_context.set_active_persistent_person(
                    resolution.person
                )
            except ValueError:
                return PERSON_RESOLUTION_ERROR_MESSAGE

            return None

        if resolution.status == "ambiguous":
            return (
                "Plusieurs personnes enregistrées portent le nom "
                f"« {presented_person.name} ». "
                "L'interlocuteur actif n'a pas changé."
            )

        if resolution.status == "stale":
            return (
                "Le registre des personnes a changé pendant la "
                "confirmation. L'interlocuteur actif n'a pas changé."
            )

        return (
            "D'accord, je ne crée pas de nouvelle personne pour "
            f"« {presented_person.name} ». "
            "L'interlocuteur actif n'a pas changé."
        )

    def _select_profile_fact_proposal(
        self,
        candidates: tuple[ProfileFactCandidate, ...],
    ) -> ProfileFactPromotionProposal | None:
        """Choisir au plus une proposition de profil, sans persister."""
        if self._profile_fact_promotion_service is None:
            return None
        known: ProfileFactPromotionProposal | None = None
        for candidate in candidates:
            proposal = self._profile_fact_promotion_service.propose(candidate)
            if proposal.requires_confirmation:
                return proposal
            if known is None:
                known = proposal
        return known

    def _resolve_pending_profile_confirmation(
        self,
        user_message: str,
    ) -> str | None:
        """Appliquer seulement un oui/non à la proposition de profil précise."""
        proposal = self._pending_profile_fact_promotion
        if proposal is None:
            return None
        normalized = normalize_profile_fact_equivalence(user_message)
        if normalized not in (
            _PROFILE_CONFIRMATION_ACCEPTED | _PROFILE_CONFIRMATION_REFUSED
        ):
            return None

        self._context.add_user_message(user_message)
        self._pending_profile_fact_promotion = None
        self._last_profile_fact_candidates = ()
        self._last_profile_fact_promotion_proposal = proposal
        self._last_memory_candidates = ()
        self._last_memory_promotion_proposal = None
        self._pending_memory_promotion = None
        self._last_intent = None
        self._last_person_resolution = None

        if normalized in _PROFILE_CONFIRMATION_REFUSED:
            response = "D'accord, je n'ajoute pas cette information au profil."
        elif (
            self._person_context.active_person_id
            != proposal.candidate.person_id
        ):
            response = PROFILE_PROMOTION_ERROR_MESSAGE
        elif self._profile_fact_promotion_service is None:
            response = PROFILE_PROMOTION_ERROR_MESSAGE
        else:
            try:
                fact = self._profile_fact_promotion_service.apply_confirmed(
                    proposal
                )
            except (DatabaseError, RepositoryError, ValueError):
                response = PROFILE_PROMOTION_ERROR_MESSAGE
            else:
                if proposal.operation in {"update", "conflict"}:
                    response = (
                        f"Fait de profil corrigé : [#{fact.id}] "
                        f"{fact.content}"
                    )
                else:
                    response = (
                        f"Fait de profil enregistré : [#{fact.id}] "
                        f"{fact.content}"
                    )

        self._context.add_assistant_message(response)
        return response

    def _select_memory_promotion_proposal(
        self,
        candidates: tuple[MemoryCandidate, ...],
    ) -> MemoryPromotionProposal | None:
        """Choisir au plus une proposition actionnable parmi les candidats.

        Une demande nécessitant confirmation est prioritaire. Sinon la
        première proposition informative (« déjà connu » par exemple) est
        conservée. La méthode ne modifie jamais la base.
        """
        if self._memory_promotion_service is None:
            return None

        known_proposal: MemoryPromotionProposal | None = None
        for candidate in candidates:
            active_person_id = self._person_context.active_person_id
            subject_person_id = (
                active_person_id
                if (
                    active_person_id is not None
                    and candidate_targets_active_person(candidate)
                )
                else None
            )
            proposal = self._memory_promotion_service.propose(
                candidate,
                subject_person_id=subject_person_id,
            )
            if proposal.requires_confirmation:
                return proposal
            if known_proposal is None:
                known_proposal = proposal
        return known_proposal

    def _resolve_pending_memory_confirmation(
        self,
        user_message: str,
    ) -> str | None:
        """Traiter un oui/non explicite sans invoquer modèle ni registre.

        Retourne ``None`` si aucune proposition n'attend ou si le message ne
        correspond pas aux réponses bornées. Une acceptation peut persister
        la proposition ; un refus ne touche pas à SQLite.
        """
        proposal = self._pending_memory_promotion
        if proposal is None:
            return None

        normalized_response = normalize_memory_equivalence(user_message)
        if normalized_response not in (
            _MEMORY_CONFIRMATION_ACCEPTED
            | _MEMORY_CONFIRMATION_REFUSED
        ):
            return None

        self._context.add_user_message(user_message)
        self._pending_memory_promotion = None
        self._last_memory_candidates = ()
        self._last_memory_promotion_proposal = proposal
        self._last_intent = None
        self._last_person_resolution = None

        if normalized_response in _MEMORY_CONFIRMATION_REFUSED:
            response = "D'accord, je ne conserverai pas cette information."
        elif (
            proposal.subject_person_id is not None
            and self._person_context.active_person_id
            != proposal.subject_person_id
        ):
            response = MEMORY_PROMOTION_ERROR_MESSAGE
        elif self._memory_promotion_service is None:
            response = MEMORY_PROMOTION_ERROR_MESSAGE
        else:
            try:
                memory = self._memory_promotion_service.apply_confirmed(
                    proposal
                )
            except (DatabaseError, RepositoryError, ValueError):
                response = MEMORY_PROMOTION_ERROR_MESSAGE
            else:
                if proposal.operation in {"update", "conflict"}:
                    response = (
                        f"Souvenir corrigé : [#{memory.id}] "
                        f"{memory.content}"
                    )
                else:
                    response = (
                        f"Souvenir enregistré : [#{memory.id}] "
                        f"{memory.content}"
                    )

        self._context.add_assistant_message(response)
        return response

    def _resolve_assistant_content(
        self,
        *,
        model_response: ModelResponse,
        intent: Intent,
    ) -> str:
        """Résoudre le contenu visible à partir d'une réponse structurée.

        Une intention conversationnelle garde le texte du modèle. Toute
        intention exécutable passe par ``ActionRegistry`` ; les noms inconnus,
        validations et erreurs d'exécution suivent des réponses contrôlées.
        """
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
    """Récupérer un contenu de journal explicite omis par le modèle.

    Cette correction déterministe ne s'applique qu'à ``write_journal`` et au
    séparateur public prévu. Elle retourne un nouvel :class:`Intent` immuable
    plutôt que de modifier celui produit par le client.
    """
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


def _render_profile_fact_proposal(
    proposal: ProfileFactPromotionProposal,
) -> str:
    """Présenter la décision de profil sans déclencher d'écriture."""
    candidate = proposal.candidate
    related = proposal.related_fact
    if proposal.operation == "create":
        return (
            "Cette information pourrait être ajoutée au profil de "
            f"l'interlocuteur actif : « {candidate.content} » "
            "Veux-tu que je l'enregistre ? Réponds oui ou non."
        )
    if related is None:
        raise ValueError("Profile promotion proposal is incomplete.")
    if proposal.operation == "already_known":
        return f"Ce fait figure déjà dans le profil [#{related.id}]."
    if proposal.operation == "update":
        return (
            f"Cela ressemble à une correction du fait de profil [#{related.id}] "
            f"« {related.content} ». Veux-tu le remplacer par "
            f"« {candidate.content} » ? Réponds oui ou non."
        )
    return (
        f"Cette information peut contredire le fait de profil [#{related.id}] "
        f"« {related.content} ». Veux-tu le remplacer par "
        f"« {candidate.content} » ? Réponds oui ou non."
    )


def _render_memory_promotion_proposal(
    proposal: MemoryPromotionProposal,
) -> str:
    """Formuler une proposition mémoire contrôlée sans lui donner d'autorité.

    Le texte explique l'opération calculée par l'application. Il ne déclenche
    aucune écriture : ``AssistantCore`` attend encore le consentement lorsque
    ``requires_confirmation`` est vrai.
    """
    candidate_content = proposal.candidate.content
    related_memory = proposal.related_memory

    if proposal.operation == "create":
        return (
            "Ça ressemble à une information utile à retenir : "
            f"« {candidate_content} » Veux-tu que je la garde ? "
            "Réponds oui ou non."
        )

    if related_memory is None:
        raise ValueError("Memory promotion proposal is incomplete.")

    if proposal.operation == "already_known":
        return (
            "Je connais déjà cette information dans le souvenir "
            f"[#{related_memory.id}]."
        )

    if proposal.operation == "possible_duplicate":
        return (
            "Cette information ressemble au souvenir "
            f"[#{related_memory.id}] « {related_memory.content} ». "
            "Veux-tu tout de même la conserver séparément ? "
            "Réponds oui ou non."
        )

    if proposal.operation == "update":
        return (
            "Ça ressemble à une correction du souvenir "
            f"[#{related_memory.id}] « {related_memory.content} ». "
            f"Veux-tu le remplacer par « {candidate_content} » ? "
            "Réponds oui ou non."
        )

    return (
        "Cette information peut contredire le souvenir "
        f"[#{related_memory.id}] « {related_memory.content} ». "
        f"Veux-tu le remplacer par « {candidate_content} » ? "
        "Réponds oui ou non."
    )
