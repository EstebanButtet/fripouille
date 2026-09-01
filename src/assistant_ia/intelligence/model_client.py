"""Contrat de modèle de langage et implémentation HTTP pour Ollama.

``OllamaModelClient`` transforme un historique en un tour borné, effectue
d'abord une requête structurée d'interprétation, puis choisit soit une réponse
d'action, soit une génération conversationnelle. Le rappel mémoire n'arrive
qu'après la décision conversationnelle et ses résultats sont injectés comme
contexte non autoritatif.

Flux simplifié::

    ConversationTurn -> interprétation JSON validée
        | intention d'action -> ModelResponse pour AssistantCore
        ` conversation      -> rappel mémoire -> génération naturelle

Ce client propose des intentions ; il ne les exécute jamais et n'écrit jamais
dans les repositories.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant_ia.capabilities.context import CapabilityContext
from assistant_ia.intelligence.allocation import (
    ALLOCATION_PROPOSAL_SCHEMA,
    AllocationFormatError,
    AllocationMismatchError,
    AllocationTarget,
    FixedTotalAllocation,
    parse_allocation_proposal_json,
)
from assistant_ia.core.context import ConversationMessage
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.identity.models import AssistantIdentity
from assistant_ia.intelligence.conversation import (
    ConversationDirectiveProposal,
    ConversationDirectiveResolutionError,
    ConversationMode,
    resolve_conversation_directive,
)
from assistant_ia.intelligence.intent import (
    Intent,
    IntentName,
)
from assistant_ia.intelligence.interpretation import TurnInterpretation
from assistant_ia.intelligence.prompt import (
    CURRENT_TURN_CONTEXT_PROMPT,
    INTERPRETATION_RESPONSE_SCHEMA,
    INTENT_RESPONSE_SCHEMA,
    INTENT_SYSTEM_PROMPT,
    build_allocation_prompt,
    build_conversation_prompt,
    build_interpretation_prompt,
)
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.intelligence.turn import (
    ConversationTurn,
    build_conversation_turn,
)
from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import DatabaseError
from assistant_ia.memory.retrieval import (
    MAX_INJECTED_CONTEXTUAL_MEMORIES,
    ContextualMemoryRetriever,
    RetrievedMemory,
    bound_contextual_memories,
)
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0
DEFAULT_OLLAMA_CONTEXT_LENGTH = 4096
DEFAULT_OLLAMA_KEEP_ALIVE = "10m"

_OLLAMA_CHAT_PATH = "/api/chat"

_ACTION_INTERPRETED_CONTENT = (
    "Demande d'action interprétée."
)


class ModelClientError(RuntimeError):
    """Signaler une indisponibilité ou une réponse invalide du modèle."""


class ModelClient(Protocol):
    """Contrat minimal attendu par ``AssistantCore`` pour un modèle.

    Grâce au ``Protocol``, un test peut fournir n'importe quel objet possédant
    ``generate_response`` sans hériter d'une classe concrète Ollama.
    """

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Produire une réponse structurée depuis un historique ordonné."""
        ...


class OllamaModelClient:
    """Interpréter puis générer via l'API locale de dialogue Ollama.

    L'instance conserve la configuration réseau et les contextes stables,
    mais pas l'historique : celui-ci est fourni à chaque appel par le coeur.
    ``identity`` reste immuable et séparée des souvenirs et de la personne
    actuellement active.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        identity: AssistantIdentity | None = None,
        person_context: ActivePersonContext | None = None,
        capability_context: CapabilityContext | None = None,
        contextual_memory_retriever: (
            ContextualMemoryRetriever | None
        ) = None,
    ) -> None:
        """Créer le client avec une configuration locale explicite.

        Les dépendances contextuelles sont injectées afin de garder l'appel
        réseau indépendant de leur construction. Les validations finales
        garantissent que capacités, retriever, identité et personne décrivent
        le même assemblage applicatif.
        """
        if not isinstance(model, str):
            raise TypeError("Model name must be a string.")

        if not isinstance(base_url, str):
            raise TypeError("Ollama base URL must be a string.")

        if isinstance(timeout, bool) or not isinstance(
            timeout,
            (int, float),
        ):
            raise TypeError("Ollama timeout must be a number.")

        if (
            identity is not None
            and not isinstance(identity, AssistantIdentity)
        ):
            raise TypeError(
                "Ollama identity must be an AssistantIdentity."
            )

        if (
            person_context is not None
            and not isinstance(
                person_context,
                ActivePersonContext,
            )
        ):
            raise TypeError(
                "Ollama person context must be an "
                "ActivePersonContext."
            )

        if (
            capability_context is not None
            and not isinstance(
                capability_context,
                CapabilityContext,
            )
        ):
            raise TypeError(
                "Ollama capability context must be a "
                "CapabilityContext."
            )

        if (
            contextual_memory_retriever is not None
            and not isinstance(
                contextual_memory_retriever,
                ContextualMemoryRetriever,
            )
        ):
            raise TypeError(
                "Ollama contextual memory retriever must be a "
                "ContextualMemoryRetriever."
            )

        normalized_model = model.strip()
        normalized_base_url = base_url.strip().rstrip("/")

        if not normalized_model:
            raise ValueError(
                "Model name cannot be empty."
            )

        if not normalized_base_url:
            raise ValueError(
                "Ollama base URL cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "Ollama timeout must be greater than zero."
            )

        self._model = normalized_model
        self._base_url = normalized_base_url
        self._timeout = float(timeout)
        self._identity = (
            identity
            if identity is not None
            else build_default_identity()
        )
        self._person_context = (
            person_context
            if person_context is not None
            else ActivePersonContext(
                assistant_name=self._identity.name,
                default_person=build_default_person(),
            )
        )
        self._capability_context = (
            capability_context
            if capability_context is not None
            else CapabilityContext(
                available_actions=(),
                automatic_memory_retrieval=(
                    contextual_memory_retriever is not None
                ),
            )
        )
        self._contextual_memory_retriever = (
            contextual_memory_retriever
        )

        if (
            self._capability_context.automatic_memory_retrieval
            != (self._contextual_memory_retriever is not None)
        ):
            raise ValueError(
                "Ollama automatic memory capability must match "
                "its contextual memory retriever."
            )

        if (
            self._person_context.assistant_name.casefold()
            != self._identity.name.casefold()
        ):
            raise ValueError(
                "Ollama person context assistant name must "
                "match the assistant identity."
            )

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Interpréter un tour puis produire le type de réponse requis.

        Une intention d'action s'arrête après l'interprétation : le texte du
        modèle n'est alors pas utilisé pour agir. Une conversation spécialisée
        est validée déterministement avant calcul ; sinon le chemin standard
        peut rappeler des souvenirs puis demander un texte naturel.
        """
        # 1. Le tour courant est isolé de l'historique projeté avant tout appel
        # réseau. Le dernier message demeure la source d'autorité actuelle.
        turn = build_conversation_turn(
            messages
        )

        # 2. Cette première requête produit uniquement des métadonnées
        # structurées. Elle ne génère pas encore la réponse conversationnelle.
        interpretation = self._interpret_turn(
            turn
        )
        intent = interpretation.intent

        if intent.name != "conversation":
            # AssistantCore remettra l'intention au registre. Le client ne
            # contacte aucune action et ne reprend pas un texte libre d'Ollama.
            return ModelResponse(
                content=_ACTION_INTERPRETED_CONTENT,
                model=interpretation.model,
                intent=intent,
            )

        # 3. Les directives du modèle ne sont que des propositions. La preuve
        # textuelle doit être retrouvée et interprétée par l'application.
        try:
            directive = resolve_conversation_directive(
                interpretation.conversation_directive_proposal,
                user_message=messages[-1].content,
            )
        except ConversationDirectiveResolutionError:
            directive = None

        if (
            directive is not None
            and directive.mode
            is ConversationMode.FIXED_TOTAL_ALLOCATION
        ):
            target = directive.allocation_target

            if target is None:
                raise ModelClientError(
                    "Validated fixed-total directive has no target."
                )

            allocation = self._generate_fixed_total_allocation(
                turn,
                target,
            )

            return ModelResponse(
                content=self._render_fixed_total_allocation(
                    allocation
                ),
                model=interpretation.model,
                intent=intent,
            )

        # 4. Le rappel est volontairement postérieur à l'interprétation : un
        # souvenir ne peut donc pas être pris pour une commande utilisateur.
        contextual_memories = (
            self._retrieve_contextual_memories(
                turn.current_user_message.content
            )
        )

        conversation_content, conversation_model = (
            self._generate_conversation(
                turn,
                contextual_memories=contextual_memories,
            )
        )

        return ModelResponse(
            content=conversation_content,
            model=conversation_model,
            intent=intent,
        )

    def _interpret_turn(
        self,
        turn: ConversationTurn,
    ) -> TurnInterpretation:
        """Interpréter la demande courante en JSON strict.

        La méthode exige exactement les champs du schéma, reconstruit les
        objets validés et convertit toute incohérence en ``ModelClientError``.
        """
        # Température nulle et schéma JSON réduisent la variabilité, mais la
        # validation Python reste nécessaire : le modèle demeure non fiable.
        payload = {
            "model": self._model,
            "messages": self._build_turn_messages(
                system_prompt=build_interpretation_prompt(),
                turn=turn,
            ),
            "format": INTERPRETATION_RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DEFAULT_OLLAMA_CONTEXT_LENGTH,
            },
        }

        model, structured_content = (
            self._request_ollama(
                payload
            )
        )

        try:
            interpretation_data = json.loads(
                structured_content
            )
        except JSONDecodeError as error:
            raise ModelClientError(
                "Ollama returned invalid interpreted intent JSON."
            ) from error

        if not isinstance(
            interpretation_data,
            dict,
        ):
            raise ModelClientError(
                "Ollama interpretation must be an object."
            )

        if set(interpretation_data) != {
            "name",
            "parameters",
            "conversation",
        }:
            raise ModelClientError(
                "Ollama interpreted intent contains invalid fields."
            )

        conversation_data = interpretation_data.get(
            "conversation"
        )

        if not isinstance(
            conversation_data,
            dict,
        ):
            raise ModelClientError(
                "Ollama conversation metadata must be an object."
            )

        if set(conversation_data) != {
            "mode",
            "target_text",
        }:
            raise ModelClientError(
                "Ollama conversation metadata contains invalid fields."
            )

        try:
            intent = Intent(
                name=cast(
                    IntentName,
                    interpretation_data.get("name"),
                ),
                parameters=interpretation_data.get(
                    "parameters"
                ),
            )

            mode = ConversationMode(
                conversation_data.get(
                    "mode"
                )
            )

            proposal = ConversationDirectiveProposal(
                mode=mode,
                target_text=conversation_data.get(
                    "target_text"
                ),
            )

            return TurnInterpretation(
                intent=intent,
                conversation_directive_proposal=proposal,
                model=model,
            )
        except (TypeError, ValueError) as error:
            raise ModelClientError(
                "Ollama returned an invalid interpreted intent or conversation metadata."
            ) from error

    def _render_fixed_total_allocation(
        self,
        allocation: FixedTotalAllocation,
    ) -> str:
        """Rendre lisible une répartition validée sans changer ses nombres."""
        if not isinstance(
            allocation,
            FixedTotalAllocation,
        ):
            raise TypeError(
                "Rendered allocation must be a FixedTotalAllocation."
            )

        allocation.require_exact()

        lines = [
            (
                "Répartition validée sur "
                f"{allocation.total} {allocation.unit} :"
            ),
        ]

        lines.extend(
            (
                f"- {part.label} : "
                f"{part.amount} {allocation.unit}"
            )
            for part in allocation.parts
        )

        lines.append(
            (
                "Total : "
                f"{allocation.allocated_total} "
                f"{allocation.unit}"
            )
        )

        return "\n".join(
            lines
        )

    def _generate_fixed_total_allocation(
        self,
        turn: ConversationTurn,
        target: AllocationTarget,
    ) -> FixedTotalAllocation:
        """Demander des parts à Ollama puis vérifier leur somme exactement."""
        if not isinstance(turn, ConversationTurn):
            raise TypeError(
                "Allocation turn must be a ConversationTurn."
            )

        if not isinstance(target, AllocationTarget):
            raise TypeError(
                "Allocation target must be an AllocationTarget."
            )

        payload = {
            "model": self._model,
            "messages": self._build_turn_messages(
                system_prompt=build_allocation_prompt(
                    target
                ),
                turn=turn,
            ),
            "format": ALLOCATION_PROPOSAL_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DEFAULT_OLLAMA_CONTEXT_LENGTH,
            },
        }

        _, structured_content = self._request_ollama(
            payload
        )

        try:
            allocation = parse_allocation_proposal_json(
                structured_content,
                target=target,
            )
            allocation.require_exact()
        except (
            AllocationFormatError,
            AllocationMismatchError,
        ) as error:
            raise ModelClientError(
                "Ollama returned an invalid fixed-total allocation."
            ) from error

        return allocation

    def _generate_conversation(
        self,
        turn: ConversationTurn,
        *,
        contextual_memories: tuple[RetrievedMemory, ...] = (),
    ) -> tuple[str, str]:
        """Générer uniquement le texte naturel d'une conversation.

        Les souvenirs reçus sont à nouveau bornés avant leur rendu dans le
        prompt. Retourne le contenu normalisé et le nom de modèle annoncé.
        """
        payload = {
            "model": self._model,
            "messages": self._build_turn_messages(
                system_prompt=build_conversation_prompt(
                    self._identity,
                    self._person_context,
                    self._capability_context,
                    contextual_memories=(
                        bound_contextual_memories(
                            contextual_memories
                        )
                    ),
                ),
                turn=turn,
            ),
            "stream": False,
            "think": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DEFAULT_OLLAMA_CONTEXT_LENGTH,
            },
        }

        model, content = self._request_ollama(
            payload
        )

        normalized_content = content.strip()

        if not normalized_content:
            raise ModelClientError(
                "Ollama returned empty conversational content."
            )

        return normalized_content, model

    def _retrieve_contextual_memories(
        self,
        user_message: str,
    ) -> tuple[RetrievedMemory, ...]:
        """Rappeler un contexte mémoire en tolérant les erreurs de stockage.

        Une panne mémoire ne doit pas empêcher Fripouille de converser : les
        erreurs contrôlées produisent simplement un contexte vide.
        """
        if self._contextual_memory_retriever is None:
            return ()

        try:
            retrieved_memories = (
                self._contextual_memory_retriever.retrieve(
                    user_message,
                    limit=MAX_INJECTED_CONTEXTUAL_MEMORIES,
                )
            )
        except (DatabaseError, RepositoryError):
            return ()

        return bound_contextual_memories(
            retrieved_memories
        )

    @staticmethod
    def _build_turn_messages(
        *,
        system_prompt: str,
        turn: ConversationTurn,
    ) -> list[dict[str, str]]:
        """Construire les messages Ollama ordonnés pour un tour préparé.

        Le prompt système précède l'historique. Lorsqu'un historique existe,
        un second marqueur système délimite explicitement le tour courant afin
        que les anciens messages ne soient pas confondus avec la demande.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in turn.history
            ],
        ]

        if turn.history:
            messages.append(
                {
                    "role": "system",
                    "content": CURRENT_TURN_CONTEXT_PROMPT,
                }
            )

        messages.append(
            {
                "role": turn.current_user_message.role,
                "content": turn.current_user_message.content,
            }
        )

        return messages

    def _request_ollama(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        """Envoyer une requête locale et valider l'enveloppe de réponse.

        Les erreurs HTTP, réseau, délai, décodage et format sont traduites en
        ``ModelClientError``. La méthode retourne deux chaînes validées : le
        nom du modèle et le contenu du message Ollama.
        """
        request_data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            url=(
                f"{self._base_url}"
                f"{_OLLAMA_CHAT_PATH}"
            ),
            data=request_data,
            headers={
                "Accept": "application/json",
                "Content-Type": (
                    "application/json; charset=utf-8"
                ),
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self._timeout,
            ) as response:
                raw_response = response.read()
        except HTTPError as error:
            raise ModelClientError(
                f"Ollama returned HTTP status {error.code}."
            ) from error
        except URLError as error:
            raise ModelClientError(
                "Could not connect to the local Ollama server."
            ) from error
        except TimeoutError as error:
            raise ModelClientError(
                "The Ollama request exceeded the configured timeout."
            ) from error
        except OSError as error:
            raise ModelClientError(
                "An operating system error occurred while contacting Ollama."
            ) from error

        try:
            response_data = json.loads(
                raw_response.decode("utf-8")
            )
        except UnicodeDecodeError as error:
            raise ModelClientError(
                "Ollama returned a response that is not valid UTF-8."
            ) from error
        except JSONDecodeError as error:
            raise ModelClientError(
                "Ollama returned invalid JSON."
            ) from error

        if not isinstance(
            response_data,
            dict,
        ):
            raise ModelClientError(
                "Ollama returned an invalid response structure."
            )

        message_data = response_data.get(
            "message"
        )

        if not isinstance(
            message_data,
            dict,
        ):
            raise ModelClientError(
                "Ollama response does not contain a valid message."
            )

        content = message_data.get(
            "content"
        )

        if not isinstance(content, str):
            raise ModelClientError(
                "Ollama response message does not contain valid content."
            )

        model = response_data.get(
            "model"
        )

        if not isinstance(model, str):
            raise ModelClientError(
                "Ollama response does not contain a valid model name."
            )

        normalized_model = model.strip()

        if not normalized_model:
            raise ModelClientError(
                "Ollama response model name cannot be empty."
            )

        return normalized_model, content
