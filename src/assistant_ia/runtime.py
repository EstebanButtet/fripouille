"""Frontière entre les interfaces utilisateur et le coeur conversationnel.

Le runtime reçoit un message brut d'une interface, le transmet à
:class:`AssistantCore`, construit un diagnostic séparé, puis transforme la
réponse brute en texte présentable. Il peut enfin envoyer ce texte à un
``ResponsePresenter`` (écran, autre sortie) sans que le coeur connaisse cette
interface.

Il produit donc deux sorties distinctes : une réponse destinée à la personne
et, si demandé, un :class:`TurnDiagnostics` destiné au débogage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistant_ia.core.assistant import AssistantCore
from assistant_ia.expressions import ExpressionController, expression_for_state
from assistant_ia.actions.result import ActionExecutionResult
from assistant_ia.intelligence.intent import Intent
from assistant_ia.interfaces.presentation import (
    build_user_facing_response,
)
from assistant_ia.memory.models import MemoryCandidate
from assistant_ia.memory.promotion import MemoryPromotionProposal


@dataclass(frozen=True, slots=True)
class TurnDiagnostics:
    """Photographie technique immuable d'un tour terminé.

    ``raw_response`` est la réponse avant présentation ; ``intent`` est la
    décision structurée retenue ; les deux champs mémoire décrivent l'analyse
    non persistante et l'éventuelle proposition applicative. ``frozen`` évite
    qu'un reporter de diagnostic modifie rétroactivement ces résultats.
    """

    user_message: str
    raw_response: str
    intent: Intent | None
    action_result: ActionExecutionResult | None
    memory_candidates: tuple[MemoryCandidate, ...]
    memory_promotion_proposal: MemoryPromotionProposal | None


class DiagnosticReporter(Protocol):
    """Contrat minimal d'un consommateur de diagnostics.

    Un ``Protocol`` décrit ici les méthodes attendues sans imposer de classe
    mère. Tout objet possédant une méthode ``report`` compatible peut donc
    être injecté, notamment un double de test ou la console de débogage.
    """

    def report(self, diagnostics: TurnDiagnostics) -> None:
        """Publier les diagnostics sans modifier la réponse visible."""


class ResponsePresenter(Protocol):
    """Contrat d'une sortie capable de présenter une réponse finale."""

    def present(self, response: str) -> None:
        """Afficher ou transmettre une réponse déjà résolue."""


class AssistantRuntime:
    """Coordonner le coeur avec les sorties externes optionnelles.

    Une instance vit aussi longtemps que l'interface qui la possède afin de
    conserver le même :class:`AssistantCore` et donc le même contexte de
    conversation entre les appels à :meth:`process_message`.
    """

    def __init__(
        self,
        assistant: AssistantCore,
        presenter: ResponsePresenter | None = None,
        diagnostic_reporter: DiagnosticReporter | None = None,
    ) -> None:
        """Créer le runtime autour d'un coeur déjà assemblé."""
        if not isinstance(assistant, AssistantCore):
            raise TypeError(
                "Assistant runtime requires an AssistantCore."
            )

        self._assistant = assistant
        self._presenter = presenter
        self._diagnostic_reporter = diagnostic_reporter
        self.expressions = ExpressionController()

    @property
    def assistant(self) -> AssistantCore:
        """Retourner le coeur conversationnel possédé par ce runtime."""
        return self._assistant

    def process_message(
        self,
        user_message: str,
    ) -> str:
        """Traiter un tour, présenter sa réponse et retourner le même texte.

        Effets de bord possibles : mise à jour du contexte par le coeur,
        rapport de diagnostic et appel du presenter. Les informations
        techniques ne sont jamais concaténées à la conversation visible.
        """
        # 1. Le coeur reste l'unique endroit qui interprète et exécute le
        # message. Le runtime ne tente pas de déduire une intention lui-même.
        try:
            raw_response = self._assistant.process_message(user_message)
        finally:
            self.expressions.show(expression_for_state(self._assistant.internal_state.snapshot))
        # 2. La photographie est construite immédiatement après le traitement
        # pour que tous ses champs décrivent exactement le même tour.
        diagnostics = TurnDiagnostics(
            user_message=user_message,
            raw_response=raw_response,
            intent=self._assistant.last_intent,
            action_result=self._assistant.last_action_result,
            memory_candidates=(
                self._assistant.last_memory_candidates
            ),
            memory_promotion_proposal=(
                self._assistant.last_memory_promotion_proposal
            ),
        )

        if self._diagnostic_reporter is not None:
            self._diagnostic_reporter.report(diagnostics)

        # 3. La présentation peut retirer la plomberie structurée ou ajouter
        # une demande de confirmation, sans réécrire la décision du coeur.
        response = build_user_facing_response(
            raw_response,
            intent=diagnostics.intent,
            memory_proposal=(
                diagnostics.memory_promotion_proposal
            ),
            awaiting_memory_confirmation=(
                self._assistant.pending_memory_promotion is not None
            ),
        )

        if self._presenter is not None:
            # Le presenter est un effet de sortie facultatif ; la valeur de
            # retour demeure disponible pour le terminal et pour les tests.
            self._presenter.present(
                response
            )

        return response

    def reset_conversation(self) -> None:
        """Demander au coeur d'oublier l'état conversationnel temporaire."""
        self._assistant.reset_conversation()
        self.expressions.reset()
