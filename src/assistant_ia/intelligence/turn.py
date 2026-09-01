"""Préparation d'un tour à partir de messages conversationnels ordonnés.

Le dernier message utilisateur est séparé de l'historique, puis l'historique
est borné sans couper le contenu d'un message. Le client Ollama reçoit ainsi
un contexte récent et maîtrisé, distinct de la demande actuelle.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.core.context import ConversationMessage

# Limites empiriques initiales, centralisées afin que leur impact puisse être
# mesuré avant un éventuel ajustement futur.
MAX_PROJECTED_HISTORY_MESSAGES = 8
MAX_PROJECTED_HISTORY_CHARACTERS = 3000


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Séparer l'historique projeté du message utilisateur courant.

    Cette dataclass immuable évite de confondre un ancien message avec la
    demande qui doit être interprétée pendant le tour présent.
    """

    history: tuple[ConversationMessage, ...]
    current_user_message: ConversationMessage


def build_conversation_turn(
    messages: tuple[ConversationMessage, ...],
) -> ConversationTurn:
    """Construire un tour depuis une séquence terminée par l'utilisateur.

    Lève ``TypeError`` si l'entrée n'est pas un tuple et ``ValueError`` si
    elle est vide ou ne se termine pas par un message utilisateur.
    """
    if not isinstance(messages, tuple):
        raise TypeError(
            "Conversation messages must be provided as a tuple."
        )

    if not messages:
        raise ValueError(
            "At least one conversation message is required."
        )

    current_user_message = messages[-1]

    if current_user_message.role != "user":
        raise ValueError(
            "The latest conversation message must be from the user."
        )

    # Le dernier message reste entier et hors de la projection : les limites
    # ne s'appliquent qu'au contexte historique qui le précède.
    return ConversationTurn(
        history=project_conversation_history(
            messages[:-1]
        ),
        current_user_message=current_user_message,
    )


def project_conversation_history(
    history: tuple[ConversationMessage, ...],
) -> tuple[ConversationMessage, ...]:
    """Retourner le suffixe récent respectant les deux limites globales.

    Les échanges utilisateur/assistant sont sélectionnés comme groupes pour
    ne pas conserver une réponse sans la question qui lui donnait son sens.
    Aucun message individuel n'est tronqué.
    """
    if not isinstance(history, tuple):
        raise TypeError(
            "Conversation history must be provided as a tuple."
        )

    groups = _group_conversation_history(history)
    selected_groups: list[tuple[ConversationMessage, ...]] = []
    selected_message_count = 0
    selected_character_count = 0

    # On part du groupe le plus récent et on s'arrête au premier groupe qui
    # dépasserait le budget : le résultat reste donc un suffixe chronologique.
    for group in reversed(groups):
        group_message_count = len(group)
        group_character_count = sum(
            len(message.content)
            for message in group
        )

        if (
            selected_message_count + group_message_count
            > MAX_PROJECTED_HISTORY_MESSAGES
            or selected_character_count + group_character_count
            > MAX_PROJECTED_HISTORY_CHARACTERS
        ):
            break

        selected_groups.append(group)
        selected_message_count += group_message_count
        selected_character_count += group_character_count

    return tuple(
        message
        for group in reversed(selected_groups)
        for message in group
    )


def _group_conversation_history(
    history: tuple[ConversationMessage, ...],
) -> tuple[tuple[ConversationMessage, ...], ...]:
    """Regrouper les paires normales tout en tolérant un historique atypique.

    Un message isolé devient son propre groupe. Cette tolérance simplifie les
    tests et évite que la projection ne répare silencieusement l'historique.
    """
    groups: list[tuple[ConversationMessage, ...]] = []
    index = 0

    while index < len(history):
        message = history[index]

        if (
            message.role == "user"
            and index + 1 < len(history)
            and history[index + 1].role == "assistant"
        ):
            groups.append(
                (
                    message,
                    history[index + 1],
                )
            )
            index += 2
            continue

        groups.append((message,))
        index += 1

    return tuple(groups)
