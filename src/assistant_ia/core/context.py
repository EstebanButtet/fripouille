"""Historique conversationnel temporaire, conservé uniquement en mémoire.

Ce module reçoit les textes utilisateur et assistant validés par le coeur et
les conserve dans leur ordre d'arrivée. Il produit des tuples immuables pour
le client de modèle et :mod:`assistant_ia.intelligence.turn`.

Il ne s'agit pas de la mémoire persistante de Fripouille : fermer le processus
fait disparaître cet historique, tandis que les souvenirs durables passent par
les repositories du paquet :mod:`assistant_ia.memory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """Message horodaté par son ordre, avec un rôle et un contenu.

    ``role`` indique qui parle et ``content`` contient le texte normalisé. La
    dataclass est ``frozen`` : après création, un ancien tour ne peut plus être
    modifié par mégarde lorsqu'il est transmis au modèle.
    """

    role: MessageRole
    content: str


class ConversationContext:
    """Posséder et ordonner les messages de la conversation courante.

    La liste interne reste mutable pour ajouter des tours, mais la propriété
    publique :attr:`messages` en retourne une copie sous forme de tuple. Cette
    encapsulation empêche un appelant de réordonner directement l'historique.
    """

    def __init__(self) -> None:
        """Créer un contexte sans aucun message."""
        self._messages: list[ConversationMessage] = []

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        """Retourner un instantané immuable des messages ordonnés."""
        return tuple(self._messages)

    @property
    def message_count(self) -> int:
        """Retourner le nombre de messages actuellement conservés."""
        return len(self._messages)

    def add_user_message(self, content: str) -> ConversationMessage:
        """Valider puis ajouter un message utilisateur."""
        return self._add_message(role="user", content=content)

    def add_assistant_message(self, content: str) -> ConversationMessage:
        """Valider puis ajouter un message assistant."""
        return self._add_message(role="assistant", content=content)

    def clear(self) -> None:
        """Supprimer tout l'historique conversationnel temporaire."""
        self._messages.clear()

    def _add_message(
        self,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage:
        """Normaliser, valider puis stocker un message de l'un des deux rôles.

        Lève ``TypeError`` pour un contenu non textuel et ``ValueError`` pour
        un texte vide après retrait des espaces extérieurs.
        """
        if not isinstance(content, str):
            raise TypeError("Message content must be a string.")

        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError("Conversation messages cannot be empty.")

        message = ConversationMessage(
            role=role,
            content=normalized_content,
        )
        self._messages.append(message)

        return message
