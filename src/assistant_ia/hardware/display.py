"""Contrôle de haut niveau de l'écran physique de Fripouille.

``DisplayController`` convertit une intention d'affichage en commande du
protocole ``TEXT`` et vérifie l'accusé de réception. Il ignore le port série et
n'expose pas de primitive écran brute au modèle de langage.
"""

from __future__ import annotations

from assistant_ia.hardware.transport import HardwareTransport

TEXT_COMMAND_PREFIX = "TEXT "
TEXT_RESPONSE = "OK TEXT"
MAX_PROTOCOL_LINE_BYTES = 255


class DisplayProtocolError(RuntimeError):
    """Signaler que l'écran n'a pas répondu selon le protocole attendu."""


class DisplayController:
    """Contrôler le texte visible sans exposer les détails du transport."""

    def __init__(
        self,
        transport: HardwareTransport,
    ) -> None:
        """Créer le contrôleur au-dessus d'un transport injecté."""
        self._transport = transport

    def set_text(self, text: str) -> None:
        """Remplacer le texte de l'écran après validation du protocole.

        Les retours à la ligne et ``@`` sont réservés au cadrage. La taille est
        contrôlée en octets UTF-8, car le protocole borne des octets et non le
        nombre de caractères Python.
        """
        if not isinstance(text, str):
            raise TypeError(
                "Display text must be a string."
            )

        if "\r" in text or "\n" in text:
            raise ValueError(
                "Display text cannot contain line breaks."
            )

        if "@" in text:
            raise ValueError(
                "Display text cannot contain the protocol "
                "synchronization marker."
            )

        command = f"{TEXT_COMMAND_PREFIX}{text}"

        if len(command.encode("utf-8")) > MAX_PROTOCOL_LINE_BYTES:
            raise ValueError(
                "Display text exceeds the protocol line limit."
            )

        response = self._transport.request(command)

        if response != TEXT_RESPONSE:
            raise DisplayProtocolError(
                "The display did not acknowledge the text command."
            )
