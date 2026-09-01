"""Contrat abstrait des transports vers le hardware de Fripouille.

Le reste de l'application échange des commandes textuelles structurées via ce
``Protocol`` sans connaître Win32, un port COM ou le cadrage série concret.
Cette abstraction n'accorde aucune autorité au LLM : seuls des contrôleurs
applicatifs validés doivent produire les commandes transmises ici.
"""

from __future__ import annotations

from typing import Protocol


class HardwareTransportError(RuntimeError):
    """Signaler un échec contrôlé de communication matérielle."""


class HardwareTransport(Protocol):
    """Contrat d'échange de commandes structurées avec le matériel."""

    def request(self, command: str) -> str:
        """Envoyer une commande validée et retourner sa réponse structurée."""
