"""Assemblage Windows de toute la chaîne de présentation physique.

Ce module ouvre le port COM puis relie, dans l'ordre, connexion brute,
transport cadré, contrôleur d'écran et presenter de réponse. Il constitue une
racine de composition matérielle et ne donne jamais la connexion brute au LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from assistant_ia.hardware.display import DisplayController
from assistant_ia.hardware.presentation import DisplayResponsePresenter
from assistant_ia.hardware.serial_transport import (
    FramedSerialTransport,
    SerialConnection,
)
from assistant_ia.hardware.transport import HardwareTransportError
from assistant_ia.hardware.windows_serial import WindowsSerialConnection


class SerialConnectionFactory(Protocol):
    """Contrat d'une fabrique de connexion pour un port COM Windows."""

    def __call__(
        self,
        port_name: str,
        *,
        startup_delay: float,
    ) -> SerialConnection:
        """Ouvrir et retourner une connexion série."""


class WindowsDisplayPresenter:
    """Posséder toute la chaîne Windows jusqu'à l'écran physique.

    L'instance garde le transport ouvert pendant son cycle de vie. ``close``
    est idempotente et toute présentation ultérieure est refusée.
    """

    def __init__(
        self,
        port_name: str,
        *,
        startup_delay: float = 2.5,
        response_timeout: float = 2.0,
        connection_factory: SerialConnectionFactory = (
            WindowsSerialConnection
        ),
    ) -> None:
        """Ouvrir le port et assembler les couches de présentation.

        Si la construction du transport échoue après l'ouverture, la connexion
        est immédiatement fermée afin d'éviter de conserver le port COM.
        """
        if not callable(connection_factory):
            raise TypeError(
                "Serial connection factory must be callable."
            )

        if (
            isinstance(response_timeout, bool)
            or not isinstance(
                response_timeout,
                (int, float),
            )
        ):
            raise TypeError(
                "Display response timeout must be a number."
            )

        if response_timeout <= 0:
            raise ValueError(
                "Display response timeout must be positive."
            )

        connection = connection_factory(
            port_name,
            startup_delay=startup_delay,
        )

        try:
            transport = FramedSerialTransport(
                connection,
                response_timeout=float(response_timeout),
            )
        except Exception:
            connection.close()
            raise

        display = DisplayController(
            transport
        )

        self._transport = transport
        self._presenter = DisplayResponsePresenter(
            display
        )
        self._closed = False

    def present(
        self,
        response: str,
    ) -> None:
        """Présenter une réponse finale via la chaîne contrôlée."""
        self._require_open()

        self._presenter.present(
            response
        )

    def close(self) -> None:
        """Fermer une seule fois la connexion de l'écran physique."""
        if self._closed:
            return

        self._closed = True
        self._transport.close()

    def _require_open(self) -> None:
        """Refuser toute opération d'affichage après la fermeture."""
        if self._closed:
            raise HardwareTransportError(
                "Windows display presenter is closed."
            )
