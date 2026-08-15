"""Windows assembly for the physical assistant display."""

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
    """Create one serial connection for a Windows COM port."""

    def __call__(
        self,
        port_name: str,
        *,
        startup_delay: float,
    ) -> SerialConnection:
        """Open and return one serial connection."""


class WindowsDisplayPresenter:
    """Own the complete Windows-to-display presentation chain."""

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
        """Open and assemble the physical display connection."""
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
        """Present one final assistant response on the physical display."""
        self._require_open()

        self._presenter.present(
            response
        )

    def close(self) -> None:
        """Close the physical display connection once."""
        if self._closed:
            return

        self._closed = True
        self._transport.close()

    def _require_open(self) -> None:
        """Reject display operations after shutdown."""
        if self._closed:
            raise HardwareTransportError(
                "Windows display presenter is closed."
            )
