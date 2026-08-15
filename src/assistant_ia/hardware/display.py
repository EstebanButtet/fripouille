"""High-level control of the assistant physical display."""

from __future__ import annotations

from assistant_ia.hardware.transport import HardwareTransport

TEXT_COMMAND_PREFIX = "TEXT "
TEXT_RESPONSE = "OK TEXT"
MAX_PROTOCOL_LINE_BYTES = 255


class DisplayProtocolError(RuntimeError):
    """Raised when the display returns an unexpected response."""


class DisplayController:
    """Control visible content without exposing transport details."""

    def __init__(
        self,
        transport: HardwareTransport,
    ) -> None:
        """Create a display controller using the provided transport."""
        self._transport = transport

    def set_text(self, text: str) -> None:
        """Replace the text shown on the physical display."""
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
