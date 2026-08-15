"""Framed serial transport for assistant hardware."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from assistant_ia.hardware.transport import HardwareTransportError

FRAME_MARKER = b"@"
FRAME_TERMINATOR = b"\n"
MAX_COMMAND_BYTES = 255


class SerialConnection(Protocol):
    """Provide raw byte access to one serial connection."""

    def read(self, max_bytes: int) -> bytes:
        """Read up to the requested number of available bytes."""

    def write(self, data: bytes) -> None:
        """Write raw bytes to the serial connection."""

    def close(self) -> None:
        """Close the serial connection."""


class FramedSerialTransport:
    """Exchange synchronized line commands over a serial connection."""

    def __init__(
        self,
        connection: SerialConnection,
        *,
        response_timeout: float = 1.0,
        poll_interval: float = 0.01,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the transport around an already-open connection."""
        if response_timeout <= 0:
            raise ValueError(
                "Serial response timeout must be positive."
            )

        if poll_interval <= 0:
            raise ValueError(
                "Serial poll interval must be positive."
            )

        if not callable(monotonic):
            raise TypeError(
                "Serial monotonic clock must be callable."
            )

        if not callable(sleep):
            raise TypeError(
                "Serial sleep function must be callable."
            )

        self._connection = connection
        self._response_timeout = response_timeout
        self._poll_interval = poll_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._receive_buffer = bytearray()

    def request(self, command: str) -> str:
        """Send one framed command and return one framed response."""
        encoded_command = self._encode_command(
            command
        )

        frame = (
            FRAME_MARKER
            + encoded_command
            + FRAME_TERMINATOR
        )

        try:
            self._connection.write(frame)
        except OSError as error:
            raise HardwareTransportError(
                "Hardware serial command could not be written."
            ) from error

        deadline = (
            self._monotonic()
            + self._response_timeout
        )

        while self._monotonic() < deadline:
            response = self._extract_response()

            if response is not None:
                return response

            try:
                received = self._connection.read(256)
            except OSError as error:
                raise HardwareTransportError(
                    "Hardware serial response could not be read."
                ) from error

            if received:
                self._receive_buffer.extend(
                    received
                )
                continue

            self._sleep(
                self._poll_interval
            )

        response = self._extract_response()

        if response is not None:
            return response

        raise HardwareTransportError(
            "Hardware did not return a framed response in time."
        )

    def close(self) -> None:
        """Close the underlying serial connection."""
        try:
            self._connection.close()
        except OSError as error:
            raise HardwareTransportError(
                "Hardware serial connection could not be closed."
            ) from error

    @staticmethod
    def _encode_command(
        command: str,
    ) -> bytes:
        """Validate and encode one protocol command."""
        if not isinstance(command, str):
            raise TypeError(
                "Hardware command must be a string."
            )

        if not command:
            raise ValueError(
                "Hardware command cannot be empty."
            )

        if (
            "\r" in command
            or "\n" in command
            or "@" in command
        ):
            raise ValueError(
                "Hardware command contains protocol control "
                "characters."
            )

        encoded_command = command.encode(
            "utf-8"
        )

        if len(encoded_command) > MAX_COMMAND_BYTES:
            raise ValueError(
                "Hardware command exceeds the protocol line limit."
            )

        return encoded_command

    def _extract_response(
        self,
    ) -> str | None:
        """Extract the next complete framed response from buffered bytes."""
        while True:
            marker_index = self._receive_buffer.find(
                FRAME_MARKER
            )

            if marker_index < 0:
                if len(self._receive_buffer) > 4096:
                    self._receive_buffer.clear()

                return None

            if marker_index > 0:
                del self._receive_buffer[
                    :marker_index
                ]

            terminator_index = self._receive_buffer.find(
                FRAME_TERMINATOR,
                1,
            )

            if terminator_index < 0:
                return None

            payload = bytes(
                self._receive_buffer[
                    1:terminator_index
                ]
            ).rstrip(b"\r")

            del self._receive_buffer[
                :terminator_index + 1
            ]

            if not payload:
                continue

            try:
                return payload.decode(
                    "utf-8"
                )
            except UnicodeDecodeError as error:
                raise HardwareTransportError(
                    "Hardware returned an invalid UTF-8 response."
                ) from error
