"""Transport contracts for assistant hardware."""

from __future__ import annotations

from typing import Protocol


class HardwareTransportError(RuntimeError):
    """Raised when communication with assistant hardware fails."""


class HardwareTransport(Protocol):
    """Exchange structured commands with assistant hardware."""

    def request(self, command: str) -> str:
        """Send one command and return its structured response."""
