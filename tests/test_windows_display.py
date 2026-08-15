"""Tests for complete Windows physical display assembly."""

from __future__ import annotations

import unittest

from assistant_ia.hardware.transport import HardwareTransportError
from assistant_ia.hardware.windows_display import WindowsDisplayPresenter


class FakeSerialConnection:
    """Provide deterministic framed serial responses."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.responses: list[bytes] = []
        self.close_count = 0

    def read(
        self,
        max_bytes: int,
    ) -> bytes:
        if not self.responses:
            return b""

        return self.responses.pop(0)

    def write(
        self,
        data: bytes,
    ) -> None:
        self.writes.append(
            data
        )
        self.responses.append(
            b"@OK TEXT\n"
        )

    def close(self) -> None:
        self.close_count += 1


class RecordingConnectionFactory:
    """Record serial connection construction."""

    def __init__(self) -> None:
        self.connection = FakeSerialConnection()
        self.calls: list[tuple[str, float]] = []

    def __call__(
        self,
        port_name: str,
        *,
        startup_delay: float,
    ) -> FakeSerialConnection:
        self.calls.append(
            (
                port_name,
                startup_delay,
            )
        )
        return self.connection


class WindowsDisplayPresenterTests(unittest.TestCase):
    """Validate complete Windows display presentation."""

    def test_builds_connection_with_requested_port(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            startup_delay=3.0,
            connection_factory=factory,
        )

        self.assertEqual(
            factory.calls,
            [
                (
                    "COM4",
                    3.0,
                ),
            ],
        )

        presenter.close()

    def test_presents_response_through_complete_chain(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            connection_factory=factory,
        )

        presenter.present(
            "Bonjour Fripouille."
        )

        self.assertEqual(
            factory.connection.writes,
            [
                b"@TEXT Bonjour Fripouille.\n",
            ],
        )

        presenter.close()

    def test_applies_display_response_conversion(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            connection_factory=factory,
        )

        presenter.present(
            "Ligne une.\nLigne deux @ test."
        )

        self.assertEqual(
            factory.connection.writes,
            [
                b"@TEXT Ligne une. Ligne deux (at) test.\n",
            ],
        )

        presenter.close()

    def test_close_closes_serial_connection(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            connection_factory=factory,
        )

        presenter.close()

        self.assertEqual(
            factory.connection.close_count,
            1,
        )

    def test_close_is_idempotent(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            connection_factory=factory,
        )

        presenter.close()
        presenter.close()

        self.assertEqual(
            factory.connection.close_count,
            1,
        )

    def test_rejects_presentation_after_close(self) -> None:
        factory = RecordingConnectionFactory()

        presenter = WindowsDisplayPresenter(
            "COM4",
            connection_factory=factory,
        )

        presenter.close()

        with self.assertRaisesRegex(
            HardwareTransportError,
            "presenter is closed",
        ):
            presenter.present(
                "Bonjour."
            )

    def test_rejects_invalid_response_timeout_before_opening(self) -> None:
        factory = RecordingConnectionFactory()

        with self.assertRaisesRegex(
            ValueError,
            "must be positive",
        ):
            WindowsDisplayPresenter(
                "COM4",
                response_timeout=0,
                connection_factory=factory,
            )

        self.assertEqual(
            factory.calls,
            [],
        )


if __name__ == "__main__":
    unittest.main()
