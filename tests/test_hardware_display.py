"""Tests for high-level physical display control."""

from __future__ import annotations

import unittest

from assistant_ia.hardware.display import (
    DisplayController,
    DisplayProtocolError,
)


class FakeHardwareTransport:
    """Record commands and return a configurable response."""

    def __init__(
        self,
        response: str = "OK TEXT",
    ) -> None:
        self.response = response
        self.commands: list[str] = []

    def request(self, command: str) -> str:
        self.commands.append(command)
        return self.response


class DisplayControllerTests(unittest.TestCase):
    """Validate display commands independently from real hardware."""

    def test_set_text_sends_expected_command(self) -> None:
        transport = FakeHardwareTransport()
        display = DisplayController(transport)

        display.set_text("Bonjour depuis Fripouille")

        self.assertEqual(
            transport.commands,
            ["TEXT Bonjour depuis Fripouille"],
        )

    def test_set_text_preserves_utf8_content(self) -> None:
        transport = FakeHardwareTransport()
        display = DisplayController(transport)

        display.set_text("R?ponse pr?te.")

        self.assertEqual(
            transport.commands,
            ["TEXT R?ponse pr?te."],
        )

    def test_set_text_allows_maximum_protocol_size(self) -> None:
        transport = FakeHardwareTransport()
        display = DisplayController(transport)

        display.set_text("x" * 250)

        self.assertEqual(
            len(transport.commands),
            1,
        )

    def test_set_text_rejects_oversized_protocol_line(self) -> None:
        display = DisplayController(
            FakeHardwareTransport()
        )

        with self.assertRaises(ValueError):
            display.set_text("x" * 251)

    def test_set_text_rejects_line_breaks(self) -> None:
        display = DisplayController(
            FakeHardwareTransport()
        )

        with self.assertRaises(ValueError):
            display.set_text("ligne 1\nligne 2")

    def test_set_text_rejects_synchronization_marker(self) -> None:
        display = DisplayController(
            FakeHardwareTransport()
        )

        with self.assertRaises(ValueError):
            display.set_text("adresse@test")

    def test_set_text_rejects_non_string_content(self) -> None:
        display = DisplayController(
            FakeHardwareTransport()
        )

        with self.assertRaises(TypeError):
            display.set_text(123)  # type: ignore[arg-type]

    def test_set_text_rejects_unexpected_response(self) -> None:
        display = DisplayController(
            FakeHardwareTransport("ERR DISPLAY")
        )

        with self.assertRaises(DisplayProtocolError):
            display.set_text("Bonjour")


if __name__ == "__main__":
    unittest.main()
