"""Tests for physical display response presentation."""

from __future__ import annotations

import unittest

from assistant_ia.hardware.display import DisplayController
from assistant_ia.hardware.presentation import (
    DisplayResponsePresenter,
)


class FakeHardwareTransport:
    """Record display protocol commands."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def request(
        self,
        command: str,
    ) -> str:
        self.commands.append(
            command
        )
        return "OK TEXT"


class DisplayResponsePresenterTests(unittest.TestCase):
    """Validate safe conversion of final responses for the display."""

    def _build(
        self,
    ) -> tuple[
        DisplayResponsePresenter,
        FakeHardwareTransport,
    ]:
        transport = FakeHardwareTransport()
        display = DisplayController(
            transport
        )

        return (
            DisplayResponsePresenter(
                display
            ),
            transport,
        )

    def test_presents_short_response_unchanged(self) -> None:
        presenter, transport = self._build()

        presenter.present(
            "Salut Fripouille."
        )

        self.assertEqual(
            transport.commands,
            [
                "TEXT Salut Fripouille.",
            ],
        )

    def test_collapses_multiline_response(self) -> None:
        presenter, transport = self._build()

        presenter.present(
            "Premi?re ligne.\n\nDeuxi?me ligne."
        )

        self.assertEqual(
            transport.commands,
            [
                "TEXT Premi?re ligne. Deuxi?me ligne.",
            ],
        )

    def test_replaces_protocol_synchronization_marker(self) -> None:
        presenter, transport = self._build()

        presenter.present(
            "Contact: test@example.com"
        )

        self.assertEqual(
            transport.commands,
            [
                "TEXT Contact: test(at)example.com",
            ],
        )

    def test_truncates_long_response_to_protocol_limit(self) -> None:
        presenter, transport = self._build()

        presenter.present(
            "\u00e9" * 200
        )

        command = transport.commands[0]

        self.assertLessEqual(
            len(command.encode("utf-8")),
            255,
        )
        self.assertTrue(
            command.endswith("...")
        )

    def test_empty_response_can_clear_display_text(self) -> None:
        presenter, transport = self._build()

        presenter.present("")

        self.assertEqual(
            transport.commands,
            [
                "TEXT ",
            ],
        )

    def test_rejects_non_string_response(self) -> None:
        presenter, _ = self._build()

        with self.assertRaisesRegex(
            TypeError,
            "must be a string",
        ):
            presenter.present(
                42  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
