"""Tests for terminal-compatible module launch selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from assistant_ia import __main__ as entry_point


class MainEntryPointTests(unittest.TestCase):
    def test_default_launch_keeps_historical_terminal(self) -> None:
        with patch.object(entry_point, "run_terminal") as run_terminal:
            entry_point.main([])

        run_terminal.assert_called_once_with()

    def test_debug_terminal_is_explicit(self) -> None:
        with patch.object(entry_point, "run_terminal") as run_terminal:
            entry_point.main(["--debug"])

        run_terminal.assert_called_once_with(debug=True)

    def test_gui_launch_forwards_debug_flag(self) -> None:
        with patch(
            "assistant_ia.interfaces.gui.run_gui"
        ) as run_gui:
            entry_point.main(["--gui", "--debug"])

        run_gui.assert_called_once_with(debug=True)


if __name__ == "__main__":
    unittest.main()
