"""Entry point for running the assistant as a Python module."""

from __future__ import annotations

import argparse

from assistant_ia.interfaces.terminal import run_terminal


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lancer Fripouille.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="utiliser l'interface graphique provisoire",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="afficher les diagnostics techniques dans la console",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Start the personal AI assistant."""
    arguments = _build_argument_parser().parse_args(argv)

    if arguments.gui:
        from assistant_ia.interfaces.gui import run_gui

        run_gui(debug=arguments.debug)
        return

    if arguments.debug:
        run_terminal(debug=True)
    else:
        run_terminal()


if __name__ == "__main__":
    main()
