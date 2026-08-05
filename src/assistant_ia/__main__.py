"""Entry point for running the assistant as a Python module."""

from __future__ import annotations

from assistant_ia.interfaces.terminal import run_terminal


def main() -> None:
    """Start the personal AI assistant."""
    run_terminal()


if __name__ == "__main__":
    main()