"""Presentation of assistant responses on the physical display."""

from __future__ import annotations

from assistant_ia.hardware.display import (
    MAX_PROTOCOL_LINE_BYTES,
    TEXT_COMMAND_PREFIX,
    DisplayController,
)

_DISPLAY_TEXT_MAX_BYTES = (
    MAX_PROTOCOL_LINE_BYTES
    - len(TEXT_COMMAND_PREFIX.encode("utf-8"))
)
_TRUNCATION_SUFFIX = "..."


class DisplayResponsePresenter:
    """Present final assistant responses on the physical display."""

    def __init__(
        self,
        display: DisplayController,
    ) -> None:
        """Create a presenter using one display controller."""
        if not isinstance(display, DisplayController):
            raise TypeError(
                "Display response presenter requires "
                "a DisplayController."
            )

        self._display = display

    def present(
        self,
        response: str,
    ) -> None:
        """Present a protocol-safe preview of one assistant response."""
        if not isinstance(response, str):
            raise TypeError(
                "Presented assistant response must be a string."
            )

        display_text = _prepare_display_text(
            response
        )

        self._display.set_text(
            display_text
        )


def _prepare_display_text(
    response: str,
) -> str:
    """Convert arbitrary response text to the current display protocol."""
    normalized = " ".join(
        response.split()
    )

    normalized = normalized.replace(
        "@",
        "(at)",
    )

    return _truncate_utf8(
        normalized,
        _DISPLAY_TEXT_MAX_BYTES,
    )


def _truncate_utf8(
    text: str,
    max_bytes: int,
) -> str:
    """Truncate text without splitting a UTF-8 character."""
    encoded = text.encode(
        "utf-8"
    )

    if len(encoded) <= max_bytes:
        return text

    suffix_bytes = _TRUNCATION_SUFFIX.encode(
        "utf-8"
    )
    content_limit = (
        max_bytes
        - len(suffix_bytes)
    )

    characters: list[str] = []
    used_bytes = 0

    for character in text:
        character_size = len(
            character.encode("utf-8")
        )

        if used_bytes + character_size > content_limit:
            break

        characters.append(
            character
        )
        used_bytes += character_size

    return (
        "".join(characters).rstrip()
        + _TRUNCATION_SUFFIX
    )
