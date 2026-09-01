"""Présentation d'une réponse finale sur l'écran physique.

Ce presenter implémente le même contrat que les autres sorties du runtime. Il
aplatit les retours à la ligne, neutralise le marqueur de synchronisation et
tronque proprement en UTF-8 avant d'appeler ``DisplayController``.
"""

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
    """Présenter sur l'écran une prévisualisation de la réponse finale.

    Il reçoit seulement un texte déjà résolu ; il ne demande rien au LLM et ne
    choisit aucune action physique.
    """

    def __init__(
        self,
        display: DisplayController,
    ) -> None:
        """Créer le presenter avec un contrôleur d'écran de haut niveau."""
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
        """Présenter une version de la réponse compatible avec le protocole."""
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
    """Convertir un texte quelconque vers le protocole d'affichage actuel."""
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
    """Tronquer en octets sans couper un caractère UTF-8.

    Python itère par caractères Unicode ; la boucle additionne leur taille
    encodée et réserve d'abord la place du suffixe ``...``.
    """
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
