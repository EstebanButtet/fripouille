"""Détection déterministe d'une présentation explicite du locuteur.

Seules quelques phrases complètes reconnues produisent un ``PersonProfile``.
Une simple mention de nom, une déduction du LLM ou une phrase plus large ne
change pas la personne active. Le résultat reste temporaire à la session.
"""

from __future__ import annotations

import re
import unicodedata

from assistant_ia.people.models import PersonProfile


_NAME_PATTERN = (
    r"([A-Za-z\u00C0-\u024F]"
    r"[A-Za-z\u00C0-\u024F'\u2019-]{0,63})"
)

_PRESENTATION_PREFIX = (
    r"(?:(?:salut|bonjour|bonsoir|hello|hey|coucou|au\s+fait)"
    r"\s*[,!:\-]?\s*)?"
)

_PRESENTATION_PATTERNS = (
    re.compile(
        rf"{_PRESENTATION_PREFIX}"
        rf"moi\s*,?\s*c['\u2019]est\s+{_NAME_PATTERN}"
        r"\s*[.!]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_PRESENTATION_PREFIX}"
        rf"je\s+m['\u2019]appelle\s+{_NAME_PATTERN}"
        r"\s*[.!]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_PRESENTATION_PREFIX}"
        rf"mon\s+pr(?:e|\u00E9)nom\s+est\s+{_NAME_PATTERN}"
        r"\s*[.!]?\s*",
        re.IGNORECASE,
    ),
)


def detect_presented_person(
    message: str,
) -> PersonProfile | None:
    """Retourner une personne uniquement pour une présentation explicite.

    Le message est normalisé en Unicode NFC puis comparé entièrement aux motifs
    autorisés. Retourne ``None`` lorsque la preuve est absente ou ambiguë.
    """
    if not isinstance(message, str):
        raise TypeError(
            "Presentation message must be a string."
        )

    normalized_message = unicodedata.normalize(
        "NFC",
        message,
    ).strip()

    # ``fullmatch`` empêche une sous-phrase telle qu'une citation de déclencher
    # involontairement un changement de locuteur.
    for pattern in _PRESENTATION_PATTERNS:
        match = pattern.fullmatch(
            normalized_message
        )

        if match is not None:
            return PersonProfile(
                name=match.group(1),
            )

    return None
