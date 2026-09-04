"""Politique déterministe d'attribution d'un candidat au locuteur actif.

Cette couche ne résout aucun nom et ne consulte ni LLM ni SQLite. Elle décide
seulement si la preuve source parle explicitement à la première personne ; le
``person_id`` éventuel reste fourni séparément par le contexte applicatif.
"""

from __future__ import annotations

import re
import unicodedata

from assistant_ia.memory.models import MemoryCandidate

_FIRST_PERSON_PATTERN = re.compile(
    r"(?:\bje\b|\bj['’]|\bmoi\b|\bmon\b|\bma\b|\bmes\b|"
    r"\bnous\b|\bnotre\b|\bnos\b)",
    flags=re.IGNORECASE,
)

_THIRD_PARTY_OR_GENERAL_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:il|elle|ils|elles)\b",
        r"\b(?:mon|ma|mes)\s+(?:"
        r"ami|amie|frere|soeur|pere|mere|collegue|voisin|voisine"
        r")\b",
        r"\bje\s+(?:sais|crois|pense|dis)\s+qu(?:e\b|['’])",
        r"\bje\s+(?:te\s+)?(?:rappelle|confirme|signale)\s+que\b",
        r"\bje\s+(?:veux|souhaite)\s+que\s+tu\b",
    )
)


def candidate_targets_active_person(candidate: MemoryCandidate) -> bool:
    """Indiquer si la preuve autorise le lien ``subject`` vers le locuteur.

    La règle accepte seulement une marque explicite de première personne et
    refuse les formes courantes d'attribution à un tiers ou d'encapsulation
    d'un fait général. Une réponse ``False`` conserve la mémoire sans sujet.
    """
    if not isinstance(candidate, MemoryCandidate):
        raise TypeError("Memory subject policy requires a MemoryCandidate.")

    normalized_source = _normalize_text(candidate.source_text)
    normalized_content = _normalize_text(candidate.content)
    if (
        not _FIRST_PERSON_PATTERN.search(normalized_source)
        or not _FIRST_PERSON_PATTERN.search(normalized_content)
    ):
        return False
    return not any(
        pattern.search(normalized_source)
        or pattern.search(normalized_content)
        for pattern in _THIRD_PARTY_OR_GENERAL_PATTERNS
    )


def _normalize_text(value: str) -> str:
    """Normaliser casse et accents sans rapprocher les identités nommées."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
