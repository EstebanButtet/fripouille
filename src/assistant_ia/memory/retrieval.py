"""Deterministic lexical retrieval for persistent memories."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from assistant_ia.memory.memory_repository import (
    MAX_MEMORY_LIST_LIMIT,
    MemoryRepository,
)
from assistant_ia.memory.models import Memory

DEFAULT_CONTEXTUAL_MEMORY_LIMIT = 3
MAX_CONTEXTUAL_MEMORY_LIMIT = 20
DEFAULT_RETRIEVAL_CANDIDATE_LIMIT = 500

MIN_TOKEN_LENGTH = 2

IGNORED_TERMS: frozenset[str] = frozenset(
    {
        "a",
        "ai",
        "and",
        "are",
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "combien",
        "comment",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "en",
        "est",
        "et",
        "for",
        "from",
        "how",
        "il",
        "in",
        "is",
        "je",
        "la",
        "le",
        "les",
        "ma",
        "mais",
        "me",
        "mes",
        "mon",
        "ne",
        "nous",
        "of",
        "on",
        "ou",
        "par",
        "pas",
        "pour",
        "pourquoi",
        "qu",
        "quand",
        "que",
        "quel",
        "quelle",
        "quelles",
        "quels",
        "qui",
        "quoi",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "the",
        "to",
        "tu",
        "un",
        "une",
        "vous",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "you",
        "your",
    }
)

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """Expose one memory with its inspectable lexical relevance."""

    memory: Memory
    score: float
    matched_terms: tuple[str, ...]


class ContextualMemoryRetriever:
    """Select relevant persistent memories without changing them."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        candidate_limit: int = DEFAULT_RETRIEVAL_CANDIDATE_LIMIT,
    ) -> None:
        """Create a retriever over one bounded repository view."""
        if not isinstance(repository, MemoryRepository):
            raise TypeError(
                "Contextual memory repository must be a MemoryRepository."
            )

        if (
            isinstance(candidate_limit, bool)
            or not isinstance(candidate_limit, int)
        ):
            raise TypeError(
                "Memory retrieval candidate limit must be an integer."
            )

        if (
            candidate_limit < 1
            or candidate_limit > MAX_MEMORY_LIST_LIMIT
        ):
            raise ValueError(
                "Memory retrieval candidate limit must be between 1 and 1000."
            )

        self._repository = repository
        self._candidate_limit = candidate_limit

    def retrieve(
        self,
        query: str,
        limit: int = DEFAULT_CONTEXTUAL_MEMORY_LIMIT,
    ) -> tuple[RetrievedMemory, ...]:
        """Return the strongest lexical matches in deterministic order."""
        query_terms = _useful_terms(query)
        normalized_limit = _validate_retrieval_limit(limit)

        if not query_terms:
            return ()

        matches: list[RetrievedMemory] = []

        for memory in self._repository.list_memories(
            limit=self._candidate_limit,
        ):
            memory_terms = _useful_terms(memory.content)
            matched_terms = tuple(
                sorted(query_terms.intersection(memory_terms))
            )

            if not matched_terms:
                continue

            matches.append(
                RetrievedMemory(
                    memory=memory,
                    score=len(matched_terms) / len(query_terms),
                    matched_terms=matched_terms,
                )
            )

        matches.sort(
            key=lambda match: (
                match.score,
                match.memory.confidence,
                match.memory.created_at,
                match.memory.id,
            ),
            reverse=True,
        )

        return tuple(matches[:normalized_limit])


def _useful_terms(value: str) -> frozenset[str]:
    """Return distinct normalized lexical terms from one text."""
    if not isinstance(value, str):
        raise TypeError(
            "Memory retrieval text must be a string."
        )

    normalized_value = "".join(
        character
        for character in unicodedata.normalize(
            "NFKD",
            value.casefold(),
        )
        if not unicodedata.combining(character)
    )

    return frozenset(
        token
        for token in _TOKEN_PATTERN.findall(normalized_value)
        if len(token) >= MIN_TOKEN_LENGTH
        and token not in IGNORED_TERMS
    )


def _validate_retrieval_limit(limit: int) -> int:
    """Return a bounded positive contextual result limit."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Memory retrieval limit must be an integer."
        )

    if limit < 1 or limit > MAX_CONTEXTUAL_MEMORY_LIMIT:
        raise ValueError(
            "Memory retrieval limit must be between 1 and 20."
        )

    return limit
