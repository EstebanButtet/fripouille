"""Rappel lexical déterministe des souvenirs persistants.

Ce module reçoit le message conversationnel courant, extrait des termes utiles
et classe une vue bornée des souvenirs SQLite. Il produit des
:class:`RetrievedMemory` inspectables et limite ensuite ce qui peut entrer dans
le prompt. Il ne modifie aucun souvenir et n'interprète aucune action.
"""

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
MAX_INJECTED_CONTEXTUAL_MEMORIES = 5
MAX_INJECTED_MEMORY_CONTENT_CHARACTERS = 500
MAX_INJECTED_MEMORY_TOTAL_CHARACTERS = 1500

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
    """Associer un souvenir à sa pertinence lexicale inspectable.

    ``score`` est la part des termes utiles de la requête retrouvée dans le
    souvenir ; ce n'est ni une confiance de vérité ni une décision du LLM.
    ``matched_terms`` rend ce classement explicable.
    """

    memory: Memory
    score: float
    matched_terms: tuple[str, ...]


class ContextualMemoryRetriever:
    """Sélectionner des souvenirs pertinents sans jamais les modifier.

    Le repository fournit une vue récente bornée ; le retriever applique en
    mémoire un classement lexical stable et retourne au plus ``limit`` objets.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        candidate_limit: int = DEFAULT_RETRIEVAL_CANDIDATE_LIMIT,
    ) -> None:
        """Créer le retriever sur une vue bornée du repository."""
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
        """Retourner les correspondances lexicales les plus fortes.

        Les égalités sont départagées par confiance d'enregistrement, date de
        création puis identifiant. Une requête sans terme significatif ne
        rappelle rien, plutôt que de choisir des souvenirs arbitraires.
        """
        query_terms = _useful_terms(query)
        normalized_limit = _validate_retrieval_limit(limit)

        if not query_terms:
            return ()

        matches: list[RetrievedMemory] = []

        # Le score est calculé localement sur des ensembles de mots ; Ollama
        # n'intervient ni dans la sélection ni dans son ordre.
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


def bound_contextual_memories(
    memories: tuple[RetrievedMemory, ...],
) -> tuple[RetrievedMemory, ...]:
    """Sélectionner des souvenirs entiers dans les budgets stricts du prompt.

    Un souvenir trop long est ignoré, jamais tronqué : son sens et sa preuve
    restent intacts. L'ordre classé fourni en entrée est préservé.
    """
    if not isinstance(memories, tuple):
        raise TypeError(
            "Retrieved memories must be provided as a tuple."
        )

    selected: list[RetrievedMemory] = []
    total_content_characters = 0

    for retrieved_memory in memories:
        if not isinstance(retrieved_memory, RetrievedMemory):
            raise TypeError(
                "Contextual memory entries must be RetrievedMemory objects."
            )

        content_length = len(retrieved_memory.memory.content)

        if content_length > MAX_INJECTED_MEMORY_CONTENT_CHARACTERS:
            continue

        if (
            total_content_characters + content_length
            > MAX_INJECTED_MEMORY_TOTAL_CHARACTERS
        ):
            continue

        selected.append(retrieved_memory)
        total_content_characters += content_length

        if len(selected) == MAX_INJECTED_CONTEXTUAL_MEMORIES:
            break

    return tuple(selected)


def _useful_terms(value: str) -> frozenset[str]:
    """Extraire les termes lexicaux distincts et utiles d'un texte."""
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
    """Valider une limite positive de résultats contextuels."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "Memory retrieval limit must be an integer."
        )

    if limit < 1 or limit > MAX_CONTEXTUAL_MEMORY_LIMIT:
        raise ValueError(
            "Memory retrieval limit must be between 1 and 20."
        )

    return limit
