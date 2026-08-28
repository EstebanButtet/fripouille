"""Controlled application-owned promotion of memory candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, cast
import unicodedata

from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import Memory, MemoryCandidate

MemoryPromotionOperation = Literal[
    "create",
    "already_known",
    "possible_duplicate",
    "update",
    "conflict",
]

ALLOWED_MEMORY_PROMOTION_OPERATIONS: frozenset[str] = frozenset(
    {
        "create",
        "already_known",
        "possible_duplicate",
        "update",
        "conflict",
    }
)

MAX_PROMOTION_COMPARISON_MEMORIES = 500
QUASI_DUPLICATE_SIMILARITY = 0.6

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CORRECTION_MARKERS = (
    "a partir de maintenant",
    "desormais",
    "en fait",
    "je corrige",
    "maintenant",
    "plutot",
)
_RELATION_ANCHORS = frozenset(
    {
        "choix",
        "favori",
        "favorite",
        "outil",
        "preference",
        "prefere",
        "preferes",
        "logiciel",
        "utilise",
    }
)
_IGNORED_COMPARISON_TERMS = frozenset(
    {
        "a",
        "au",
        "aux",
        "comme",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "faire",
        "je",
        "la",
        "le",
        "les",
        "ma",
        "mes",
        "mon",
        "pour",
        "sur",
        "un",
        "une",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryPromotionProposal:
    """Describe one bounded persistence operation chosen by the application."""

    operation: MemoryPromotionOperation
    candidate: MemoryCandidate
    related_memory: Memory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str):
            raise TypeError("Memory promotion operation must be a string.")

        normalized_operation = self.operation.strip()
        if normalized_operation not in ALLOWED_MEMORY_PROMOTION_OPERATIONS:
            raise ValueError(
                f"Unknown memory promotion operation: {normalized_operation!r}."
            )
        if not isinstance(self.candidate, MemoryCandidate):
            raise TypeError(
                "Memory promotion requires a validated MemoryCandidate."
            )

        requires_related_memory = normalized_operation != "create"
        if requires_related_memory and not isinstance(
            self.related_memory,
            Memory,
        ):
            raise TypeError(
                "This memory promotion operation requires an existing Memory."
            )
        if not requires_related_memory and self.related_memory is not None:
            raise ValueError(
                "A create proposal cannot select an existing Memory."
            )

        object.__setattr__(
            self,
            "operation",
            cast(MemoryPromotionOperation, normalized_operation),
        )

    @property
    def requires_confirmation(self) -> bool:
        """Return whether the proposal may mutate persistence after consent."""
        return self.operation != "already_known"


class MemoryPromotionService:
    """Compare, propose and apply explicitly confirmed memory changes."""

    def __init__(self, repository: MemoryRepository) -> None:
        if not isinstance(repository, MemoryRepository):
            raise TypeError(
                "Memory promotion requires a MemoryRepository."
            )
        self._repository = repository

    def propose(
        self,
        candidate: MemoryCandidate,
    ) -> MemoryPromotionProposal:
        """Classify one validated candidate without writing persistence."""
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "Memory promotion requires a validated MemoryCandidate."
            )

        memories = self._repository.list_memories(
            limit=MAX_PROMOTION_COMPARISON_MEMORIES
        )
        candidate_equivalent = normalize_memory_equivalence(
            candidate.content
        )

        for memory in memories:
            if normalize_memory_equivalence(memory.content) == (
                candidate_equivalent
            ):
                return MemoryPromotionProposal(
                    operation="already_known",
                    candidate=candidate,
                    related_memory=memory,
                )

        ranked_memories = sorted(
            memories,
            key=lambda memory: (
                _lexical_similarity(candidate.content, memory.content),
                memory.updated_at,
                memory.id,
            ),
            reverse=True,
        )

        if _is_explicit_correction(candidate.source_text):
            correction_target = _first_related_memory(
                candidate.content,
                ranked_memories,
            )
            if correction_target is not None:
                return MemoryPromotionProposal(
                    operation="update",
                    candidate=candidate,
                    related_memory=correction_target,
                )

        if ranked_memories:
            closest_memory = ranked_memories[0]
            closest_similarity = _lexical_similarity(
                candidate.content,
                closest_memory.content,
            )
            if closest_similarity >= QUASI_DUPLICATE_SIMILARITY:
                return MemoryPromotionProposal(
                    operation="possible_duplicate",
                    candidate=candidate,
                    related_memory=closest_memory,
                )

            if _has_shared_relation_anchor(
                candidate.content,
                closest_memory.content,
            ):
                return MemoryPromotionProposal(
                    operation="conflict",
                    candidate=candidate,
                    related_memory=closest_memory,
                )

        return MemoryPromotionProposal(
            operation="create",
            candidate=candidate,
        )

    def apply_confirmed(
        self,
        proposal: MemoryPromotionProposal,
    ) -> Memory:
        """Apply only a still-current proposal after external confirmation."""
        if not isinstance(proposal, MemoryPromotionProposal):
            raise TypeError(
                "Confirmed promotion requires a MemoryPromotionProposal."
            )
        if not proposal.requires_confirmation:
            if proposal.related_memory is None:
                raise ValueError("Known memory proposal is incomplete.")
            return proposal.related_memory

        current_proposal = self.propose(proposal.candidate)

        if current_proposal.operation == "already_known":
            if current_proposal.related_memory is None:
                raise ValueError("Known memory proposal is incomplete.")
            return current_proposal.related_memory

        if current_proposal.operation != proposal.operation:
            raise ValueError("Memory promotion proposal is no longer current.")

        if proposal.operation in {"create", "possible_duplicate"}:
            return self._repository.save_candidate(proposal.candidate)

        current_memory = current_proposal.related_memory
        proposed_memory = proposal.related_memory
        if current_memory is None or proposed_memory is None:
            raise ValueError("Memory update proposal is incomplete.")
        if current_memory.id != proposed_memory.id:
            raise ValueError("Memory promotion target is no longer current.")

        return self._repository.update_memory(
            current_memory.id,
            proposal.candidate,
        )


def normalize_memory_equivalence(value: str) -> str:
    """Normalize Unicode, case, spacing and cosmetic punctuation."""
    if not isinstance(value, str):
        raise TypeError("Memory comparison text must be a string.")
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(_WORD_PATTERN.findall(normalized))


def _comparison_terms(value: str) -> frozenset[str]:
    return frozenset(
        term
        for term in normalize_memory_equivalence(value).split()
        if len(term) >= 2 and term not in _IGNORED_COMPARISON_TERMS
    )


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = _comparison_terms(left)
    right_terms = _comparison_terms(right)
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _is_explicit_correction(source_text: str) -> bool:
    normalized = normalize_memory_equivalence(source_text)
    return any(marker in normalized for marker in _CORRECTION_MARKERS)


def _has_shared_relation_anchor(left: str, right: str) -> bool:
    left_terms = _comparison_terms(left)
    right_terms = _comparison_terms(right)
    shared_terms = left_terms & right_terms
    if not shared_terms & _RELATION_ANCHORS:
        return False
    if shared_terms - _RELATION_ANCHORS:
        return True
    return max(len(left_terms), len(right_terms)) <= 3


def _first_related_memory(
    candidate_content: str,
    memories: list[Memory],
) -> Memory | None:
    candidate_terms = _comparison_terms(candidate_content)
    for memory in memories:
        memory_terms = _comparison_terms(memory.content)
        if (
            candidate_terms
            & memory_terms
            & _RELATION_ANCHORS
        ):
            return memory
    return None
