"""Promotion contrôlée des candidats de profil, limitée à leur personne."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, cast
import unicodedata

from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.profile_models import ProfileFact, ProfileFactCandidate

ProfileFactPromotionOperation = Literal[
    "create",
    "already_known",
    "update",
    "conflict",
]

_ALLOWED_OPERATIONS = frozenset(
    {"create", "already_known", "update", "conflict"}
)
_CORRECTION_MARKERS = (
    "a partir de maintenant",
    "desormais",
    "en fait",
    "je corrige",
    "plutot",
)
_IGNORED_TERMS = frozenset(
    {
        "a", "au", "aux", "de", "des", "du", "en", "est", "et",
        "fait", "j", "je", "la", "le", "les", "ma", "mes", "mon",
        "maintenant", "pour", "prefere", "plutot", "un", "une",
    }
)
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_RELATED_SIMILARITY = 0.25
MAX_PROFILE_FACT_COMPARISONS = 500


@dataclass(frozen=True, slots=True)
class ProfileFactPromotionProposal:
    """Décrire une décision applicative sans effectuer d'écriture."""

    operation: ProfileFactPromotionOperation
    candidate: ProfileFactCandidate
    related_fact: ProfileFact | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str):
            raise TypeError("Profile promotion operation must be a string.")
        operation = self.operation.strip()
        if operation not in _ALLOWED_OPERATIONS:
            raise ValueError(f"Unknown profile promotion operation: {operation!r}.")
        if not isinstance(self.candidate, ProfileFactCandidate):
            raise TypeError("Profile promotion requires a ProfileFactCandidate.")
        needs_fact = operation != "create"
        if needs_fact and not isinstance(self.related_fact, ProfileFact):
            raise TypeError("This profile promotion requires an existing fact.")
        if not needs_fact and self.related_fact is not None:
            raise ValueError("A create proposal cannot select an existing fact.")
        if (
            self.related_fact is not None
            and self.related_fact.person_id != self.candidate.person_id
        ):
            raise ValueError("A profile proposal cannot cross person boundaries.")
        object.__setattr__(
            self,
            "operation",
            cast(ProfileFactPromotionOperation, operation),
        )

    @property
    def requires_confirmation(self) -> bool:
        """Toute opération qui écrit doit attendre un consentement explicite."""
        return self.operation != "already_known"


class ProfileFactPromotionService:
    """Comparer et appliquer les faits sans LLM ni comparaison globale."""

    def __init__(self, repository: ProfileFactRepository) -> None:
        if not isinstance(repository, ProfileFactRepository):
            raise TypeError(
                "Profile promotion requires a ProfileFactRepository."
            )
        self._repository = repository

    def propose(
        self,
        candidate: ProfileFactCandidate,
    ) -> ProfileFactPromotionProposal:
        """Classer le candidat parmi les seuls faits de sa personne."""
        if not isinstance(candidate, ProfileFactCandidate):
            raise TypeError("Profile promotion requires a ProfileFactCandidate.")
        facts = self._repository.list_profile_facts(
            candidate.person_id,
            limit=MAX_PROFILE_FACT_COMPARISONS,
        )
        equivalent = normalize_profile_fact_equivalence(candidate.content)
        for fact in facts:
            if (
                fact.category == candidate.category
                and normalize_profile_fact_equivalence(fact.content) == equivalent
            ):
                return ProfileFactPromotionProposal(
                    "already_known", candidate, fact
                )

        same_category = [
            fact for fact in facts if fact.category == candidate.category
        ]
        ranked = sorted(
            same_category,
            key=lambda fact: (
                _lexical_similarity(candidate.content, fact.content),
                fact.updated_at,
                fact.id,
            ),
            reverse=True,
        )
        related = next(
            (
                fact for fact in ranked
                if _lexical_similarity(candidate.content, fact.content)
                >= _RELATED_SIMILARITY
            ),
            None,
        )
        if related is not None:
            operation: ProfileFactPromotionOperation = (
                "update"
                if _is_explicit_correction(candidate.source_text)
                else "conflict"
            )
            return ProfileFactPromotionProposal(operation, candidate, related)

        return ProfileFactPromotionProposal("create", candidate)

    def apply_confirmed(
        self,
        proposal: ProfileFactPromotionProposal,
    ) -> ProfileFact:
        """Revalider une proposition précise avant toute écriture SQLite."""
        if not isinstance(proposal, ProfileFactPromotionProposal):
            raise TypeError(
                "Confirmed profile promotion requires a profile proposal."
            )
        if not proposal.requires_confirmation:
            if proposal.related_fact is None:
                raise ValueError("Known profile fact proposal is incomplete.")
            return proposal.related_fact

        current = self.propose(proposal.candidate)
        if current.operation == "already_known":
            if current.related_fact is None:
                raise ValueError("Known profile fact proposal is incomplete.")
            return current.related_fact
        if current.operation != proposal.operation:
            raise ValueError("Profile promotion proposal is no longer current.")

        if proposal.operation == "create":
            return self._repository.save_candidate(proposal.candidate)

        current_fact = current.related_fact
        proposed_fact = proposal.related_fact
        if current_fact is None or proposed_fact is None:
            raise ValueError("Profile update proposal is incomplete.")
        if current_fact.id != proposed_fact.id:
            raise ValueError("Profile promotion target is no longer current.")
        return self._repository.update_profile_fact(
            current_fact.id,
            proposal.candidate,
        )


def normalize_profile_fact_equivalence(value: str) -> str:
    """Normaliser accents, casse, espaces et ponctuation cosmétique."""
    if not isinstance(value, str):
        raise TypeError("Profile comparison text must be a string.")
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(_WORD_PATTERN.findall(normalized))


def _comparison_terms(value: str) -> frozenset[str]:
    return frozenset(
        term
        for term in normalize_profile_fact_equivalence(value).split()
        if len(term) >= 2 and term not in _IGNORED_TERMS
    )


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = _comparison_terms(left)
    right_terms = _comparison_terms(right)
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _is_explicit_correction(source_text: str) -> bool:
    normalized = normalize_profile_fact_equivalence(source_text)
    return any(marker in normalized for marker in _CORRECTION_MARKERS)
