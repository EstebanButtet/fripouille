"""Directives internes qui choisissent le mode de génération conversationnel.

Ollama peut proposer un mode spécialisé, mais cette proposition reste non
fiable. :func:`resolve_conversation_directive` la recoupe avec le texte exact
de l'utilisateur avant de produire une directive autorisée. Ce module ne
génère aucun texte et n'exécute aucune action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from assistant_ia.intelligence.allocation import (
    AllocationTarget,
    AllocationTargetParseError,
    parse_duration_target_text,
)


class ConversationMode(str, Enum):
    """Énumérer les seuls modes de génération connus de l'application."""

    STANDARD = "standard"
    FIXED_TOTAL_ALLOCATION = "fixed_total_allocation"


class ConversationDirectiveResolutionError(ValueError):
    """Signaler qu'une proposition du modèle ne peut pas être vérifiée."""


@dataclass(frozen=True, slots=True)
class ConversationDirectiveProposal:
    """Porter des métadonnées de routage proposées par le modèle.

    ``target_text`` est une preuve textuelle annoncée, pas encore une valeur
    autoritative. Le mot « Proposal » rappelle que cet objet ne doit pas être
    utilisé directement pour calculer une répartition.
    """

    mode: ConversationMode
    target_text: str | None = None

    def __post_init__(self) -> None:
        """Vérifier la cohérence entre le mode proposé et son texte cible."""
        if not isinstance(self.mode, ConversationMode):
            raise TypeError(
                "Conversation directive proposal mode must be a "
                "ConversationMode."
            )

        if (
            self.target_text is not None
            and not isinstance(
                self.target_text,
                str,
            )
        ):
            raise TypeError(
                "Conversation directive proposal target text "
                "must be a string."
            )

        normalized_target_text = (
            self.target_text.strip()
            if self.target_text is not None
            else None
        )

        if self.mode is ConversationMode.STANDARD:
            if normalized_target_text is not None:
                raise ValueError(
                    "Standard conversation proposal cannot have "
                    "target text."
                )

            return

        if self.mode is ConversationMode.FIXED_TOTAL_ALLOCATION:
            if not normalized_target_text:
                raise ValueError(
                    "Fixed-total conversation proposal requires "
                    "target text."
                )

            object.__setattr__(
                self,
                "target_text",
                normalized_target_text,
            )

            return

        raise ValueError(
            "Unsupported conversation directive proposal mode."
        )

    @classmethod
    def standard(
        cls,
    ) -> "ConversationDirectiveProposal":
        """Construire une proposition de conversation ordinaire."""
        return cls(
            mode=ConversationMode.STANDARD,
            target_text=None,
        )

    @classmethod
    def fixed_total_allocation(
        cls,
        target_text: str,
    ) -> "ConversationDirectiveProposal":
        """Construire une proposition de total fixe ancrée dans le message."""
        return cls(
            mode=ConversationMode.FIXED_TOTAL_ALLOCATION,
            target_text=target_text,
        )


@dataclass(frozen=True, slots=True)
class ConversationDirective:
    """Décrire un mode de génération vérifié par l'application.

    Contrairement à la proposition, ``allocation_target`` est un total exact
    construit déterministement depuis le message utilisateur.
    """

    mode: ConversationMode
    allocation_target: AllocationTarget | None = None

    def __post_init__(self) -> None:
        """Garantir que le mode vérifié possède exactement les données requises."""
        if not isinstance(self.mode, ConversationMode):
            raise TypeError(
                "Conversation directive mode must be a ConversationMode."
            )

        if (
            self.allocation_target is not None
            and not isinstance(
                self.allocation_target,
                AllocationTarget,
            )
        ):
            raise TypeError(
                "Conversation allocation target must be an "
                "AllocationTarget."
            )

        if self.mode is ConversationMode.STANDARD:
            if self.allocation_target is not None:
                raise ValueError(
                    "Standard conversation cannot have an "
                    "allocation target."
                )

            return

        if self.mode is ConversationMode.FIXED_TOTAL_ALLOCATION:
            if self.allocation_target is None:
                raise ValueError(
                    "Fixed-total conversation requires an "
                    "allocation target."
                )

            return

        raise ValueError(
            "Unsupported conversation directive mode."
        )

    @classmethod
    def standard(
        cls,
    ) -> "ConversationDirective":
        """Construire la directive de génération conversationnelle ordinaire."""
        return cls(
            mode=ConversationMode.STANDARD,
            allocation_target=None,
        )

    @classmethod
    def fixed_total_allocation(
        cls,
        target: AllocationTarget,
    ) -> "ConversationDirective":
        """Construire une directive de répartition avec sa cible autoritative."""
        return cls(
            mode=ConversationMode.FIXED_TOTAL_ALLOCATION,
            allocation_target=target,
        )

_APPROXIMATE_PREFIX_PATTERNS = (
    r"\benviron\s*$",
    r"\bapproximativement\s*$",
    r"\bpresque\s*$",
    r"\bvers\s*$",
    r"\bautour de\s*$",
    r"\b\u00e0 peu pr\u00e8s\s*$",
)

_APPROXIMATE_SUFFIX_PATTERNS = (
    r"^\s*environ\b",
    r"^\s*approximativement\b",
    r"^\s*\u00e0 peu pr\u00e8s\b",
)

_RANGE_OR_BOUND_PREFIX_PATTERNS = (
    r"\b[0-9]+(?:[.,][0-9]+)?\s*[-\u2013\u2014]\s*$",
    r"\bentre\s+[0-9]+(?:[.,][0-9]+)?\s+et\s*$",
    r"\b[0-9]+(?:[.,][0-9]+)?\s+\u00e0\s*$",
    r"\bde\s+[0-9]+(?:[.,][0-9]+)?\s+\u00e0\s*$",
    r"\bjusqu['\u2019]\u00e0\s*$",
    r"\bau maximum\s*$",
    r"\bmaximum\s*$",
    r"\bau moins\s*$",
    r"\bminimum\s*$",
    r"\bmoins de\s*$",
    r"\bplus de\s*$",
)

_RANGE_OR_BOUND_SUFFIX_PATTERNS = (
    r"^\s*[-\u2013\u2014]\s*[0-9]+",
    r"^\s+\u00e0\s+[0-9]+",
    r"^\s+et\s+[0-9]+",
)


def _normalize_directive_evidence_text(
    value: str,
) -> str:
    """Normaliser casse et espaces sans réinterpréter le contenu."""
    return " ".join(
        value.strip().casefold().split()
    )


def _duration_evidence_occurrences(
    *,
    target_text: str,
    user_message: str,
) -> tuple[tuple[int, int], ...]:
    """Localiser les occurrences autonomes de la preuve dans le message."""
    pattern = re.compile(
        r"(?<!\w)"
        + re.escape(target_text)
        + r"(?!\w)"
    )

    return tuple(
        (
            match.start(),
            match.end(),
        )
        for match in pattern.finditer(
            user_message
        )
    )


def _has_unsafe_duration_context(
    *,
    user_message: str,
    start: int,
    end: int,
) -> bool:
    """Détecter approximation, intervalle ou borne autour d'une durée."""
    prefix = user_message[:start]
    suffix = user_message[end:]

    if any(
        re.search(
            pattern,
            prefix,
        )
        is not None
        for pattern in _APPROXIMATE_PREFIX_PATTERNS
    ):
        return True

    if any(
        re.search(
            pattern,
            suffix,
        )
        is not None
        for pattern in _APPROXIMATE_SUFFIX_PATTERNS
    ):
        return True

    if any(
        re.search(
            pattern,
            prefix,
        )
        is not None
        for pattern in _RANGE_OR_BOUND_PREFIX_PATTERNS
    ):
        return True

    if any(
        re.search(
            pattern,
            suffix,
        )
        is not None
        for pattern in _RANGE_OR_BOUND_SUFFIX_PATTERNS
    ):
        return True

    return False


def resolve_conversation_directive(
    proposal: ConversationDirectiveProposal,
    *,
    user_message: str,
) -> ConversationDirective:
    """Résoudre une proposition non fiable contre le texte utilisateur.

    La durée doit apparaître textuellement, hors contexte approximatif ou
    borné, puis être comprise par le parseur déterministe. Sinon une
    ``ConversationDirectiveResolutionError`` est levée et le client revient à
    la conversation standard.
    """
    if not isinstance(
        proposal,
        ConversationDirectiveProposal,
    ):
        raise TypeError(
            "Conversation directive proposal must be a "
            "ConversationDirectiveProposal."
        )

    if not isinstance(
        user_message,
        str,
    ):
        raise TypeError(
            "Conversation directive user message must be a string."
        )

    if proposal.mode is ConversationMode.STANDARD:
        return ConversationDirective.standard()

    if proposal.mode is not ConversationMode.FIXED_TOTAL_ALLOCATION:
        raise ConversationDirectiveResolutionError(
            "Unsupported conversation directive proposal."
        )

    if proposal.target_text is None:
        raise ConversationDirectiveResolutionError(
            "Fixed-total proposal is missing target evidence."
        )

    normalized_target = _normalize_directive_evidence_text(
        proposal.target_text
    )
    normalized_message = _normalize_directive_evidence_text(
        user_message
    )

    # La proposition du LLM fournit seulement une piste. La source d'autorité
    # reste le texte réellement envoyé par l'utilisateur.
    occurrences = _duration_evidence_occurrences(
        target_text=normalized_target,
        user_message=normalized_message,
    )

    if not occurrences:
        raise ConversationDirectiveResolutionError(
            "Proposed target evidence is not present in the user message."
        )

    safe_occurrence_found = any(
        not _has_unsafe_duration_context(
            user_message=normalized_message,
            start=start,
            end=end,
        )
        for start, end in occurrences
    )

    if not safe_occurrence_found:
        raise ConversationDirectiveResolutionError(
            "Proposed duration is approximate, ranged or bounded."
        )

    try:
        target = parse_duration_target_text(
            normalized_target
        )
    except AllocationTargetParseError as error:
        raise ConversationDirectiveResolutionError(
            "Proposed target evidence is not an exact supported duration."
        ) from error

    return ConversationDirective.fixed_total_allocation(
        target
    )
