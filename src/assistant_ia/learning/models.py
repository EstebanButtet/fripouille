"""Modèles immuables des expériences et leçons comportementales candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast

from assistant_ia.learning.outcomes import (
    ExperienceOutcome,
    OutcomeKind,
    OutcomeMeasurement,
    OutcomeStatus,
    UserFeedback,
    UserFeedbackKind,
)

ExperienceSourceType = Literal[
    "manual_entry",
    "conversation_turn",
    "action_execution",
    "external_system",
]
LearningRecordStatus = Literal["active", "invalidated"]
BehavioralRuleConfirmation = Literal["explicit_application_confirmation"]

ALLOWED_EXPERIENCE_SOURCE_TYPES = frozenset(
    {
        "manual_entry",
        "conversation_turn",
        "action_execution",
        "external_system",
    }
)
ALLOWED_LEARNING_RECORD_STATUSES = frozenset({"active", "invalidated"})


@dataclass(frozen=True, slots=True)
class BehavioralConsolidation:
    """Synthèse déterministe recalculée à partir des preuves d'un candidat."""

    candidate_id: int
    relevant_experience_ids: tuple[int, ...]
    favorable_experience_ids: tuple[int, ...]
    contradictory_experience_ids: tuple[int, ...]
    ambiguous_experience_ids: tuple[int, ...]
    excluded_experience_ids: tuple[int, ...]
    duplicate_experience_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "relevant_experience_ids", "favorable_experience_ids",
            "contradictory_experience_ids", "ambiguous_experience_ids",
            "excluded_experience_ids", "duplicate_experience_ids",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in value
            ):
                raise TypeError(f"{name} must contain positive integer identifiers.")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} identifiers must be unique.")
        object.__setattr__(self, "candidate_id", _validate_identifier(self.candidate_id, "Candidate"))

    @property
    def has_contradictions(self) -> bool:
        return bool(self.contradictory_experience_ids)

    @property
    def can_be_confirmed(self) -> bool:
        return (
            bool(self.favorable_experience_ids)
            and not self.has_contradictions
            and not self.ambiguous_experience_ids
        )


@dataclass(frozen=True, slots=True)
class ConfirmedBehavioralRule:
    """Règle comportementale confirmée explicitement et liée à ses preuves."""

    id: int
    person_id: int | None
    context_pattern: str
    proposed_strategy: str
    rationale: str
    source_experience_ids: tuple[int, ...]
    confirmation: BehavioralRuleConfirmation
    status: LearningRecordStatus
    invalidation_reason: str | None
    confirmed_at: datetime
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        created = _normalize_datetime(self.created_at, "Rule creation time")
        updated = _normalize_datetime(self.updated_at, "Rule update time")
        confirmed = _normalize_datetime(self.confirmed_at, "Rule confirmation time")
        if updated < created or confirmed < created:
            raise ValueError("Rule timestamps are inconsistent.")
        status, reason = _normalize_status(self.status, self.invalidation_reason, subject="Rule")
        if self.confirmation != "explicit_application_confirmation":
            raise ValueError("Unknown behavioral rule confirmation.")
        source_ids = _normalize_source_ids(self.source_experience_ids)
        object.__setattr__(self, "id", _validate_identifier(self.id, "Rule"))
        object.__setattr__(self, "person_id", _validate_optional_identifier(self.person_id, "Person"))
        for field, label in (("context_pattern", "Rule context pattern"), ("proposed_strategy", "Rule strategy"), ("rationale", "Rule rationale")):
            object.__setattr__(self, field, _normalize_required_text(getattr(self, field), label))
        object.__setattr__(self, "source_experience_ids", source_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "invalidation_reason", reason)
        object.__setattr__(self, "confirmed_at", confirmed)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)


@dataclass(frozen=True, slots=True)
class ExperienceProvenance:
    """Décrire l'origine réelle d'une expérience sans texte opaque unique."""

    source_type: ExperienceSourceType
    source_reference: str | None = None
    source_text: str | None = None

    def __post_init__(self) -> None:
        source_type = _normalize_choice(
            self.source_type,
            ALLOWED_EXPERIENCE_SOURCE_TYPES,
            "experience source type",
        )
        source_reference = _normalize_optional_text(
            self.source_reference,
            "Experience source reference",
        )
        source_text = _normalize_optional_text(
            self.source_text,
            "Experience source text",
            preserve=True,
        )

        if source_type == "conversation_turn" and source_text is None:
            raise ValueError(
                "Conversation experiences require their exact source text."
            )
        if (
            source_type in {"action_execution", "external_system"}
            and source_reference is None
        ):
            raise ValueError(
                "Action and external experiences require a source reference."
            )

        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_reference", source_reference)
        object.__setattr__(self, "source_text", source_text)


@dataclass(frozen=True, slots=True)
class BehavioralExperience:
    """Conserver une tentative située, son résultat et sa provenance."""

    id: int
    person_id: int | None
    context: str
    objective: str
    strategy: str
    result: str
    evaluation: str | None
    provenance: ExperienceProvenance
    status: LearningRecordStatus
    invalidation_reason: str | None
    created_at: datetime
    updated_at: datetime
    outcome_status: OutcomeStatus = "unknown"
    outcome_kind: OutcomeKind = "reported_result"
    result_code: str | None = None
    feedback_kind: UserFeedbackKind | None = None
    feedback_text: str | None = None
    measurements: tuple[OutcomeMeasurement, ...] = ()

    def __post_init__(self) -> None:
        created_at = _normalize_datetime(self.created_at, "Experience creation time")
        updated_at = _normalize_datetime(self.updated_at, "Experience update time")
        if updated_at < created_at:
            raise ValueError("Experience update time cannot precede creation time.")
        status, reason = _normalize_status(
            self.status,
            self.invalidation_reason,
            subject="Experience",
        )
        if not isinstance(self.provenance, ExperienceProvenance):
            raise TypeError("Experience provenance must be ExperienceProvenance.")
        feedback = None
        if self.feedback_kind is not None or self.feedback_text is not None:
            if self.feedback_kind is None or self.feedback_text is None:
                raise ValueError(
                    "Experience feedback kind and text must be present together."
                )
            feedback = UserFeedback(
                kind=self.feedback_kind,
                content=self.feedback_text,
            )
        outcome = ExperienceOutcome(
            status=self.outcome_status,
            kind=self.outcome_kind,
            summary=self.result,
            result_code=self.result_code,
            feedback=feedback,
            measurements=self.measurements,
        )

        object.__setattr__(self, "id", _validate_identifier(self.id, "Experience"))
        object.__setattr__(
            self,
            "person_id",
            _validate_optional_identifier(self.person_id, "Person"),
        )
        for field_name, label in (
            ("context", "Experience context"),
            ("objective", "Experience objective"),
            ("strategy", "Experience strategy"),
            ("result", "Experience result"),
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_required_text(getattr(self, field_name), label),
            )
        object.__setattr__(
            self,
            "evaluation",
            _normalize_optional_text(self.evaluation, "Experience evaluation"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "invalidation_reason", reason)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "outcome_status", outcome.status)
        object.__setattr__(self, "outcome_kind", outcome.kind)
        object.__setattr__(self, "result_code", outcome.result_code)
        object.__setattr__(
            self,
            "feedback_kind",
            None if outcome.feedback is None else outcome.feedback.kind,
        )
        object.__setattr__(
            self,
            "feedback_text",
            None if outcome.feedback is None else outcome.feedback.content,
        )
        object.__setattr__(self, "measurements", outcome.measurements)


@dataclass(frozen=True, slots=True)
class BehavioralLessonCandidate:
    """Proposer une stratégie explicable sans constituer une règle stable."""

    id: int
    person_id: int | None
    context_pattern: str
    proposed_strategy: str
    rationale: str
    source_experience_ids: tuple[int, ...]
    status: LearningRecordStatus
    invalidation_reason: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        created_at = _normalize_datetime(self.created_at, "Candidate creation time")
        updated_at = _normalize_datetime(self.updated_at, "Candidate update time")
        if updated_at < created_at:
            raise ValueError("Candidate update time cannot precede creation time.")
        status, reason = _normalize_status(
            self.status,
            self.invalidation_reason,
            subject="Candidate",
        )
        source_ids = _normalize_source_ids(self.source_experience_ids)

        object.__setattr__(self, "id", _validate_identifier(self.id, "Candidate"))
        object.__setattr__(
            self,
            "person_id",
            _validate_optional_identifier(self.person_id, "Person"),
        )
        for field_name, label in (
            ("context_pattern", "Candidate context pattern"),
            ("proposed_strategy", "Candidate proposed strategy"),
            ("rationale", "Candidate rationale"),
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_required_text(getattr(self, field_name), label),
            )
        object.__setattr__(self, "source_experience_ids", source_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "invalidation_reason", reason)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


def _normalize_source_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Candidate source experience identifiers must be a tuple.")
    if not value:
        raise ValueError("A lesson candidate requires at least one source experience.")
    normalized = tuple(
        _validate_identifier(item, "Experience") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Candidate source experience identifiers must be unique.")
    return tuple(sorted(normalized))


def _normalize_status(
    status: object,
    reason: object,
    *,
    subject: str,
) -> tuple[LearningRecordStatus, str | None]:
    normalized_status = _normalize_choice(
        status,
        ALLOWED_LEARNING_RECORD_STATUSES,
        f"{subject.lower()} status",
    )
    normalized_reason = _normalize_optional_text(
        reason,
        f"{subject} invalidation reason",
    )
    if normalized_status == "active" and normalized_reason is not None:
        raise ValueError(f"Active {subject.lower()}s cannot have an invalidation reason.")
    if normalized_status == "invalidated" and normalized_reason is None:
        raise ValueError(f"Invalidated {subject.lower()}s require a reason.")
    return cast(LearningRecordStatus, normalized_status), normalized_reason


def _validate_identifier(value: object, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} identifier must be an integer.")
    if value < 1:
        raise ValueError(f"{subject} identifier must be greater than zero.")
    return value


def _validate_optional_identifier(value: object, subject: str) -> int | None:
    if value is None:
        return None
    return _validate_identifier(value, subject)


def _normalize_required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_text(
    value: object,
    field_name: str,
    *,
    preserve: bool = False,
) -> str | None:
    if value is None:
        return None
    normalized = _normalize_required_text(value, field_name)
    return cast(str, value) if preserve else normalized


def _normalize_choice(value: object, allowed: frozenset[str], field_name: str) -> str:
    normalized = _normalize_required_text(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"Unknown {field_name}: {normalized!r}.")
    return normalized


def _normalize_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)
