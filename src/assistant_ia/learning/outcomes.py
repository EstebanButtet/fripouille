"""Tentatives et retours d'expérience structurés, sans consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal, cast

from assistant_ia.actions.result import ActionExecutionResult

OutcomeStatus = Literal[
    "success",
    "failure",
    "partial",
    "unknown",
    "not_executed",
]
OutcomeKind = Literal[
    "reported_result",
    "verified_action",
    "technical_error",
    "user_feedback",
    "external_result",
]
UserFeedbackKind = Literal[
    "approval",
    "disapproval",
    "correction",
    "retry_request",
]

ALLOWED_OUTCOME_STATUSES = frozenset(
    {"success", "failure", "partial", "unknown", "not_executed"}
)
ALLOWED_OUTCOME_KINDS = frozenset(
    {
        "reported_result",
        "verified_action",
        "technical_error",
        "user_feedback",
        "external_result",
    }
)
ALLOWED_USER_FEEDBACK_KINDS = frozenset(
    {"approval", "disapproval", "correction", "retry_request"}
)


@dataclass(frozen=True, slots=True)
class BehavioralAttempt:
    """Décrire une stratégie tentée avant de connaître son issue réelle."""

    person_id: int | None
    context: str
    objective: str
    strategy: str
    started_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "person_id",
            _validate_optional_identifier(self.person_id),
        )
        for name, label in (
            ("context", "Attempt context"),
            ("objective", "Attempt objective"),
            ("strategy", "Attempt strategy"),
        ):
            object.__setattr__(self, name, _normalize_text(getattr(self, name), label))
        object.__setattr__(
            self,
            "started_at",
            _normalize_datetime(self.started_at, "Attempt start time"),
        )


@dataclass(frozen=True, slots=True)
class OutcomeMeasurement:
    """Conserver une mesure factuelle nommée, sans l'interpréter en score."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("Outcome measurement value must be a number.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("Outcome measurement value must be finite.")
        object.__setattr__(self, "name", _normalize_text(self.name, "Measurement name"))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", _normalize_text(self.unit, "Measurement unit"))


@dataclass(frozen=True, slots=True)
class UserFeedback:
    """Représenter un retour utilisateur explicite sans le classifier implicitement."""

    kind: UserFeedbackKind
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            cast(
                UserFeedbackKind,
                _normalize_choice(
                    self.kind,
                    ALLOWED_USER_FEEDBACK_KINDS,
                    "user feedback kind",
                ),
            ),
        )
        object.__setattr__(
            self,
            "content",
            _normalize_text(self.content, "User feedback content"),
        )


@dataclass(frozen=True, slots=True)
class ExperienceOutcome:
    """Décrire un résultat observé sans en déduire une règle comportementale."""

    status: OutcomeStatus
    kind: OutcomeKind
    summary: str
    result_code: str | None = None
    feedback: UserFeedback | None = None
    measurements: tuple[OutcomeMeasurement, ...] = ()

    def __post_init__(self) -> None:
        status = cast(
            OutcomeStatus,
            _normalize_choice(
                self.status,
                ALLOWED_OUTCOME_STATUSES,
                "outcome status",
            ),
        )
        kind = cast(
            OutcomeKind,
            _normalize_choice(self.kind, ALLOWED_OUTCOME_KINDS, "outcome kind"),
        )
        result_code = _normalize_optional_text(self.result_code, "Outcome result code")
        if self.feedback is not None and not isinstance(self.feedback, UserFeedback):
            raise TypeError("Outcome feedback must be UserFeedback or None.")
        if kind == "user_feedback" and self.feedback is None:
            raise ValueError("User-feedback outcomes require explicit feedback.")
        if kind != "user_feedback" and self.feedback is not None:
            raise ValueError("Only user-feedback outcomes may contain feedback.")
        if kind == "technical_error" and status != "failure":
            raise ValueError("Technical-error outcomes must have failure status.")
        measurements = _normalize_measurements(self.measurements)

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "summary", _normalize_text(self.summary, "Outcome summary"))
        object.__setattr__(self, "result_code", result_code)
        object.__setattr__(self, "measurements", measurements)


def outcome_from_action_result(result: ActionExecutionResult) -> ExperienceOutcome:
    """Traduire un résultat applicatif sans consulter ni croire le LLM."""
    if not isinstance(result, ActionExecutionResult):
        raise TypeError("Action outcome conversion requires ActionExecutionResult.")
    if result.status == "success":
        return ExperienceOutcome(
            status="success",
            kind="verified_action",
            summary=result.message,
        )
    if result.status == "cancelled":
        return ExperienceOutcome(
            status="not_executed",
            kind="verified_action",
            summary=result.message,
            result_code="action_cancelled",
        )
    if result.error_kind == "validation":
        return ExperienceOutcome(
            status="not_executed",
            kind="verified_action",
            summary=result.message,
            result_code="action_validation_error",
        )
    return ExperienceOutcome(
        status="failure",
        kind="technical_error",
        summary=result.message,
        result_code="action_execution_error",
    )


def _normalize_measurements(value: object) -> tuple[OutcomeMeasurement, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Outcome measurements must be a tuple.")
    if any(not isinstance(item, OutcomeMeasurement) for item in value):
        raise TypeError("Outcome measurements must contain OutcomeMeasurement items.")
    keys = tuple((item.name.casefold(), item.unit.casefold()) for item in value)
    if len(set(keys)) != len(keys):
        raise ValueError("Outcome measurement name and unit pairs must be unique.")
    return tuple(sorted(value, key=lambda item: (item.name.casefold(), item.unit.casefold())))


def _validate_optional_identifier(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Attempt person identifier must be an integer or None.")
    if value < 1:
        raise ValueError("Attempt person identifier must be greater than zero.")
    return value


def _normalize_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_text(value, field_name)


def _normalize_choice(value: object, allowed: frozenset[str], field_name: str) -> str:
    normalized = _normalize_text(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"Unknown {field_name}: {normalized!r}.")
    return normalized


def _normalize_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)
