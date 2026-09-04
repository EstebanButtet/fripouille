"""Persistance SQLite des expériences et leçons comportementales candidates."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from json import JSONDecodeError
from typing import cast

from assistant_ia.learning.models import (
    BehavioralExperience,
    BehavioralLessonCandidate,
    ExperienceProvenance,
    ExperienceSourceType,
    LearningRecordStatus,
)
from assistant_ia.learning.outcomes import (
    ExperienceOutcome,
    OutcomeKind,
    OutcomeMeasurement,
    OutcomeStatus,
    UserFeedback,
    UserFeedbackKind,
)
from assistant_ia.memory.errors import RepositoryError
from assistant_ia.memory.repository import SQLiteDatabase

DEFAULT_LEARNING_RESULT_LIMIT = 50
MAX_LEARNING_RESULT_LIMIT = 500

_EXPERIENCE_SELECT = """
    SELECT id, person_id, context, objective, strategy, result, evaluation,
           source_type, source_reference, source_text, status,
           invalidation_reason, created_at, updated_at, outcome_status,
           outcome_kind, result_code, feedback_kind, feedback_text,
           measurements_json
    FROM behavioral_experiences
"""
_CANDIDATE_SELECT = """
    SELECT id, person_id, context_pattern, proposed_strategy, rationale,
           status, invalidation_reason, created_at, updated_at
    FROM behavioral_lesson_candidates
"""


class BehavioralExperienceNotFoundError(RepositoryError):
    """Signaler qu'une expérience demandée n'existe pas."""


class BehavioralLessonCandidateNotFoundError(RepositoryError):
    """Signaler qu'une leçon candidate demandée n'existe pas."""


class BehavioralExperienceInUseError(RepositoryError):
    """Empêcher la suppression d'une preuve encore reliée à un candidat."""


class BehavioralLearningRepository:
    """Gérer des données d'apprentissage inspectables sans les consolider."""

    def __init__(
        self,
        database: SQLiteDatabase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(database, SQLiteDatabase):
            raise TypeError("Learning repository database must be a SQLiteDatabase.")
        if clock is not None and not callable(clock):
            raise TypeError("Learning repository clock must be callable.")
        self._database = database
        self._clock = clock if clock is not None else _utc_now

    def create_experience(
        self,
        *,
        context: str,
        objective: str,
        strategy: str,
        result: str,
        provenance: ExperienceProvenance,
        evaluation: str | None = None,
        person_id: int | None = None,
        outcome: ExperienceOutcome | None = None,
    ) -> BehavioralExperience:
        if outcome is None:
            outcome = ExperienceOutcome(
                status="unknown",
                kind="reported_result",
                summary=result,
            )
        elif not isinstance(outcome, ExperienceOutcome):
            raise TypeError("Experience outcome must be ExperienceOutcome or None.")
        if not isinstance(result, str):
            raise TypeError("Experience result must be a string.")
        result = result.strip()
        if result != outcome.summary:
            raise ValueError("Experience result must match its structured outcome.")
        now = _normalized_now(self._clock())
        validated = BehavioralExperience(
            id=1,
            person_id=person_id,
            context=context,
            objective=objective,
            strategy=strategy,
            result=result,
            evaluation=evaluation,
            provenance=provenance,
            status="active",
            invalidation_reason=None,
            created_at=now,
            updated_at=now,
            outcome_status=outcome.status,
            outcome_kind=outcome.kind,
            result_code=outcome.result_code,
            feedback_kind=(
                None if outcome.feedback is None else outcome.feedback.kind
            ),
            feedback_text=(
                None if outcome.feedback is None else outcome.feedback.content
            ),
            measurements=outcome.measurements,
        )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO behavioral_experiences (
                    person_id, context, objective, strategy, result, evaluation,
                    source_type, source_reference, source_text, status,
                    invalidation_reason, created_at, updated_at
                    , outcome_status, outcome_kind, result_code,
                    feedback_kind, feedback_text, measurements_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.person_id,
                    validated.context,
                    validated.objective,
                    validated.strategy,
                    validated.result,
                    validated.evaluation,
                    validated.provenance.source_type,
                    validated.provenance.source_reference,
                    validated.provenance.source_text,
                    validated.status,
                    validated.invalidation_reason,
                    now.isoformat(),
                    now.isoformat(),
                    validated.outcome_status,
                    validated.outcome_kind,
                    validated.result_code,
                    validated.feedback_kind,
                    validated.feedback_text,
                    _serialize_measurements(validated.measurements),
                ),
            )
            experience_id = _created_identifier(cursor.lastrowid, "experience")
            row = connection.execute(
                f"{_EXPERIENCE_SELECT} WHERE id = ?", (experience_id,)
            ).fetchone()
        return _experience_from_row(row)

    def get_experience(self, experience_id: int) -> BehavioralExperience | None:
        normalized_id = _validate_identifier(experience_id, "Experience")
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_EXPERIENCE_SELECT} WHERE id = ?", (normalized_id,)
            ).fetchone()
        return None if row is None else _experience_from_row(row)

    def list_person_experiences(
        self,
        person_id: int,
        *,
        include_invalidated: bool = False,
        limit: int = DEFAULT_LEARNING_RESULT_LIMIT,
    ) -> tuple[BehavioralExperience, ...]:
        return self._list_experiences(
            person_id=_validate_identifier(person_id, "Person"),
            include_invalidated=include_invalidated,
            limit=limit,
        )

    def list_global_experiences(
        self,
        *,
        include_invalidated: bool = False,
        limit: int = DEFAULT_LEARNING_RESULT_LIMIT,
    ) -> tuple[BehavioralExperience, ...]:
        return self._list_experiences(
            person_id=None,
            include_invalidated=include_invalidated,
            limit=limit,
        )

    def _list_experiences(
        self,
        *,
        person_id: int | None,
        include_invalidated: bool,
        limit: int,
    ) -> tuple[BehavioralExperience, ...]:
        normalized_limit = _validate_limit(limit)
        if not isinstance(include_invalidated, bool):
            raise TypeError("Learning invalidated inclusion flag must be boolean.")
        scope_clause = "person_id IS NULL" if person_id is None else "person_id = ?"
        parameters: list[object] = [] if person_id is None else [person_id]
        status_clause = "" if include_invalidated else " AND status = 'active'"
        parameters.append(normalized_limit)
        with self._database.connect() as connection:
            rows = connection.execute(
                f"""
                {_EXPERIENCE_SELECT}
                WHERE {scope_clause}{status_clause}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(_experience_from_row(row) for row in rows)

    def update_experience(
        self,
        experience_id: int,
        *,
        context: str,
        objective: str,
        strategy: str,
        result: str,
        evaluation: str | None = None,
        outcome: ExperienceOutcome | None = None,
    ) -> BehavioralExperience:
        existing = self._require_experience(experience_id)
        if outcome is None:
            feedback = None
            if existing.feedback_kind is not None:
                assert existing.feedback_text is not None
                feedback = UserFeedback(
                    kind=existing.feedback_kind,
                    content=existing.feedback_text,
                )
            outcome = ExperienceOutcome(
                status=existing.outcome_status,
                kind=existing.outcome_kind,
                summary=result,
                result_code=existing.result_code,
                feedback=feedback,
                measurements=existing.measurements,
            )
        elif not isinstance(outcome, ExperienceOutcome):
            raise TypeError("Experience outcome must be ExperienceOutcome or None.")
        if not isinstance(result, str):
            raise TypeError("Experience result must be a string.")
        result = result.strip()
        if result != outcome.summary:
            raise ValueError("Experience result must match its structured outcome.")
        now = _normalized_now(self._clock())
        validated = BehavioralExperience(
            id=existing.id,
            person_id=existing.person_id,
            context=context,
            objective=objective,
            strategy=strategy,
            result=result,
            evaluation=evaluation,
            provenance=existing.provenance,
            status=existing.status,
            invalidation_reason=existing.invalidation_reason,
            created_at=existing.created_at,
            updated_at=now,
            outcome_status=outcome.status,
            outcome_kind=outcome.kind,
            result_code=outcome.result_code,
            feedback_kind=(
                None if outcome.feedback is None else outcome.feedback.kind
            ),
            feedback_text=(
                None if outcome.feedback is None else outcome.feedback.content
            ),
            measurements=outcome.measurements,
        )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE behavioral_experiences
                SET context = ?, objective = ?, strategy = ?, result = ?,
                    evaluation = ?, updated_at = ?, outcome_status = ?,
                    outcome_kind = ?, result_code = ?, feedback_kind = ?,
                    feedback_text = ?, measurements_json = ?
                WHERE id = ?
                """,
                (
                    validated.context,
                    validated.objective,
                    validated.strategy,
                    validated.result,
                    validated.evaluation,
                    now.isoformat(),
                    validated.outcome_status,
                    validated.outcome_kind,
                    validated.result_code,
                    validated.feedback_kind,
                    validated.feedback_text,
                    _serialize_measurements(validated.measurements),
                    existing.id,
                ),
            )
            _require_changed(cursor.rowcount, "Experience update")
            row = connection.execute(
                f"{_EXPERIENCE_SELECT} WHERE id = ?", (existing.id,)
            ).fetchone()
        return _experience_from_row(row)

    def invalidate_experience(
        self,
        experience_id: int,
        reason: str,
    ) -> BehavioralExperience:
        existing = self._require_experience(experience_id)
        if existing.status == "invalidated":
            raise ValueError("Experience is already invalidated.")
        reason = _normalize_text(reason, "Experience invalidation reason")
        now = _normalized_now(self._clock())
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE behavioral_experiences
                SET status = 'invalidated', invalidation_reason = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (reason, now.isoformat(), existing.id),
            )
            _require_changed(cursor.rowcount, "Experience invalidation")
            row = connection.execute(
                f"{_EXPERIENCE_SELECT} WHERE id = ?", (existing.id,)
            ).fetchone()
        return _experience_from_row(row)

    def delete_experience(self, experience_id: int) -> BehavioralExperience:
        existing = self._require_experience(experience_id)
        with self._database.connect() as connection:
            linked = connection.execute(
                """
                SELECT 1 FROM behavioral_lesson_sources
                WHERE experience_id = ? LIMIT 1
                """,
                (existing.id,),
            ).fetchone()
            if linked is not None:
                raise BehavioralExperienceInUseError(
                    "Experience is still evidence for a lesson candidate."
                )
            cursor = connection.execute(
                "DELETE FROM behavioral_experiences WHERE id = ?", (existing.id,)
            )
            _require_changed(cursor.rowcount, "Experience deletion")
        return existing

    def create_lesson_candidate(
        self,
        *,
        source_experience_ids: tuple[int, ...],
        context_pattern: str,
        proposed_strategy: str,
        rationale: str,
    ) -> BehavioralLessonCandidate:
        source_ids = _validate_source_ids(source_experience_ids)
        with self._database.connect() as connection:
            placeholders = ", ".join("?" for _ in source_ids)
            rows = connection.execute(
                f"""
                {_EXPERIENCE_SELECT}
                WHERE id IN ({placeholders})
                ORDER BY id
                """,
                source_ids,
            ).fetchall()
            experiences = tuple(_experience_from_row(row) for row in rows)
            if tuple(item.id for item in experiences) != tuple(sorted(source_ids)):
                raise BehavioralExperienceNotFoundError(
                    "A source experience does not exist."
                )
            if any(item.status != "active" for item in experiences):
                raise ValueError("Invalidated experiences cannot support a new candidate.")
            person_ids = {item.person_id for item in experiences}
            if len(person_ids) != 1:
                raise ValueError(
                    "Lesson candidate sources must belong to one exact person scope."
                )
            person_id = person_ids.pop()
            now = _normalized_now(self._clock())
            validated = BehavioralLessonCandidate(
                id=1,
                person_id=person_id,
                context_pattern=context_pattern,
                proposed_strategy=proposed_strategy,
                rationale=rationale,
                source_experience_ids=source_ids,
                status="active",
                invalidation_reason=None,
                created_at=now,
                updated_at=now,
            )
            cursor = connection.execute(
                """
                INSERT INTO behavioral_lesson_candidates (
                    person_id, context_pattern, proposed_strategy, rationale,
                    status, invalidation_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.person_id,
                    validated.context_pattern,
                    validated.proposed_strategy,
                    validated.rationale,
                    validated.status,
                    validated.invalidation_reason,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            candidate_id = _created_identifier(cursor.lastrowid, "lesson candidate")
            connection.executemany(
                """
                INSERT INTO behavioral_lesson_sources (candidate_id, experience_id)
                VALUES (?, ?)
                """,
                ((candidate_id, source_id) for source_id in source_ids),
            )
            row = connection.execute(
                f"{_CANDIDATE_SELECT} WHERE id = ?", (candidate_id,)
            ).fetchone()
        return _candidate_from_row(row, source_ids)

    def get_lesson_candidate(
        self,
        candidate_id: int,
    ) -> BehavioralLessonCandidate | None:
        normalized_id = _validate_identifier(candidate_id, "Candidate")
        with self._database.connect() as connection:
            row = connection.execute(
                f"{_CANDIDATE_SELECT} WHERE id = ?", (normalized_id,)
            ).fetchone()
            if row is None:
                return None
            source_ids = _source_ids(connection, normalized_id)
        return _candidate_from_row(row, source_ids)

    def list_person_lesson_candidates(
        self,
        person_id: int,
        *,
        include_invalidated: bool = False,
        limit: int = DEFAULT_LEARNING_RESULT_LIMIT,
    ) -> tuple[BehavioralLessonCandidate, ...]:
        return self._list_candidates(
            person_id=_validate_identifier(person_id, "Person"),
            include_invalidated=include_invalidated,
            limit=limit,
        )

    def list_global_lesson_candidates(
        self,
        *,
        include_invalidated: bool = False,
        limit: int = DEFAULT_LEARNING_RESULT_LIMIT,
    ) -> tuple[BehavioralLessonCandidate, ...]:
        return self._list_candidates(
            person_id=None,
            include_invalidated=include_invalidated,
            limit=limit,
        )

    def _list_candidates(
        self,
        *,
        person_id: int | None,
        include_invalidated: bool,
        limit: int,
    ) -> tuple[BehavioralLessonCandidate, ...]:
        normalized_limit = _validate_limit(limit)
        if not isinstance(include_invalidated, bool):
            raise TypeError("Learning invalidated inclusion flag must be boolean.")
        scope_clause = "person_id IS NULL" if person_id is None else "person_id = ?"
        parameters: list[object] = [] if person_id is None else [person_id]
        status_clause = "" if include_invalidated else " AND status = 'active'"
        parameters.append(normalized_limit)
        with self._database.connect() as connection:
            rows = connection.execute(
                f"""
                {_CANDIDATE_SELECT}
                WHERE {scope_clause}{status_clause}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
            return tuple(
                _candidate_from_row(row, _source_ids(connection, cast(int, row[0])))
                for row in rows
            )

    def update_lesson_candidate(
        self,
        candidate_id: int,
        *,
        context_pattern: str,
        proposed_strategy: str,
        rationale: str,
    ) -> BehavioralLessonCandidate:
        existing = self._require_candidate(candidate_id)
        now = _normalized_now(self._clock())
        validated = BehavioralLessonCandidate(
            id=existing.id,
            person_id=existing.person_id,
            context_pattern=context_pattern,
            proposed_strategy=proposed_strategy,
            rationale=rationale,
            source_experience_ids=existing.source_experience_ids,
            status=existing.status,
            invalidation_reason=existing.invalidation_reason,
            created_at=existing.created_at,
            updated_at=now,
        )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE behavioral_lesson_candidates
                SET context_pattern = ?, proposed_strategy = ?, rationale = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    validated.context_pattern,
                    validated.proposed_strategy,
                    validated.rationale,
                    now.isoformat(),
                    existing.id,
                ),
            )
            _require_changed(cursor.rowcount, "Lesson candidate update")
            row = connection.execute(
                f"{_CANDIDATE_SELECT} WHERE id = ?", (existing.id,)
            ).fetchone()
        return _candidate_from_row(row, existing.source_experience_ids)

    def invalidate_lesson_candidate(
        self,
        candidate_id: int,
        reason: str,
    ) -> BehavioralLessonCandidate:
        existing = self._require_candidate(candidate_id)
        if existing.status == "invalidated":
            raise ValueError("Lesson candidate is already invalidated.")
        reason = _normalize_text(reason, "Candidate invalidation reason")
        now = _normalized_now(self._clock())
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE behavioral_lesson_candidates
                SET status = 'invalidated', invalidation_reason = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (reason, now.isoformat(), existing.id),
            )
            _require_changed(cursor.rowcount, "Lesson candidate invalidation")
            row = connection.execute(
                f"{_CANDIDATE_SELECT} WHERE id = ?", (existing.id,)
            ).fetchone()
        return _candidate_from_row(row, existing.source_experience_ids)

    def delete_lesson_candidate(
        self,
        candidate_id: int,
    ) -> BehavioralLessonCandidate:
        existing = self._require_candidate(candidate_id)
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM behavioral_lesson_candidates WHERE id = ?",
                (existing.id,),
            )
            _require_changed(cursor.rowcount, "Lesson candidate deletion")
        return existing

    def _require_experience(self, experience_id: int) -> BehavioralExperience:
        experience = self.get_experience(experience_id)
        if experience is None:
            raise BehavioralExperienceNotFoundError(
                f"Behavioral experience {experience_id} does not exist."
            )
        return experience

    def _require_candidate(
        self,
        candidate_id: int,
    ) -> BehavioralLessonCandidate:
        candidate = self.get_lesson_candidate(candidate_id)
        if candidate is None:
            raise BehavioralLessonCandidateNotFoundError(
                f"Behavioral lesson candidate {candidate_id} does not exist."
            )
        return candidate


def _experience_from_row(row: tuple[object, ...] | None) -> BehavioralExperience:
    if row is None or len(row) != 20:
        raise RepositoryError("Stored behavioral experience data is incomplete.")
    try:
        return BehavioralExperience(
            id=cast(int, row[0]),
            person_id=cast(int | None, row[1]),
            context=cast(str, row[2]),
            objective=cast(str, row[3]),
            strategy=cast(str, row[4]),
            result=cast(str, row[5]),
            evaluation=cast(str | None, row[6]),
            provenance=ExperienceProvenance(
                source_type=cast(ExperienceSourceType, row[7]),
                source_reference=cast(str | None, row[8]),
                source_text=cast(str | None, row[9]),
            ),
            status=cast(LearningRecordStatus, row[10]),
            invalidation_reason=cast(str | None, row[11]),
            created_at=_parse_datetime(row[12]),
            updated_at=_parse_datetime(row[13]),
            outcome_status=cast(OutcomeStatus, row[14]),
            outcome_kind=cast(OutcomeKind, row[15]),
            result_code=cast(str | None, row[16]),
            feedback_kind=cast(UserFeedbackKind | None, row[17]),
            feedback_text=cast(str | None, row[18]),
            measurements=_parse_measurements(row[19]),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError("Stored behavioral experience data is invalid.") from error


def _candidate_from_row(
    row: tuple[object, ...] | None,
    source_ids: tuple[int, ...],
) -> BehavioralLessonCandidate:
    if row is None or len(row) != 9:
        raise RepositoryError("Stored behavioral lesson candidate data is incomplete.")
    try:
        return BehavioralLessonCandidate(
            id=cast(int, row[0]),
            person_id=cast(int | None, row[1]),
            context_pattern=cast(str, row[2]),
            proposed_strategy=cast(str, row[3]),
            rationale=cast(str, row[4]),
            source_experience_ids=source_ids,
            status=cast(LearningRecordStatus, row[5]),
            invalidation_reason=cast(str | None, row[6]),
            created_at=_parse_datetime(row[7]),
            updated_at=_parse_datetime(row[8]),
        )
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            "Stored behavioral lesson candidate data is invalid."
        ) from error


def _source_ids(connection: object, candidate_id: int) -> tuple[int, ...]:
    rows = connection.execute(
        """
        SELECT experience_id FROM behavioral_lesson_sources
        WHERE candidate_id = ? ORDER BY experience_id
        """,
        (candidate_id,),
    ).fetchall()
    return tuple(cast(int, row[0]) for row in rows)


def _validate_source_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Source experience identifiers must be a tuple.")
    if not value:
        raise ValueError("A lesson candidate requires at least one source experience.")
    normalized = tuple(_validate_identifier(item, "Experience") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Source experience identifiers must be unique.")
    return tuple(sorted(normalized))


def _validate_identifier(value: object, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} identifier must be an integer.")
    if value < 1:
        raise ValueError(f"{subject} identifier must be greater than zero.")
    return value


def _validate_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Learning result limit must be an integer.")
    if value < 1 or value > MAX_LEARNING_RESULT_LIMIT:
        raise ValueError("Learning result limit must be between 1 and 500.")
    return value


def _normalize_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _created_identifier(value: object, subject: str) -> int:
    try:
        return _validate_identifier(value, subject.capitalize())
    except (TypeError, ValueError) as error:
        raise RepositoryError(f"Created {subject} identifier is invalid.") from error


def _require_changed(rowcount: int, operation: str) -> None:
    if rowcount != 1:
        raise RepositoryError(f"{operation} could not be persisted.")


def _normalized_now(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Learning clock must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Learning clock must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Stored learning timestamp must be text.")
    return datetime.fromisoformat(value)


def _serialize_measurements(
    measurements: tuple[OutcomeMeasurement, ...],
) -> str:
    return json.dumps(
        [
            {"name": item.name, "value": item.value, "unit": item.unit}
            for item in measurements
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_measurements(value: object) -> tuple[OutcomeMeasurement, ...]:
    if not isinstance(value, str):
        raise TypeError("Stored outcome measurements must be text.")
    try:
        items = json.loads(value)
    except JSONDecodeError as error:
        raise ValueError("Stored outcome measurements are not valid JSON.") from error
    if not isinstance(items, list):
        raise TypeError("Stored outcome measurements must be a JSON list.")
    measurements: list[OutcomeMeasurement] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "value", "unit"}:
            raise ValueError("Stored outcome measurement data is invalid.")
        measurements.append(
            OutcomeMeasurement(
                name=item["name"],
                value=item["value"],
                unit=item["unit"],
            )
        )
    return tuple(measurements)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
