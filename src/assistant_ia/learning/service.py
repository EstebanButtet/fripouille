"""Frontière applicative de l'apprentissage comportemental explicite."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from assistant_ia.actions.result import ActionExecutionResult

from assistant_ia.learning.models import (
    BehavioralConsolidation,
    BehavioralExperience,
    BehavioralLessonCandidate,
    ExperienceProvenance,
    ConfirmedBehavioralRule,
)
from assistant_ia.learning.outcomes import (
    BehavioralAttempt,
    ExperienceOutcome,
    OutcomeStatus,
    UserFeedback,
    outcome_from_action_result,
)
from assistant_ia.learning.repository import BehavioralLearningRepository
from assistant_ia.people.context import ActivePersonContext


class BehavioralLearningScopeError(RuntimeError):
    """Signaler qu'une opération n'appartient pas à la portée active."""


class BehavioralOutcomeNotRecordableError(ValueError):
    """Une issue non exécutée ne constitue pas une expérience."""


class BehavioralLearningService:
    """Relier l'apprentissage à une personne résolue par l'application.

    Le service n'analyse aucun texte avec un LLM et n'observe pas
    automatiquement les tours. Ses méthodes constituent un point d'entrée
    explicite pour du code applicatif qui possède déjà un résultat vérifié.
    """

    def __init__(
        self,
        repository: BehavioralLearningRepository,
        person_context: ActivePersonContext,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(repository, BehavioralLearningRepository):
            raise TypeError(
                "Behavioral learning service requires its repository."
            )
        if not isinstance(person_context, ActivePersonContext):
            raise TypeError(
                "Behavioral learning service requires an ActivePersonContext."
            )
        self._repository = repository
        self._person_context = person_context
        if clock is not None and not callable(clock):
            raise TypeError("Behavioral learning service clock must be callable.")
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)

    def begin_active_person_attempt(self, *, context: str, objective: str, strategy: str) -> BehavioralAttempt:
        return BehavioralAttempt(
            person_id=self._require_active_person_id(), context=context,
            objective=objective, strategy=strategy,
            started_at=_normalized_now(self._clock()),
        )

    def begin_global_attempt(self, *, context: str, objective: str, strategy: str) -> BehavioralAttempt:
        return BehavioralAttempt(
            person_id=None, context=context, objective=objective,
            strategy=strategy, started_at=_normalized_now(self._clock()),
        )

    def record_active_person_outcome(self, attempt: BehavioralAttempt, outcome: ExperienceOutcome, *, provenance: ExperienceProvenance, evaluation: str | None = None) -> BehavioralExperience:
        person_id = self._require_active_person_id()
        self._validate_attempt(attempt, person_id)
        self._validate_outcome(outcome)
        return self._repository.create_experience(
            person_id=person_id, context=attempt.context, objective=attempt.objective,
            strategy=attempt.strategy, result=outcome.summary, evaluation=evaluation,
            provenance=provenance, outcome=outcome,
        )

    def record_global_outcome(self, attempt: BehavioralAttempt, outcome: ExperienceOutcome, *, provenance: ExperienceProvenance, evaluation: str | None = None) -> BehavioralExperience:
        self._validate_attempt(attempt, None)
        self._validate_outcome(outcome)
        return self._repository.create_experience(
            person_id=None, context=attempt.context, objective=attempt.objective,
            strategy=attempt.strategy, result=outcome.summary, evaluation=evaluation,
            provenance=provenance, outcome=outcome,
        )

    def record_active_person_action_result(self, attempt: BehavioralAttempt, action_result: ActionExecutionResult, *, source_text: str | None = None, evaluation: str | None = None) -> BehavioralExperience:
        if not isinstance(action_result, ActionExecutionResult):
            raise TypeError("Action result must be ActionExecutionResult.")
        return self.record_active_person_outcome(
            attempt, outcome_from_action_result(action_result),
            provenance=ExperienceProvenance(
                source_type="action_execution",
                source_reference=action_result.action_name,
                source_text=source_text,
            ), evaluation=evaluation,
        )

    def record_active_person_feedback(self, attempt: BehavioralAttempt, feedback: UserFeedback, *, status: OutcomeStatus, summary: str, evaluation: str | None = None) -> BehavioralExperience:
        if not isinstance(feedback, UserFeedback):
            raise TypeError("Feedback must be UserFeedback.")
        return self.record_active_person_outcome(
            attempt,
            ExperienceOutcome(status=status, kind="user_feedback", summary=summary, feedback=feedback),
            provenance=ExperienceProvenance(source_type="conversation_turn", source_text=feedback.content),
            evaluation=evaluation,
        )

    def record_active_person_experience(
        self,
        *,
        context: str,
        objective: str,
        strategy: str,
        result: str,
        provenance: ExperienceProvenance,
        evaluation: str | None = None,
    ) -> BehavioralExperience:
        """Enregistrer pour la personne persistante choisie par l'application."""
        return self._repository.create_experience(
            person_id=self._require_active_person_id(),
            context=context,
            objective=objective,
            strategy=strategy,
            result=result,
            evaluation=evaluation,
            provenance=provenance,
        )

    def record_global_experience(
        self,
        *,
        context: str,
        objective: str,
        strategy: str,
        result: str,
        provenance: ExperienceProvenance,
        evaluation: str | None = None,
    ) -> BehavioralExperience:
        """Enregistrer une expérience explicitement non liée à une personne."""
        return self._repository.create_experience(
            person_id=None,
            context=context,
            objective=objective,
            strategy=strategy,
            result=result,
            evaluation=evaluation,
            provenance=provenance,
        )

    def list_active_person_experiences(
        self,
        *,
        include_invalidated: bool = False,
    ) -> tuple[BehavioralExperience, ...]:
        """Lire uniquement les expériences de la personne active résolue."""
        return self._repository.list_person_experiences(
            self._require_active_person_id(),
            include_invalidated=include_invalidated,
        )

    def list_active_person_lesson_candidates(
        self,
        *,
        include_invalidated: bool = False,
    ) -> tuple[BehavioralLessonCandidate, ...]:
        """Lire uniquement les candidats de la personne active résolue."""
        return self._repository.list_person_lesson_candidates(
            self._require_active_person_id(),
            include_invalidated=include_invalidated,
        )

    def propose_active_person_lesson(
        self,
        *,
        source_experiences: tuple[BehavioralExperience, ...],
        context_pattern: str,
        proposed_strategy: str,
        rationale: str,
    ) -> BehavioralLessonCandidate:
        """Créer un candidat sourcé, jamais une règle ni une promotion."""
        person_id = self._require_active_person_id()
        experiences = _validate_experiences(source_experiences)
        source_ids = self._validated_source_ids(
            experiences,
            expected_person_id=person_id,
        )
        return self._repository.create_lesson_candidate(
            source_experience_ids=source_ids,
            context_pattern=context_pattern,
            proposed_strategy=proposed_strategy,
            rationale=rationale,
        )

    def propose_global_lesson(
        self,
        *,
        source_experiences: tuple[BehavioralExperience, ...],
        context_pattern: str,
        proposed_strategy: str,
        rationale: str,
    ) -> BehavioralLessonCandidate:
        """Créer un candidat depuis des expériences toutes non personnelles."""
        experiences = _validate_experiences(source_experiences)
        source_ids = self._validated_source_ids(
            experiences,
            expected_person_id=None,
        )
        return self._repository.create_lesson_candidate(
            source_experience_ids=source_ids,
            context_pattern=context_pattern,
            proposed_strategy=proposed_strategy,
            rationale=rationale,
        )

    def consolidate_lesson_candidate(
        self, candidate: BehavioralLessonCandidate,
    ) -> BehavioralConsolidation:
        """Recalculer une synthèse explicable, sans modifier la base."""
        if not isinstance(candidate, BehavioralLessonCandidate):
            raise TypeError("Consolidation requires a BehavioralLessonCandidate.")
        stored = self._repository.get_lesson_candidate(candidate.id)
        if stored is None or stored != candidate:
            raise BehavioralLearningScopeError("Consolidation requires a current candidate.")
        experiences = tuple(
            self._repository.get_experience(source_id)
            for source_id in candidate.source_experience_ids
        )
        if any(item is None for item in experiences):
            raise BehavioralLearningScopeError("Consolidation source is missing.")
        active = tuple(item for item in experiences if item is not None and item.status == "active")
        excluded = tuple(item.id for item in experiences if item is not None and item.status != "active")
        seen: dict[tuple[object, ...], int] = {}
        duplicates: list[int] = []
        unique: list[BehavioralExperience] = []
        for item in active:
            key = (
                item.provenance.source_type,
                item.provenance.source_reference,
                item.provenance.source_text,
            ) if item.provenance.source_text is not None else (item.id,)
            if key in seen:
                duplicates.append(item.id)
            else:
                seen[key] = item.id
                unique.append(item)
        candidate_strategy = _comparison_text(candidate.proposed_strategy)
        favorable: list[int] = []
        contradictory: list[int] = []
        ambiguous: list[int] = []
        for item in unique:
            if _comparison_text(item.strategy) != candidate_strategy:
                contradictory.append(item.id)
            elif item.outcome_status in {"success", "partial"}:
                favorable.append(item.id)
            else:
                ambiguous.append(item.id)
        return BehavioralConsolidation(
            candidate_id=candidate.id,
            relevant_experience_ids=tuple(item.id for item in unique),
            favorable_experience_ids=tuple(favorable),
            contradictory_experience_ids=tuple(contradictory),
            ambiguous_experience_ids=tuple(ambiguous),
            excluded_experience_ids=excluded,
            duplicate_experience_ids=tuple(duplicates),
        )

    def confirm_active_person_lesson(
        self, candidate: BehavioralLessonCandidate, confirmation_note: str,
    ) -> ConfirmedBehavioralRule:
        person_id = self._require_active_person_id()
        if candidate.person_id != person_id:
            raise BehavioralLearningScopeError("Candidate does not belong to the active person.")
        summary = self.consolidate_lesson_candidate(candidate)
        if not summary.can_be_confirmed:
            raise ValueError("Candidate needs favorable evidence without contradictions.")
        return self._confirm(candidate, summary, confirmation_note)

    def confirm_global_lesson(
        self, candidate: BehavioralLessonCandidate, confirmation_note: str,
    ) -> ConfirmedBehavioralRule:
        if candidate.person_id is not None:
            raise BehavioralLearningScopeError("Candidate is not global.")
        summary = self.consolidate_lesson_candidate(candidate)
        if not summary.can_be_confirmed:
            raise ValueError("Candidate needs favorable evidence without contradictions.")
        return self._confirm(candidate, summary, confirmation_note)

    def _confirm(self, candidate: BehavioralLessonCandidate, summary: BehavioralConsolidation, note: str) -> ConfirmedBehavioralRule:
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Rule confirmation note cannot be empty.")
        return self._repository.create_confirmed_rule(
            person_id=candidate.person_id,
            context_pattern=candidate.context_pattern,
            proposed_strategy=candidate.proposed_strategy,
            rationale=f"{candidate.rationale} Confirmation: {note.strip()}",
            source_experience_ids=summary.relevant_experience_ids,
        )

    @property
    def repository(self) -> BehavioralLearningRepository:
        """Exposer le repository aux outils applicatifs d'inspection/correction."""
        return self._repository

    def _require_active_person_id(self) -> int:
        person_id = self._person_context.active_person_id
        if person_id is None:
            raise BehavioralLearningScopeError(
                "Behavioral learning requires a resolved persistent person."
            )
        return person_id

    @staticmethod
    def _validate_attempt(attempt: BehavioralAttempt, expected_person_id: int | None) -> None:
        if not isinstance(attempt, BehavioralAttempt):
            raise TypeError("Learning attempt must be BehavioralAttempt.")
        if attempt.person_id != expected_person_id:
            raise BehavioralLearningScopeError(
                "Learning attempt does not belong to the requested person scope."
            )

    @staticmethod
    def _validate_outcome(outcome: ExperienceOutcome) -> None:
        if not isinstance(outcome, ExperienceOutcome):
            raise TypeError("Learning outcome must be ExperienceOutcome.")
        if outcome.status == "not_executed":
            raise BehavioralOutcomeNotRecordableError(
                "An unexecuted attempt is not a behavioral experience."
            )

    def _validated_source_ids(
        self,
        experiences: tuple[BehavioralExperience, ...],
        *,
        expected_person_id: int | None,
    ) -> tuple[int, ...]:
        source_ids: list[int] = []
        for experience in experiences:
            stored = self._repository.get_experience(experience.id)
            if stored is None or stored != experience:
                raise BehavioralLearningScopeError(
                    "Lesson sources must be current persisted experiences."
                )
            if stored.person_id != expected_person_id:
                raise BehavioralLearningScopeError(
                    "Lesson sources do not belong to the requested person scope."
                )
            source_ids.append(stored.id)
        return tuple(source_ids)


def _validate_experiences(
    value: object,
) -> tuple[BehavioralExperience, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Lesson source experiences must be a tuple.")
    if not value:
        raise ValueError("A lesson candidate requires at least one experience.")
    if any(not isinstance(item, BehavioralExperience) for item in value):
        raise TypeError(
            "Lesson sources must be BehavioralExperience instances."
        )
    return value


def _normalized_now(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Learning service clock must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Learning service clock must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def _comparison_text(value: str) -> str:
    return " ".join(value.casefold().split())
