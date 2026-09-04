"""Frontière applicative de l'apprentissage comportemental explicite."""

from __future__ import annotations

from assistant_ia.learning.models import (
    BehavioralExperience,
    BehavioralLessonCandidate,
    ExperienceProvenance,
)
from assistant_ia.learning.repository import BehavioralLearningRepository
from assistant_ia.people.context import ActivePersonContext


class BehavioralLearningScopeError(RuntimeError):
    """Signaler qu'une opération n'appartient pas à la portée active."""


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
