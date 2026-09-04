"""Sélection bornée du contexte social confirmé et incertain d'une personne."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from assistant_ia.people.observation_repository import ObservationRepository
from assistant_ia.people.profile_fact_repository import ProfileFactRepository
from assistant_ia.people.profile_models import ProfileFact
from assistant_ia.people.relationship_repository import (
    PersonRelationshipRepository,
)
from assistant_ia.people.social_models import Observation, PersonRelationship

MAX_CONTEXTUAL_PROFILE_FACTS = 8
MAX_CONTEXTUAL_PROFILE_FACT_CHARACTERS = 300
MAX_CONTEXTUAL_PROFILE_TOTAL_CHARACTERS = 1200
MAX_CONTEXTUAL_RELATIONSHIP_CHARACTERS = 100
MAX_CONTEXTUAL_OBSERVATIONS = 3
MAX_CONTEXTUAL_OBSERVATION_CHARACTERS = 300
MAX_CONTEXTUAL_OBSERVATION_TOTAL_CHARACTERS = 600
SOCIAL_CONTEXT_CANDIDATE_LIMIT = 50

_ContextItem = TypeVar("_ContextItem", ProfileFact, Observation)


@dataclass(frozen=True, slots=True)
class SocialContext:
    """Porter des données sociales déjà filtrées pour une seule personne."""

    profile_facts: tuple[ProfileFact, ...] = ()
    relationship: PersonRelationship | None = None
    observations: tuple[Observation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile_facts, tuple) or not all(
            isinstance(item, ProfileFact) for item in self.profile_facts
        ):
            raise TypeError("Social profile facts must be a tuple of ProfileFact.")
        if self.relationship is not None and not isinstance(
            self.relationship, PersonRelationship
        ):
            raise TypeError("Social relationship must be a PersonRelationship.")
        if not isinstance(self.observations, tuple) or not all(
            isinstance(item, Observation) for item in self.observations
        ):
            raise TypeError("Social observations must be a tuple of Observation.")


class PersonSocialContextProvider:
    """Lire et borner le contexte social de l'identifiant fourni par l'application."""

    def __init__(
        self,
        profile_repository: ProfileFactRepository,
        relationship_repository: PersonRelationshipRepository,
        observation_repository: ObservationRepository,
    ) -> None:
        if not isinstance(profile_repository, ProfileFactRepository):
            raise TypeError("Social context requires a ProfileFactRepository.")
        if not isinstance(relationship_repository, PersonRelationshipRepository):
            raise TypeError(
                "Social context requires a PersonRelationshipRepository."
            )
        if not isinstance(observation_repository, ObservationRepository):
            raise TypeError("Social context requires an ObservationRepository.")
        self._profile_repository = profile_repository
        self._relationship_repository = relationship_repository
        self._observation_repository = observation_repository

    def build(self, person_id: int) -> SocialContext:
        """Retourner uniquement les données bornées de la personne demandée."""
        normalized_person_id = _validate_identifier(person_id)
        profile_facts = self._profile_repository.list_profile_facts(
            normalized_person_id,
            limit=SOCIAL_CONTEXT_CANDIDATE_LIMIT,
        )
        observations = self._observation_repository.list_observations(
            normalized_person_id,
            limit=SOCIAL_CONTEXT_CANDIDATE_LIMIT,
        )
        relationship = self._relationship_repository.get_relationship(
            normalized_person_id
        )
        if relationship is not None and (
            len(relationship.familiarity) + len(relationship.interaction_style)
            > MAX_CONTEXTUAL_RELATIONSHIP_CHARACTERS
        ):
            relationship = None
        return SocialContext(
            profile_facts=_bound_content(
                profile_facts,
                max_items=MAX_CONTEXTUAL_PROFILE_FACTS,
                max_item_characters=MAX_CONTEXTUAL_PROFILE_FACT_CHARACTERS,
                max_total_characters=MAX_CONTEXTUAL_PROFILE_TOTAL_CHARACTERS,
            ),
            relationship=relationship,
            observations=_bound_content(
                observations,
                max_items=MAX_CONTEXTUAL_OBSERVATIONS,
                max_item_characters=MAX_CONTEXTUAL_OBSERVATION_CHARACTERS,
                max_total_characters=MAX_CONTEXTUAL_OBSERVATION_TOTAL_CHARACTERS,
            ),
        )


def _bound_content(
    items: tuple[_ContextItem, ...],
    *,
    max_items: int,
    max_item_characters: int,
    max_total_characters: int,
) -> tuple[_ContextItem, ...]:
    selected: list[_ContextItem] = []
    total = 0
    for item in items:
        content_length = len(item.content)
        if content_length > max_item_characters:
            continue
        if total + content_length > max_total_characters:
            continue
        selected.append(item)
        total += content_length
        if len(selected) == max_items:
            break
    return tuple(selected)


def _validate_identifier(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Social context person identifier must be an integer.")
    if value < 1:
        raise ValueError("Social context person identifier must be positive.")
    return value
