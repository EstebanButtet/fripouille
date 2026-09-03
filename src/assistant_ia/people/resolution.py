"""Résolution déterministe d'une présentation vers le registre de personnes.

Le service ne consulte aucun modèle de langage. Il accepte uniquement un
``PersonProfile`` produit par la détection explicite, recherche toutes les
correspondances exactes et exige une confirmation applicative avant toute
création.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from assistant_ia.people.models import Person, PersonProfile
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.security.confirmation import (
    ConfirmationHandler,
    ConfirmationRequest,
    deny_confirmation,
    request_confirmation,
)

PersonResolutionStatus = Literal[
    "existing",
    "created",
    "creation_refused",
    "ambiguous",
    "stale",
]


@dataclass(frozen=True, slots=True)
class PersonCreationProposal:
    """Décrire le nom précis dont la création doit être confirmée."""

    display_name: str

    def __post_init__(self) -> None:
        """Normaliser le nom proposé sans lui attribuer d'identité."""
        if not isinstance(self.display_name, str):
            raise TypeError(
                "Proposed person display name must be a string."
            )

        normalized_display_name = unicodedata.normalize(
            "NFC",
            self.display_name,
        ).strip()

        if not normalized_display_name:
            raise ValueError(
                "Proposed person display name must not be empty."
            )

        object.__setattr__(
            self,
            "display_name",
            normalized_display_name,
        )


@dataclass(frozen=True, slots=True)
class PersonResolution:
    """Rendre inspectable le résultat d'une résolution de présentation."""

    status: PersonResolutionStatus
    person: Person | None = None
    proposal: PersonCreationProposal | None = None
    matching_person_ids: tuple[int, ...] = ()


class PersonResolutionError(RuntimeError):
    """Signaler un échec contrôlé de la confirmation de personne."""


class PersonResolutionService:
    """Résoudre une présentation sans choix arbitraire ni autorité LLM."""

    def __init__(
        self,
        repository: PersonRepository,
        confirmation_handler: ConfirmationHandler | None = None,
    ) -> None:
        """Créer le service avec repository et confirmation injectables."""
        if not isinstance(repository, PersonRepository):
            raise TypeError(
                "Person resolution repository must be a PersonRepository."
            )

        if confirmation_handler is not None and not callable(
            confirmation_handler
        ):
            raise TypeError(
                "Person confirmation handler must be callable."
            )

        self._repository = repository
        self._confirmation_handler = (
            confirmation_handler
            if confirmation_handler is not None
            else deny_confirmation
        )

    def resolve_presentation(
        self,
        presented_person: PersonProfile,
    ) -> PersonResolution:
        """Résoudre un profil explicitement présenté vers une personne."""
        if not isinstance(presented_person, PersonProfile):
            raise TypeError(
                "Presented person must be a PersonProfile."
            )

        matches = self._repository.find_persons_by_display_name(
            presented_person.name
        )

        if len(matches) == 1:
            return PersonResolution(
                status="existing",
                person=matches[0],
            )

        if len(matches) > 1:
            return PersonResolution(
                status="ambiguous",
                matching_person_ids=tuple(
                    person.id for person in matches
                ),
            )

        proposal = PersonCreationProposal(
            display_name=presented_person.name
        )
        request = ConfirmationRequest(
            action_name="create_person",
            description=(
                f"créer la personne « {proposal.display_name} » "
                "et la définir comme interlocuteur actif"
            ),
        )

        try:
            confirmed = request_confirmation(
                self._confirmation_handler,
                request,
            )
        except TypeError as error:
            raise PersonResolutionError(
                "Person creation confirmation failed."
            ) from error

        if not confirmed:
            return PersonResolution(
                status="creation_refused",
                proposal=proposal,
            )

        # La confirmation est synchrone et liée à ce nom exact. Une seconde
        # lecture empêche qu'une décision devenue obsolète crée ou sélectionne
        # silencieusement une identité apparue pendant la confirmation.
        refreshed_matches = self._repository.find_persons_by_display_name(
            proposal.display_name
        )

        if refreshed_matches:
            return PersonResolution(
                status="stale",
                proposal=proposal,
                matching_person_ids=tuple(
                    person.id for person in refreshed_matches
                ),
            )

        created_person = self._repository.create_person(
            proposal.display_name
        )

        return PersonResolution(
            status="created",
            person=created_person,
            proposal=proposal,
        )
