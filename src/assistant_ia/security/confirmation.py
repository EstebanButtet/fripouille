"""Frontière de confirmation explicite par l'utilisateur.

Une action sensible décrit sa demande dans :class:`ConfirmationRequest`, puis
appelle un handler injecté par l'interface. La sécurité ne dépend ainsi ni de
``input`` ni de tkinter ; elle exige seulement une réponse booléenne contrôlée.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """Décrire précisément l'action soumise à confirmation.

    ``action_name`` sert à identifier le type d'effet ; ``description`` est la
    formulation concrète présentée à la personne, par exemple l'application à
    lancer. L'objet est immuable après normalisation.
    """

    action_name: str
    description: str

    def __post_init__(self) -> None:
        """Valider et normaliser les deux textes de la demande."""
        if not isinstance(self.action_name, str):
            raise TypeError(
                "Confirmation action name must be a string."
            )

        if not isinstance(self.description, str):
            raise TypeError(
                "Confirmation description must be a string."
            )

        normalized_action_name = self.action_name.strip()
        normalized_description = self.description.strip()

        if not normalized_action_name:
            raise ValueError(
                "Confirmation action name cannot be empty."
            )

        if not normalized_description:
            raise ValueError(
                "Confirmation description cannot be empty."
            )

        object.__setattr__(
            self,
            "action_name",
            normalized_action_name,
        )
        object.__setattr__(
            self,
            "description",
            normalized_description,
        )


ConfirmationHandler = Callable[[ConfirmationRequest], bool]


def request_confirmation(
    handler: ConfirmationHandler,
    request: ConfirmationRequest,
) -> bool:
    """Demander une confirmation via le handler fourni par l'interface.

    Un résultat autre qu'un booléen est refusé : une chaîne non vide ne doit
    pas devenir implicitement une autorisation Python.
    """
    if not callable(handler):
        raise TypeError(
            "Confirmation handler must be callable."
        )

    if not isinstance(request, ConfirmationRequest):
        raise TypeError(
            "Confirmation requires a ConfirmationRequest."
        )

    result = handler(request)

    if not isinstance(result, bool):
        raise TypeError(
            "Confirmation handler must return a boolean."
        )

    return result


def deny_confirmation(
    request: ConfirmationRequest,
) -> bool:
    """Refuser par défaut lorsqu'aucune interface interactive n'est disponible."""
    if not isinstance(request, ConfirmationRequest):
        raise TypeError(
            "Confirmation requires a ConfirmationRequest."
        )

    return False
