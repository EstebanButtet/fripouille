"""Modèle de réponse structurée remis par le client de langage au coeur.

Le texte, le modèle utilisé et l'intention appartiennent au même résultat
validé. Cette structure ne signifie pas que le texte du LLM a autorité : le
coeur choisit encore entre conversation et exécution contrôlée d'une action.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.intelligence.intent import Intent


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Représenter une réponse du modèle normalisée mais non exécutée.

    ``content`` est le texte conversationnel ou un marqueur interne pour une
    action ; ``model`` sert au diagnostic ; ``intent`` dirige l'orchestration.
    La dataclass immuable garantit leur cohérence après validation.
    """

    content: str
    model: str
    intent: Intent

    def __post_init__(self) -> None:
        """Valider les types et retirer les espaces extérieurs des textes."""
        if not isinstance(self.content, str):
            raise TypeError("Model response content must be a string.")

        if not isinstance(self.model, str):
            raise TypeError("Model name must be a string.")

        if not isinstance(self.intent, Intent):
            raise TypeError("Model response intent must be an Intent.")

        normalized_content = self.content.strip()
        normalized_model = self.model.strip()

        if not normalized_content:
            raise ValueError("Model response content cannot be empty.")

        if not normalized_model:
            raise ValueError("Model name cannot be empty.")

        object.__setattr__(self, "content", normalized_content)
        object.__setattr__(self, "model", normalized_model)
