"""Rendu déterministe de l'identité stable dans le contexte du modèle.

Le module reçoit un :class:`AssistantIdentity` déjà validé et produit une
section textuelle de prompt. Il ne laisse pas Ollama modifier l'objet source et
n'enregistre aucune préférence nouvelle.
"""

from __future__ import annotations

from assistant_ia.identity.models import AssistantIdentity


def render_identity_context(
    identity: AssistantIdentity,
) -> str:
    """Rendre une identité stable sous forme de contexte déterministe.

    L'ordre explicite des sections rend le prompt reproductible et maintient
    séparés traits, style, niveaux, règles et limites personnelles.
    """
    if not isinstance(identity, AssistantIdentity):
        raise TypeError(
            "Identity context requires an AssistantIdentity."
        )

    lines = [
        "Assistant identity",
        f"Name: {identity.name}",
        (
            "Grammatical gender: "
            f"{identity.grammatical_gender}"
        ),
        f"Role: {identity.role}",
        (
            "Relationship to user: "
            f"{identity.relationship_to_user}"
        ),
        "",
        "Traits:",
        *(
            f"- {trait}"
            for trait in identity.traits
        ),
        "",
        "Communication style:",
        *(
            f"- {rule}"
            for rule in identity.communication_style
        ),
        "",
        "Behavior levels:",
        f"- Humor: {identity.humor_level}",
        f"- Initiative: {identity.initiative_level}",
        f"- Curiosity: {identity.curiosity_level}",
        "",
        "Behavioral rules:",
        *(
            f"- {rule}"
            for rule in identity.behavioral_rules
        ),
        "",
        "Personal boundaries:",
        *(
            f"- {boundary}"
            for boundary in identity.boundaries
        ),
    ]

    return "\n".join(lines)
