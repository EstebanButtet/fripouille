"""Rendering of structured assistant identity context."""

from __future__ import annotations

from assistant_ia.identity.models import AssistantIdentity


def render_identity_context(
    identity: AssistantIdentity,
) -> str:
    """Render one stable identity as deterministic model context."""
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
