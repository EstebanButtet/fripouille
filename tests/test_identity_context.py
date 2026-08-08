"""Tests for assistant identity context rendering."""

from __future__ import annotations

import unittest

from assistant_ia.identity.context import render_identity_context
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.identity.models import AssistantIdentity


def build_test_identity() -> AssistantIdentity:
    """Build a small deterministic identity for rendering tests."""
    return AssistantIdentity(
        name="Test",
        role="Test companion",
        relationship_to_user="Equal relationship",
        grammatical_gender="neutral",
        traits=(
            "Curious",
            "Sincere",
        ),
        communication_style=(
            "Speak naturally",
            "Adapt to context",
        ),
        humor_level="moderate",
        initiative_level="high",
        curiosity_level="low",
        behavioral_rules=(
            "Be truthful",
            "Respect autonomy",
        ),
        boundaries=(
            "Do not pretend to know",
        ),
    )


class IdentityContextTests(unittest.TestCase):
    """Validate deterministic identity context rendering."""

    def test_renders_complete_identity_context(self) -> None:
        """Rendering should preserve all structured identity fields."""
        rendered = render_identity_context(
            build_test_identity()
        )

        self.assertEqual(
            rendered,
            "\n".join(
                (
                    "Assistant identity",
                    "Name: Test",
                    "Grammatical gender: neutral",
                    "Role: Test companion",
                    "Relationship to user: Equal relationship",
                    "",
                    "Traits:",
                    "- Curious",
                    "- Sincere",
                    "",
                    "Communication style:",
                    "- Speak naturally",
                    "- Adapt to context",
                    "",
                    "Behavior levels:",
                    "- Humor: moderate",
                    "- Initiative: high",
                    "- Curiosity: low",
                    "",
                    "Behavioral rules:",
                    "- Be truthful",
                    "- Respect autonomy",
                    "",
                    "Personal boundaries:",
                    "- Do not pretend to know",
                )
            ),
        )

    def test_rendering_is_deterministic(self) -> None:
        """The same identity should always produce the same context."""
        identity = build_test_identity()

        self.assertEqual(
            render_identity_context(identity),
            render_identity_context(identity),
        )

    def test_default_context_identifies_fripouille(self) -> None:
        """Default rendering should expose Fripouille explicitly."""
        rendered = render_identity_context(
            build_default_identity()
        )

        self.assertIn(
            "Name: Fripouille",
            rendered,
        )
        self.assertIn(
            "Grammatical gender: masculine",
            rendered,
        )
        self.assertIn(
            "- Humor: high",
            rendered,
        )
        self.assertIn(
            "- Initiative: high",
            rendered,
        )
        self.assertIn(
            "- Curiosity: high",
            rendered,
        )

    def test_default_context_contains_core_behavior(self) -> None:
        """Rendered defaults should preserve designed behavior."""
        rendered = render_identity_context(
            build_default_identity()
        )

        self.assertIn(
            "Express disagreement honestly",
            rendered,
        )
        self.assertIn(
            "Respect the user's autonomy",
            rendered,
        )
        self.assertIn(
            "genuinely painful",
            rendered,
        )

    def test_default_context_remains_compact(self) -> None:
        """Identity rendering should remain a compact context block."""
        rendered = render_identity_context(
            build_default_identity()
        )

        self.assertLess(
            len(rendered.splitlines()),
            50,
        )

    def test_rejects_non_identity_object(self) -> None:
        """Rendering should require the identity domain model."""
        with self.assertRaisesRegex(
            TypeError,
            "requires an AssistantIdentity",
        ):
            render_identity_context(
                "Fripouille"
            )


if __name__ == "__main__":
    unittest.main()
