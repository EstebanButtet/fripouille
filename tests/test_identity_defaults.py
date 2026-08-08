"""Tests for the default assistant identity."""

from __future__ import annotations

import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.identity.models import AssistantIdentity


class DefaultAssistantIdentityTests(unittest.TestCase):
    """Validate the default Fripouille identity."""

    def test_builds_fripouille_identity(self) -> None:
        """Default identity should describe Fripouille explicitly."""
        identity = build_default_identity()

        self.assertIsInstance(
            identity,
            AssistantIdentity,
        )
        self.assertEqual(
            identity.name,
            "Fripouille",
        )
        self.assertEqual(
            identity.grammatical_gender,
            "masculine",
        )
        self.assertEqual(
            identity.role,
            "Friend, accomplice and personal companion",
        )
        self.assertIn(
            "equal-to-equal",
            identity.relationship_to_user,
        )

    def test_uses_high_behavior_levels(self) -> None:
        """Fripouille should be highly expressive and proactive."""
        identity = build_default_identity()

        self.assertEqual(
            identity.humor_level,
            "high",
        )
        self.assertEqual(
            identity.initiative_level,
            "high",
        )
        self.assertEqual(
            identity.curiosity_level,
            "high",
        )

    def test_contains_core_personality_traits(self) -> None:
        """Default traits should preserve the designed personality."""
        identity = build_default_identity()

        self.assertIn(
            "Mischievous",
            identity.traits,
        )
        self.assertIn(
            "Sincere",
            identity.traits,
        )
        self.assertIn(
            "Irreverent",
            identity.traits,
        )
        self.assertIn(
            "Caring",
            identity.traits,
        )

    def test_preserves_free_adaptive_language_style(self) -> None:
        """Communication style should allow natural register changes."""
        identity = build_default_identity()

        rendered_style = "\n".join(
            identity.communication_style
        )

        self.assertIn(
            "profanity",
            rendered_style,
        )
        self.assertIn(
            "Adapt tone freely",
            rendered_style,
        )
        self.assertIn(
            "seriousness",
            rendered_style,
        )

    def test_preserves_honest_disagreement_and_user_autonomy(
        self,
    ) -> None:
        """Fripouille should warn honestly without becoming paternalistic."""
        identity = build_default_identity()

        rendered_rules = "\n".join(
            identity.behavioral_rules
        )

        self.assertIn(
            "Express disagreement honestly",
            rendered_rules,
        )
        self.assertIn(
            "Respect the user's autonomy",
            rendered_rules,
        )
        self.assertIn(
            "personally present and sincere",
            rendered_rules,
        )

    def test_protects_serious_and_inclusive_behavior(self) -> None:
        """Malice should stop before gratuitous harm or real distress."""
        identity = build_default_identity()

        rendered_boundaries = "\n".join(
            identity.boundaries
        )

        self.assertIn(
            "painful",
            rendered_boundaries,
        )
        self.assertIn(
            "humiliate or exclude",
            rendered_boundaries,
        )

    def test_default_identity_is_deterministic(self) -> None:
        """Repeated construction should produce the same identity."""
        self.assertEqual(
            build_default_identity(),
            build_default_identity(),
        )


if __name__ == "__main__":
    unittest.main()
