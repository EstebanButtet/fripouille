"""Tests for structured assistant identity models."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from assistant_ia.identity.models import AssistantIdentity


def build_identity(**overrides: object) -> AssistantIdentity:
    """Build one valid identity with optional test overrides."""
    values: dict[str, object] = {
        "name": "Assistant",
        "role": "Personal assistant",
        "relationship_to_user": "Trusted personal assistant",
        "grammatical_gender": "masculine",
        "traits": (
            "Curious",
            "Reliable",
        ),
        "communication_style": (
            "Clear",
            "Natural",
        ),
        "humor_level": "moderate",
        "initiative_level": "moderate",
        "curiosity_level": "high",
        "behavioral_rules": (
            "Be truthful",
            "Stay relevant",
        ),
        "boundaries": (
            "Do not pretend to know unknown information",
        ),
    }
    values.update(overrides)

    return AssistantIdentity(**values)


class AssistantIdentityTests(unittest.TestCase):
    """Validate stable assistant identity configuration."""

    def test_normalizes_identity_values(self) -> None:
        """Identity text and levels should be normalized."""
        identity = build_identity(
            name=" Assistant ",
            role=" Personal assistant ",
            relationship_to_user=" Trusted personal assistant ",
            grammatical_gender=" masculine ",
            traits=(
                " Curious ",
                " Reliable ",
            ),
            communication_style=(
                " Clear ",
                " Natural ",
            ),
            humor_level=" moderate ",
            initiative_level=" high ",
            curiosity_level=" low ",
            behavioral_rules=(
                " Be truthful ",
                " Stay relevant ",
            ),
            boundaries=(
                " Do not pretend to know unknown information ",
            ),
        )

        self.assertEqual(identity.name, "Assistant")
        self.assertEqual(identity.role, "Personal assistant")
        self.assertEqual(
            identity.relationship_to_user,
            "Trusted personal assistant",
        )
        self.assertEqual(
            identity.grammatical_gender,
            "masculine",
        )
        self.assertEqual(
            identity.traits,
            (
                "Curious",
                "Reliable",
            ),
        )
        self.assertEqual(identity.humor_level, "moderate")
        self.assertEqual(identity.initiative_level, "high")
        self.assertEqual(identity.curiosity_level, "low")

    def test_identity_is_immutable(self) -> None:
        """Identity configuration should not change after creation."""
        identity = build_identity()

        with self.assertRaises(FrozenInstanceError):
            identity.name = "Other"

        with self.assertRaises(TypeError):
            identity.traits[0] = "Other"

    def test_rejects_empty_required_text(self) -> None:
        """Required identity text should never be empty."""
        with self.assertRaisesRegex(
            ValueError,
            "name cannot be empty",
        ):
            build_identity(name="   ")

    def test_rejects_non_string_required_text(self) -> None:
        """Required identity text should always be strings."""
        with self.assertRaisesRegex(
            TypeError,
            "role must be a string",
        ):
            build_identity(role=42)

    def test_rejects_unknown_grammatical_gender(self) -> None:
        """Grammatical gender should use only explicit categories."""
        with self.assertRaisesRegex(
            ValueError,
            "Unknown assistant identity grammatical_gender",
        ):
            build_identity(
                grammatical_gender="unknown"
            )

    def test_rejects_unknown_behavior_level(self) -> None:
        """Behavior levels should use only explicit categories."""
        with self.assertRaisesRegex(
            ValueError,
            "Unknown assistant identity humor_level",
        ):
            build_identity(humor_level="extreme")

    def test_rejects_non_string_behavior_level(self) -> None:
        """Behavior levels should never accept numeric scores."""
        with self.assertRaisesRegex(
            TypeError,
            "initiative_level must be a string",
        ):
            build_identity(initiative_level=0.8)

    def test_rejects_mutable_identity_collection(self) -> None:
        """Identity collections should be immutable tuples."""
        with self.assertRaisesRegex(
            TypeError,
            "traits must be a tuple",
        ):
            build_identity(
                traits=[
                    "Curious",
                    "Reliable",
                ]
            )

    def test_rejects_empty_identity_collection(self) -> None:
        """Core identity collections should contain useful information."""
        with self.assertRaisesRegex(
            ValueError,
            "behavioral_rules cannot be empty",
        ):
            build_identity(behavioral_rules=())

    def test_rejects_invalid_identity_collection_item(self) -> None:
        """Identity collection items should be non-empty strings."""
        with self.assertRaisesRegex(
            ValueError,
            "communication_style items cannot be empty",
        ):
            build_identity(
                communication_style=(
                    "Clear",
                    "   ",
                )
            )


if __name__ == "__main__":
    unittest.main()
