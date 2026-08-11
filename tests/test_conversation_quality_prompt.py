"""Tests for general conversational reasoning quality rules."""

import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.prompt import build_conversation_prompt


class ConversationQualityPromptTests(unittest.TestCase):
    """Validate general reasoning safeguards in conversation generation."""

    def test_prompt_does_not_invent_user_motives(
        self,
    ) -> None:
        """The model should not invent psychological explanations."""
        prompt = build_conversation_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Do not invent, speculate about or assign motives",
            prompt,
        )
        self.assertIn(
            "unless the conversation provides evidence for them",
            prompt,
        )

    def test_prompt_requires_exact_resource_allocation(
        self,
    ) -> None:
        """Fixed quantities should remain arithmetically consistent."""
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        ).casefold()

        self.assertIn(
            "verify that every proposed part sums exactly "
            "to the stated total",
            normalized_prompt,
        )
        self.assertIn(
            "if your arithmetic and your written allocation disagree",
            normalized_prompt,
        )

    def test_fixed_totals_require_common_unit_verification(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        ).casefold()

        self.assertIn(
            "convert every proposed part to one common unit",
            normalized_prompt,
        )
        self.assertIn(
            "including breaks, reserves and unallocated portions",
            normalized_prompt,
        )

    def test_equivalent_quantity_labels_must_agree(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        )

        self.assertIn(
            "If the same quantity is expressed in multiple units",
            normalized_prompt,
        )
        self.assertIn(
            "the representations must be mathematically equivalent",
            normalized_prompt,
        )

    def test_fixed_allocation_uses_one_canonical_plan(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        )

        self.assertIn(
            "establish one canonical allocation before writing the answer",
            normalized_prompt,
        )
        self.assertIn(
            "Use those same quantities throughout the entire response",
            normalized_prompt,
        )

    def test_allocation_summaries_cannot_change_numbers(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        ).casefold()

        self.assertIn(
            "headings, summaries, explanations and detailed schedules",
            normalized_prompt,
        )
        self.assertIn(
            "must all describe the same allocation",
            normalized_prompt,
        )

    def test_allocation_hides_unverified_numeric_drafts(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        ).casefold()

        self.assertIn(
            "do not present preliminary numerical allocations",
            normalized_prompt,
        )
        self.assertIn(
            "silently correct any failed calculation before writing",
            normalized_prompt,
        )

    def test_verified_allocation_is_presented_only_once(
        self,
    ) -> None:
        prompt = build_conversation_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        ).casefold()

        self.assertIn(
            "present only the verified canonical allocation",
            normalized_prompt,
        )
        self.assertIn(
            "do not narrate recalculations or self-corrections",
            normalized_prompt,
        )


if __name__ == "__main__":
    unittest.main()
