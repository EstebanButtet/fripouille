"""Tests for internal conversation-generation directives."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.allocation import AllocationTarget
from assistant_ia.intelligence.conversation import (
    ConversationDirective,
    ConversationDirectiveProposal,
    ConversationMode,
)


class ConversationDirectiveTests(unittest.TestCase):
    """Validate conversation-only internal routing metadata."""

    def test_standard_directive_has_no_allocation_target(
        self,
    ) -> None:
        directive = ConversationDirective.standard()

        self.assertEqual(
            directive.mode,
            ConversationMode.STANDARD,
        )
        self.assertIsNone(
            directive.allocation_target
        )

    def test_fixed_total_directive_requires_target(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        directive = ConversationDirective.fixed_total_allocation(
            target
        )

        self.assertEqual(
            directive.mode,
            ConversationMode.FIXED_TOTAL_ALLOCATION,
        )
        self.assertIs(
            directive.allocation_target,
            target,
        )

    def test_directive_rejects_target_for_standard_mode(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        with self.assertRaises(ValueError):
            ConversationDirective(
                mode=ConversationMode.STANDARD,
                allocation_target=target,
            )

    def test_fixed_total_mode_rejects_missing_target(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            ConversationDirective(
                mode=ConversationMode.FIXED_TOTAL_ALLOCATION,
                allocation_target=None,
            )

    def test_directive_mode_must_be_enum(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            ConversationDirective(
                mode="standard",
                allocation_target=None,
            )

    def test_standard_proposal_has_no_target_text(
        self,
    ) -> None:
        proposal = ConversationDirectiveProposal.standard()

        self.assertEqual(
            proposal.mode,
            ConversationMode.STANDARD,
        )
        self.assertIsNone(
            proposal.target_text
        )

    def test_fixed_total_proposal_keeps_user_text_evidence(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        self.assertEqual(
            proposal.mode,
            ConversationMode.FIXED_TOTAL_ALLOCATION,
        )
        self.assertEqual(
            proposal.target_text,
            "trois heures",
        )

    def test_fixed_total_proposal_requires_target_text(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            ConversationDirectiveProposal(
                mode=ConversationMode.FIXED_TOTAL_ALLOCATION,
                target_text=None,
            )

    def test_standard_proposal_rejects_target_text(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            ConversationDirectiveProposal(
                mode=ConversationMode.STANDARD,
                target_text="trois heures",
            )

    def test_proposal_normalizes_target_text_whitespace(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "  trois heures  "
            )
        )

        self.assertEqual(
            proposal.target_text,
            "trois heures",
        )


if __name__ == "__main__":
    unittest.main()
