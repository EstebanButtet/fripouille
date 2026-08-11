"""Tests for trusted conversation-directive resolution."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.conversation import (
    ConversationDirectiveProposal,
    ConversationDirectiveResolutionError,
    ConversationMode,
    resolve_conversation_directive,
)


class ConversationDirectiveResolutionTests(unittest.TestCase):
    """Validate conversion from untrusted proposals to trusted directives."""

    def test_standard_proposal_resolves_to_standard(
        self,
    ) -> None:
        proposal = ConversationDirectiveProposal.standard()

        directive = resolve_conversation_directive(
            proposal,
            user_message="Explain photosynthesis.",
        )

        self.assertEqual(
            directive.mode,
            ConversationMode.STANDARD,
        )
        self.assertIsNone(
            directive.allocation_target
        )

    def test_exact_duration_evidence_resolves(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        directive = resolve_conversation_directive(
            proposal,
            user_message=(
                "J'ai exactement trois heures ce soir. "
                "Repartis-les entre mes deux examens."
            ),
        )

        self.assertEqual(
            directive.mode,
            ConversationMode.FIXED_TOTAL_ALLOCATION,
        )
        self.assertIsNotNone(
            directive.allocation_target
        )
        self.assertEqual(
            directive.allocation_target.total,
            Decimal("180"),
        )
        self.assertEqual(
            directive.allocation_target.unit,
            "minutes",
        )

    def test_evidence_matching_normalizes_case_and_whitespace(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        directive = resolve_conversation_directive(
            proposal,
            user_message=(
                "J'ai TROIS   HEURES pour travailler."
            ),
        )

        self.assertEqual(
            directive.allocation_target.total,
            Decimal("180"),
        )

    def test_hallucinated_target_text_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "quatre heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai exactement trois heures ce soir."
                ),
            )

    def test_approximate_duration_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai environ trois heures ce soir."
                ),
            )

    def test_approximate_multiword_prefix_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai \u00e0 peu pr\u00e8s trois heures."
                ),
            )

    def test_hyphenated_range_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "3 heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai 2-3 heures pour travailler."
                ),
            )

    def test_between_range_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "3 heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai entre 2 et 3 heures pour travailler."
                ),
            )

    def test_unsupported_target_text_is_rejected(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "quelques heures"
            )
        )

        with self.assertRaises(
            ConversationDirectiveResolutionError
        ):
            resolve_conversation_directive(
                proposal,
                user_message=(
                    "J'ai quelques heures devant moi."
                ),
            )

    def test_proposal_must_have_correct_type(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            resolve_conversation_directive(
                "fixed_total_allocation",
                user_message="J'ai trois heures.",
            )

    def test_user_message_must_be_string(
        self,
    ) -> None:
        proposal = ConversationDirectiveProposal.standard()

        with self.assertRaises(TypeError):
            resolve_conversation_directive(
                proposal,
                user_message=None,
            )


if __name__ == "__main__":
    unittest.main()
