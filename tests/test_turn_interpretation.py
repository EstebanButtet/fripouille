"""Tests for validated internal turn interpretation results."""

import unittest

from assistant_ia.intelligence.conversation import (
    ConversationDirectiveProposal,
    ConversationMode,
)
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.interpretation import TurnInterpretation


class TurnInterpretationTests(unittest.TestCase):
    """Validate the internal result of one interpretation call."""

    def test_standard_conversation_interpretation(
        self,
    ) -> None:
        interpretation = TurnInterpretation(
            intent=Intent(
                name="conversation"
            ),
            conversation_directive_proposal=(
                ConversationDirectiveProposal.standard()
            ),
            model="qwen3.5:9b",
        )

        self.assertEqual(
            interpretation.intent.name,
            "conversation",
        )
        self.assertEqual(
            interpretation.conversation_directive_proposal.mode,
            ConversationMode.STANDARD,
        )
        self.assertEqual(
            interpretation.model,
            "qwen3.5:9b",
        )

    def test_fixed_total_conversation_interpretation(
        self,
    ) -> None:
        proposal = (
            ConversationDirectiveProposal.fixed_total_allocation(
                "trois heures"
            )
        )

        interpretation = TurnInterpretation(
            intent=Intent(
                name="conversation"
            ),
            conversation_directive_proposal=proposal,
            model="qwen3.5:9b",
        )

        self.assertIs(
            interpretation.conversation_directive_proposal,
            proposal,
        )

    def test_action_accepts_standard_proposal(
        self,
    ) -> None:
        interpretation = TurnInterpretation(
            intent=Intent(
                name="launch_application",
                parameters={
                    "application": "Edge",
                },
            ),
            conversation_directive_proposal=(
                ConversationDirectiveProposal.standard()
            ),
            model="qwen3.5:9b",
        )

        self.assertEqual(
            interpretation.intent.name,
            "launch_application",
        )

    def test_action_rejects_fixed_total_proposal(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            TurnInterpretation(
                intent=Intent(
                    name="launch_application",
                    parameters={
                        "application": "Edge",
                    },
                ),
                conversation_directive_proposal=(
                    ConversationDirectiveProposal.fixed_total_allocation(
                        "trois heures"
                    )
                ),
                model="qwen3.5:9b",
            )

    def test_intent_must_be_intent(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            TurnInterpretation(
                intent="conversation",
                conversation_directive_proposal=(
                    ConversationDirectiveProposal.standard()
                ),
                model="qwen3.5:9b",
            )

    def test_proposal_must_have_correct_type(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            TurnInterpretation(
                intent=Intent(
                    name="conversation"
                ),
                conversation_directive_proposal="standard",
                model="qwen3.5:9b",
            )

    def test_model_is_normalized(
        self,
    ) -> None:
        interpretation = TurnInterpretation(
            intent=Intent(
                name="conversation"
            ),
            conversation_directive_proposal=(
                ConversationDirectiveProposal.standard()
            ),
            model="  qwen3.5:9b  ",
        )

        self.assertEqual(
            interpretation.model,
            "qwen3.5:9b",
        )

    def test_model_cannot_be_empty(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            TurnInterpretation(
                intent=Intent(
                    name="conversation"
                ),
                conversation_directive_proposal=(
                    ConversationDirectiveProposal.standard()
                ),
                model="   ",
            )


if __name__ == "__main__":
    unittest.main()
