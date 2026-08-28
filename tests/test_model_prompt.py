"""Tests for the structured intent system prompt."""

from __future__ import annotations

import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.model_client import INTENT_SYSTEM_PROMPT
from assistant_ia.intelligence.prompt import build_system_prompt


class IntentSystemPromptTests(unittest.TestCase):
    """Validate security-relevant intent interpretation rules."""

    def test_latest_user_message_remains_request_to_classify(
        self,
    ) -> None:
        """Previous action failures must not redefine the current intent."""
        self.assertIn(
            "The most recent user message is the request to classify.",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "previous refusal, validation error or failed action",
            INTENT_SYSTEM_PROMPT,
        )

    def test_build_system_prompt_orders_operational_rules_before_identity(
        self,
    ) -> None:
        """Operational rules should precede the assistant identity."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        operational_position = prompt.index(
            "The application, not the language model"
        )
        priority_position = prompt.index(
            "The operational rules above always take priority"
        )
        identity_position = prompt.index(
            "Assistant identity"
        )

        self.assertLess(
            operational_position,
            priority_position,
        )
        self.assertLess(
            priority_position,
            identity_position,
        )
        self.assertIn(
            "Name: Fripouille",
            prompt,
        )

    def test_conversation_is_primary_model_role(
        self,
    ) -> None:
        """The model should primarily act as a conversational assistant."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertTrue(
            prompt.startswith(
                "You are a local personal assistant."
            )
        )
        self.assertIn(
            "Your primary job is to understand and answer "
            "the current user message.",
            prompt,
        )
        self.assertIn(
            "Intent classification is an additional structured "
            "responsibility.",
            prompt,
        )

    def test_conversation_rules_precede_intent_contracts(
        self,
    ) -> None:
        """Conversation quality should be established before classification."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertLess(
            prompt.index(
                "Conversational response quality rules:"
            ),
            prompt.index(
                "Allowed intentions and exact parameter contracts:"
            ),
        )

    def test_assistant_name_is_never_used_for_user(
        self,
    ) -> None:
        """The assistant name must never become a user form of address."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Fripouille is the assistant's name, never the user's name.",
            prompt,
        )
        self.assertIn(
            "If the user's name is unknown, address the user without "
            "using a name.",
            prompt,
        )

    def test_name_separation_precedes_intent_rules(
        self,
    ) -> None:
        """Role separation should be established before classification."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertLess(
            prompt.index(
                "Fripouille is the assistant's name, never the user's name."
            ),
            prompt.index(
                "Allowed intentions and exact parameter contracts:"
            ),
        )

    def test_painful_conversation_requires_personal_presence(
        self,
    ) -> None:
        """Painful situations should receive more than generic sympathy."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Do not stop at a generic expression of sympathy.",
            prompt,
        )
        self.assertIn(
            "Acknowledge the specific emotional weight of what "
            "the user shared.",
            prompt,
        )

    def test_painful_conversation_avoids_forced_behavior(
        self,
    ) -> None:
        """Serious replies should stay natural and grounded."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Do not force humor, advice or a question in painful "
            "situations.",
            prompt,
        )
        self.assertIn(
            "Do not invent details about the loss or distress.",
            prompt,
        )

    def test_conversation_uses_all_explicit_constraints(
        self,
    ) -> None:
        """Conversation should consider explicit constraints together."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Consider all explicit constraints together.",
            prompt,
        )
        self.assertIn(
            "Do not answer only the easiest or most salient part.",
            prompt,
        )

    def test_conversation_adapts_depth_to_the_request(
        self,
    ) -> None:
        """Substantive requests should receive substantive responses."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "Match response depth to the substance of the request.",
            prompt,
        )
        self.assertIn(
            "Develop the answer when reasoning or several factors matter.",
            prompt,
        )

    def test_identity_name_is_not_the_user_name(
        self,
    ) -> None:
        """The assistant name should never be assigned to the user."""
        prompt = build_system_prompt(
            build_default_identity()
        )

        self.assertIn(
            "The assistant identity name belongs to the assistant, "
            "not the user.",
            prompt,
        )
        self.assertIn(
            "Never address the user by the assistant's name.",
            prompt,
        )

    def test_capability_question_is_not_execution_request(
        self,
    ) -> None:
        """Questions about capabilities should not execute actions."""
        prompt = build_system_prompt(
            build_default_identity()
        )
        normalized_prompt = " ".join(
            prompt.split()
        )

        self.assertIn(
            "Questions about whether an action is available or possible "
            "are conversation, not execution requests.",
            normalized_prompt,
        )
        self.assertIn(
            "Execute an action only when the user clearly asks for it "
            "to be performed.",
            normalized_prompt,
        )

    def test_action_requires_direct_explicit_request(
        self,
    ) -> None:
        """Application mentions should not become launch requests."""
        self.assertIn(
            "Use launch_application only for a direct and explicit "
            "request to open,",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "start or launch an application.",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "A mention, preference, suggestion, hypothetical or future "
            "possibility is",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "conversation, not launch_application.",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "A question asking which application or tool the user "
            "should open is an",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "information request, not an execution request",
            INTENT_SYSTEM_PROMPT,
        )

    def test_ambiguous_launch_reference_must_not_be_invented(
        self,
    ) -> None:
        """An unresolved reference should remain an ambiguous action."""
        self.assertIn(
            "If the application cannot be identified from the current "
            "request or relevant",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "conversation context, use unknown.",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never use vague words such as this, that or it as the "
            "application parameter.",
            INTENT_SYSTEM_PROMPT,
        )

    def test_launch_application_requires_application_parameter(
        self,
    ) -> None:
        """Launch intents should preserve explicit application names."""
        self.assertIn(
            "Never omit the application parameter",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            'application "lol"',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            'application "valo"',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            'application "ow2"',
            INTENT_SYSTEM_PROMPT,
        )

    def test_model_must_not_claim_application_launch_success(
        self,
    ) -> None:
        """Only the application layer may report execution success."""
        self.assertIn(
            "The application layer alone reports execution success.",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "never state that the application is",
            INTENT_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
