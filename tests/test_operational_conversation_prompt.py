"""Tests for operational boundaries in model prompts."""

import unittest

from assistant_ia.capabilities.context import CapabilityContext
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.prompt import (
    build_conversation_prompt,
    build_interpretation_prompt,
)
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person


class OperationalConversationPromptTests(unittest.TestCase):
    """Validate action boundaries across both model phases."""

    def test_interpretation_prompt_covers_ambiguous_launch_request(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()

        self.assertIn(
            "User: Ouvre \u00e7a.",
            prompt,
        )
        self.assertIn(
            '{"name":"unknown","parameters":{},"conversation":{"mode":"standard","target_text":null}}',
            prompt,
        )

    def test_conversation_prompt_forbids_execution_claims(
        self,
    ) -> None:
        identity = build_default_identity()

        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=build_default_person(),
        )

        capability_context = CapabilityContext(
            available_actions=(
                "launch_application",
            ),
        )

        prompt = build_conversation_prompt(
            identity,
            person_context,
            capability_context,
        )

        normalized_prompt = " ".join(
            prompt.split()
        )

        self.assertIn(
            "This conversational call never executes actions.",
            normalized_prompt,
        )
        self.assertIn(
            "Never claim that an action was performed, is being "
            "performed or is about to be performed",
            normalized_prompt,
        )
        self.assertIn(
            "answer only whether that capability is available",
            normalized_prompt,
        )


if __name__ == "__main__":
    unittest.main()
