from __future__ import annotations

import unittest

from assistant_ia.capabilities.context import CapabilityContext
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.prompt import (
    build_conversation_prompt,
    build_interpretation_prompt,
)
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.models import PersonProfile


class PromptSeparationTests(unittest.TestCase):
    def test_interpretation_prompt_contains_only_operational_work(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()

        self.assertIn(
            "Intent classification and action interpretation rules:",
            prompt,
        )
        self.assertIn(
            "Allowed intentions and exact parameter contracts:",
            prompt,
        )
        self.assertIn(
            "Required JSON schema:",
            prompt,
        )

        self.assertNotIn(
            "Assistant identity",
            prompt,
        )
        self.assertNotIn(
            "Conversational response quality rules:",
            prompt,
        )
        self.assertNotIn(
            "Current assistant capabilities:",
            prompt,
        )
        self.assertNotIn(
            "Current conversation participants:",
            prompt,
        )

    def test_conversation_prompt_has_no_intent_classification_contract(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )
        capability_context = CapabilityContext(
            available_actions=(
                "find_memory",
                "launch_application",
            ),
        )

        prompt = build_conversation_prompt(
            identity,
            person_context,
            capability_context,
        )

        self.assertIn(
            "You are a local personal assistant.",
            prompt,
        )
        self.assertIn(
            "Current conversation participants:",
            prompt,
        )
        self.assertIn(
            "Current user: Este",
            prompt,
        )
        self.assertIn(
            "Current assistant capabilities:",
            prompt,
        )
        self.assertIn(
            "Conversational response quality rules:",
            prompt,
        )
        self.assertIn(
            "Assistant identity",
            prompt,
        )

        self.assertNotIn(
            "Allowed intentions and exact parameter contracts:",
            prompt,
        )
        self.assertNotIn(
            "Required JSON schema:",
            prompt,
        )
        self.assertNotIn(
            "Intent classification and action interpretation rules:",
            prompt,
        )

    def test_conversation_prompt_preserves_user_name_exactly(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )

        prompt = build_conversation_prompt(
            identity,
            person_context,
            CapabilityContext(
                available_actions=(),
            ),
        )

        self.assertIn(
            "The current user's name is exactly: Este.",
            prompt,
        )
        self.assertIn(
            "If you use the current user's name, preserve its exact spelling.",
            prompt,
        )

    def test_conversation_prompt_keeps_capability_truth(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )

        prompt = build_conversation_prompt(
            identity,
            person_context,
            CapabilityContext(
                available_actions=(
                    "find_memory",
                ),
            ),
        )

        self.assertIn(
            "find_memory",
            prompt,
        )
        self.assertIn(
            "Automatic contextual memory retrieval is not available.",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
