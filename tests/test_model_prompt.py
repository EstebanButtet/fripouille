"""Tests for the structured intent system prompt."""

from __future__ import annotations

import unittest

from assistant_ia.intelligence.model_client import INTENT_SYSTEM_PROMPT


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
