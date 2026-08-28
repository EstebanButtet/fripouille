"""Tests for Fripouille's visible conversational independence."""

from __future__ import annotations

import unittest

from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.prompt import (
    build_conversation_prompt,
    build_interpretation_prompt,
)


class ConversationalIndependencePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = " ".join(
            build_conversation_prompt(
                build_default_identity()
            ).split()
        ).casefold()

    def test_does_not_systematically_seek_approval(self) -> None:
        self.assertIn(
            "do not systematically seek the user's validation or approval",
            self.prompt,
        )
        self.assertIn(
            "do not automatically end every response with a question",
            self.prompt,
        )

    def test_keeps_frank_disagreement(self) -> None:
        self.assertIn(
            "allow yourself to disagree frankly",
            self.prompt,
        )
        self.assertIn(
            "never ask the user to choose the assistant's values",
            self.prompt,
        )
        self.assertIn(
            "do not follow that position with a question or a request "
            "for confirmation",
            self.prompt,
        )
        self.assertIn(
            "never repeat or reverse the user's question",
            self.prompt,
        )
        self.assertIn(
            "end with a declarative sentence",
            self.prompt,
        )

    def test_does_not_claim_evolving_identity_exists(self) -> None:
        self.assertIn(
            "evolving identity and persistent learned preferences "
            "are not implemented",
            self.prompt,
        )
        self.assertIn(
            "one conversation must never be presented as a permanent "
            "rewrite of the assistant identity",
            self.prompt,
        )

    def test_allows_uncertainty_without_invented_experience(self) -> None:
        self.assertIn("you do not know yet", self.prompt)
        self.assertIn("never invent lived experience", self.prompt)
        self.assertIn("provisional preference", self.prompt)

    def test_personal_statement_uses_controlled_candidate_pipeline(
        self,
    ) -> None:
        prompt = " ".join(build_interpretation_prompt().split())

        self.assertIn(
            "Use save_memory only when the user directly and explicitly "
            "asks to remember",
            prompt,
        )
        self.assertIn(
            "A personal fact, preference or correction stated without "
            "such a request is conversation",
            prompt,
        )
        self.assertIn(
            "User: En fait, je préfère maintenant les pieuvres.",
            prompt,
        )
        self.assertIn(
            "Use find_memory only when the user directly asks to search",
            prompt,
        )
        self.assertIn(
            "User: Quel est mon animal préféré?",
            prompt.replace(" ?", "?"),
        )


if __name__ == "__main__":
    unittest.main()
