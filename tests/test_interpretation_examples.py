"""Tests for authoritative intent interpretation examples."""

import unittest

from assistant_ia.intelligence.prompt import (
    build_interpretation_prompt,
)


class InterpretationExamplesTests(unittest.TestCase):
    """Validate the operational few-shot decision boundary."""

    def test_prompt_distinguishes_capability_from_execution(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()

        self.assertIn(
            "Authoritative decision examples:",
            prompt,
        )
        self.assertIn(
            "User: Tu peux lancer Edge ?",
            prompt,
        )
        self.assertIn(
            '{"name":"conversation","parameters":{},"conversation":{"mode":"standard","target_text":null}}',
            prompt,
        )
        self.assertIn(
            "User: Lance Edge.",
            prompt,
        )
        self.assertIn(
            '"launch_application"',
            prompt,
        )

    def test_prompt_distinguishes_memory_request_from_capability_question(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()

        self.assertIn(
            "Souviens-toi que mon v\u00e9lo est rouge.",
            prompt,
        )
        self.assertIn(
            '"save_memory"',
            prompt,
        )
        self.assertIn(
            "Est-ce que tu peux rechercher mes souvenirs "
            "enregistr\u00e9s ?",
            prompt,
        )
        self.assertIn(
            "Recherche dans mes souvenirs ce qui concerne le v\u00e9lo.",
            prompt,
        )
        self.assertIn(
            '"find_memory"',
            prompt,
        )

    def test_prompt_prevents_copying_example_parameters(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()
        normalized_prompt = " ".join(
            prompt.split()
        )

        self.assertIn(
            "Do not copy example parameters when the current request "
            "contains different values.",
            normalized_prompt,
        )


if __name__ == "__main__":
    unittest.main()
