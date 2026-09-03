"""Tests for subject-bound, non-persistent profile candidates."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from assistant_ia.intelligence.profile_fact_candidates import (
    PROFILE_FACT_CANDIDATE_RESPONSE_SCHEMA,
    OllamaProfileFactCandidateAnalyzer,
    ProfileFactCandidateAnalysisError,
    _parse_response,
)


class FakeHTTPResponse:
    def __init__(self, candidates: list[object]) -> None:
        content = json.dumps({"candidates": candidates}, ensure_ascii=False)
        self._body = json.dumps(
            {"message": {"content": content}}, ensure_ascii=False
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _candidate(message: str, *, category: str = "preference") -> dict[str, object]:
    return {
        "category": category,
        "content": message,
        "source_text": message,
        "confidence": 0.9,
    }


class ProfileFactCandidateAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = OllamaProfileFactCandidateAnalyzer()

    def test_first_person_preference_receives_application_subject(self) -> None:
        message = "Je préfère les réponses courtes."
        with patch(
            "assistant_ia.intelligence.profile_fact_candidates.urlopen",
            return_value=FakeHTTPResponse([
                _candidate(message, category="communication_preference")
            ]),
        ) as request_call:
            candidates = self.analyzer.analyze(message, person_id=7)

        self.assertEqual(candidates[0].person_id, 7)
        self.assertEqual(candidates[0].category, "communication_preference")
        request = request_call.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertNotIn("person_id", json.dumps(payload))
        candidate_properties = PROFILE_FACT_CANDIDATE_RESPONSE_SCHEMA[
            "properties"
        ]["candidates"]["items"]["properties"]
        self.assertNotIn("id", candidate_properties)

    def test_out_of_scope_messages_are_prefiltered(self) -> None:
        messages = (
            "Alice préfère le thé.",
            "J'ai parlé à Alice.",
            "Je m'appelle Alice.",
            "Est-ce que je préfère le thé ?",
            "Je pense que je préfère peut-être le thé.",
            "Aujourd'hui, je préfère le thé.",
        )
        with patch(
            "assistant_ia.intelligence.profile_fact_candidates.urlopen"
        ) as request_call:
            for message in messages:
                with self.subTest(message=message):
                    self.assertEqual(
                        self.analyzer.analyze(message, person_id=1), ()
                    )
        request_call.assert_not_called()

    def test_secret_is_rejected_before_model_call(self) -> None:
        with patch(
            "assistant_ia.intelligence.profile_fact_candidates.urlopen"
        ) as request_call:
            result = self.analyzer.analyze(
                "Mon mot de passe est sk-abcdefghijklmnop.", person_id=1
            )
        self.assertEqual(result, ())
        request_call.assert_not_called()

    def test_model_cannot_override_person_id_or_add_sql_fields(self) -> None:
        message = "Je préfère le thé."
        malicious = {
            **_candidate(message),
            "person_id": 99,
        }
        with self.assertRaisesRegex(
            ProfileFactCandidateAnalysisError,
            "fields",
        ):
            _parse_response(
                json.dumps({"candidates": [malicious]}),
                authorized_message=message,
                person_id=1,
            )

    def test_low_confidence_and_ungrounded_content_are_rejected(self) -> None:
        message = "Je préfère le thé."
        low = {**_candidate(message), "confidence": 0.5}
        invented = {**_candidate(message), "content": "Je préfère le café."}

        for item in (low, invented):
            with self.subTest(item=item):
                self.assertEqual(
                    _parse_response(
                        json.dumps({"candidates": [item]}),
                        authorized_message=message,
                        person_id=1,
                    ),
                    (),
                )


if __name__ == "__main__":
    unittest.main()
