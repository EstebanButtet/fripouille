"""Tests for automatic, non-persistent memory candidate analysis."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from assistant_ia.intelligence.memory_candidates import (
    MAX_MEMORY_CANDIDATES,
    MEMORY_CANDIDATE_RESPONSE_SCHEMA,
    MIN_MEMORY_CANDIDATE_CONFIDENCE,
    MemoryCandidateAnalysisError,
    OllamaMemoryCandidateAnalyzer,
    _parse_candidate_response,
)


class FakeHTTPResponse:
    """Expose one encoded Ollama envelope as a context manager."""

    def __init__(self, candidate_data: object) -> None:
        content = json.dumps(candidate_data, ensure_ascii=False)
        self._body = json.dumps(
            {"message": {"content": content}},
            ensure_ascii=False,
        ).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _candidate(
    content: object,
    source_text: object,
    confidence: object = 0.9,
) -> dict[str, object]:
    return {
        "content": content,
        "source_text": source_text,
        "confidence": confidence,
    }


class MemoryCandidateAnalyzerTests(unittest.TestCase):
    """Validate strict Ollama output and deterministic admission policy."""

    def setUp(self) -> None:
        self.analyzer = OllamaMemoryCandidateAnalyzer()

    def _analyze_with_response(
        self,
        message: str,
        candidates: list[object],
    ):
        with patch(
            "assistant_ia.intelligence.memory_candidates.urlopen",
            return_value=FakeHTTPResponse({"candidates": candidates}),
        ) as urlopen:
            result = self.analyzer.analyze(message)
        return result, urlopen

    def test_accepts_explicit_preference_with_exact_source(self) -> None:
        message = "Mon logiciel prefere pour la CAO est SolidWorks."
        result, urlopen = self._analyze_with_response(
            message,
            [_candidate(message, message)],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, message)
        self.assertEqual(result[0].source_text, message)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["messages"][-1]["content"], message)
        self.assertEqual(len(payload["messages"]), 2)

    def test_preserves_at_most_two_candidates_in_model_order(self) -> None:
        message = "Mon projet est Fripouille et mon outil est SolidWorks."
        first = "Mon projet est Fripouille"
        second = "mon outil est SolidWorks"
        result, _ = self._analyze_with_response(
            message,
            [_candidate(first, first), _candidate(second, second)],
        )

        self.assertEqual(
            tuple(candidate.content for candidate in result),
            (first, second),
        )
        self.assertEqual(MAX_MEMORY_CANDIDATES, 2)

    def test_rejects_invalid_individual_candidate_values(self) -> None:
        message = "Mon projet durable est Fripouille."
        invalid_candidates = (
            _candidate("", message),
            _candidate(message, ""),
            _candidate(message, "preuve absente"),
            _candidate("information inventee", message),
            _candidate(message, message, -0.1),
            _candidate(message, message, 1.1),
            _candidate(message, message, float("nan")),
            _candidate(message, message, float("inf")),
            _candidate(message, message, "0.9"),
            _candidate(42, message),
        )

        for candidate_data in invalid_candidates:
            with self.subTest(candidate_data=candidate_data):
                result = _parse_candidate_response(
                    json.dumps(
                        {"candidates": [candidate_data]},
                        allow_nan=True,
                    ),
                    authorized_message=message,
                )
                self.assertEqual(result, ())

    def test_rejects_unexpected_schema_and_too_many_candidates(self) -> None:
        message = "Mon projet est Fripouille."
        valid = _candidate(message, message)
        invalid_payloads = (
            {"candidates": [], "extra": True},
            {"candidates": "wrong"},
            {"candidates": [{**valid, "extra": True}]},
            {"candidates": [valid, valid, valid]},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(
                MemoryCandidateAnalysisError
            ):
                _parse_candidate_response(
                    json.dumps(payload),
                    authorized_message=message,
                )

    def test_schema_is_strict_and_bounded(self) -> None:
        candidate_array = MEMORY_CANDIDATE_RESPONSE_SCHEMA[
            "properties"
        ]["candidates"]
        self.assertFalse(
            MEMORY_CANDIDATE_RESPONSE_SCHEMA["additionalProperties"]
        )
        self.assertEqual(candidate_array["maxItems"], 2)
        self.assertFalse(candidate_array["items"]["additionalProperties"])
        self.assertEqual(MIN_MEMORY_CANDIDATE_CONFIDENCE, 0.8)

    def test_prefilters_non_assertive_or_out_of_scope_messages(self) -> None:
        messages = (
            "Quel logiciel dois-je utiliser ?",
            "Je pense que SolidWorks est peut-etre meilleur.",
            "Si je choisissais Fusion, ce serait pratique.",
            "Imagine que je prefere Blender.",
            "Mon ami m'a dit qu'il prefere Blender.",
            "Pour aujourd'hui, je travaille sur Fusion.",
            "Quand je te demande une reponse courte, reponds toujours vite.",
        )

        with patch(
            "assistant_ia.intelligence.memory_candidates.urlopen"
        ) as urlopen:
            for message in messages:
                with self.subTest(message=message):
                    self.assertEqual(self.analyzer.analyze(message), ())
        urlopen.assert_not_called()

    def test_explicit_personal_correction_may_reach_analysis(self) -> None:
        message = (
            "En fait, je prefere maintenant Fusion 360 pour la CAO."
        )
        result, urlopen = self._analyze_with_response(
            message,
            [_candidate(message, message)],
        )

        self.assertEqual(len(result), 1)
        urlopen.assert_called_once()

    def test_rejects_correction_with_fragmentary_evidence_or_content(
        self,
    ) -> None:
        message = (
            "En fait, je prefere maintenant Fusion 360 pour la CAO."
        )
        invalid_candidates = (
            _candidate("Fusion 360", "Fusion 360"),
            _candidate("Fusion 360", message),
        )

        for candidate_data in invalid_candidates:
            with self.subTest(candidate_data=candidate_data):
                result = _parse_candidate_response(
                    json.dumps({"candidates": [candidate_data]}),
                    authorized_message=message,
                )
                self.assertEqual(result, ())

    def test_accepts_exact_correction_excerpt_with_sufficient_evidence(
        self,
    ) -> None:
        message = (
            "En fait, je prefere maintenant Fusion 360 pour la CAO."
        )
        source_text = "je prefere maintenant Fusion 360 pour la CAO."
        content = "Je prefere maintenant Fusion 360 pour la CAO."

        result = _parse_candidate_response(
            json.dumps(
                {
                    "candidates": [
                        _candidate(content, source_text, 1.0)
                    ]
                }
            ),
            authorized_message=message,
        )

        self.assertEqual(result[0].content, content)
        self.assertEqual(result[0].source_text, source_text)

    def test_prefilters_obvious_authentication_secrets(self) -> None:
        messages = (
            "Mon mot de passe est Azerty123!",
            "Ma cle API est sk-abcdefghijklmnop.",
            "Mon code 2FA est 123456.",
            "-----BEGIN PRIVATE KEY----- abc",
        )

        with patch(
            "assistant_ia.intelligence.memory_candidates.urlopen"
        ) as urlopen:
            for message in messages:
                with self.subTest(message=message):
                    self.assertEqual(self.analyzer.analyze(message), ())
        urlopen.assert_not_called()

    def test_rejects_secret_returned_inside_candidate(self) -> None:
        message = "Information personnelle autorisee."
        result = _parse_candidate_response(
            json.dumps(
                {
                    "candidates": [
                        _candidate(
                            "mot de passe",
                            "Information personnelle",
                        )
                    ]
                }
            ),
            authorized_message=message,
        )
        self.assertEqual(result, ())

    def test_deduplicates_exact_candidate_content(self) -> None:
        message = "Mon projet est Fripouille."
        value = _candidate(message, message)
        result, _ = self._analyze_with_response(message, [value, value])
        self.assertEqual(len(result), 1)

    def test_invalid_ollama_result_is_contained_as_analysis_error(self) -> None:
        invalid_envelopes = (b"not-json", b"[]", b'{"message": {}}')

        for body in invalid_envelopes:
            class RawResponse:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

                def read(self):
                    return body

            with (
                self.subTest(body=body),
                patch(
                    "assistant_ia.intelligence.memory_candidates.urlopen",
                    return_value=RawResponse(),
                ),
                self.assertRaises(MemoryCandidateAnalysisError),
            ):
                self.analyzer.analyze("Mon projet est Fripouille.")

    def test_unavailable_ollama_raises_analysis_error(self) -> None:
        with (
            patch(
                "assistant_ia.intelligence.memory_candidates.urlopen",
                side_effect=URLError("offline"),
            ),
            self.assertRaises(MemoryCandidateAnalysisError),
        ):
            self.analyzer.analyze("Mon projet est Fripouille.")


if __name__ == "__main__":
    unittest.main()
