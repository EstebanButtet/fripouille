"""Tests for explicit structured behavioral feedback."""

from datetime import datetime, timezone
import unittest

from assistant_ia.actions.result import ActionExecutionResult
from assistant_ia.learning.outcomes import (
    BehavioralAttempt,
    ExperienceOutcome,
    OutcomeMeasurement,
    UserFeedback,
    outcome_from_action_result,
)


class LearningOutcomesTests(unittest.TestCase):
    def test_attempt_requires_timezone_and_preserves_strategy(self) -> None:
        attempt = BehavioralAttempt(
            person_id=7,
            context="explication",
            objective="faire comprendre",
            strategy="équations d'abord",
            started_at=datetime(2026, 9, 4, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(attempt.person_id, 7)
        self.assertEqual(attempt.strategy, "équations d'abord")

    def test_feedback_is_explicit_and_exact(self) -> None:
        feedback = UserFeedback("correction", "Je voulais X, pas Y.")
        outcome = ExperienceOutcome(
            status="failure",
            kind="user_feedback",
            summary="La réponse est corrigée par la personne.",
            feedback=feedback,
        )
        self.assertEqual(outcome.feedback, feedback)
        assert outcome.feedback is not None
        self.assertEqual(outcome.feedback.content, feedback.content)

    def test_outcome_rejects_misclassified_feedback(self) -> None:
        with self.assertRaises(ValueError):
            ExperienceOutcome(
                status="success",
                kind="user_feedback",
                summary="ok",
            )
        with self.assertRaises(ValueError):
            ExperienceOutcome(
                status="success",
                kind="verified_action",
                summary="ok",
                feedback=UserFeedback("approval", "Oui."),
            )

    def test_measurements_are_factual_finite_and_unique(self) -> None:
        outcome = ExperienceOutcome(
            status="partial",
            kind="external_result",
            summary="Cible partiellement atteinte.",
            measurements=(OutcomeMeasurement("durée", 1.5, "s"),),
        )
        self.assertEqual(outcome.measurements[0].value, 1.5)
        with self.assertRaises(ValueError):
            OutcomeMeasurement("erreur", float("nan"), "deg")
        with self.assertRaises(ValueError):
            ExperienceOutcome(
                status="partial",
                kind="external_result",
                summary="partiel",
                measurements=(
                    OutcomeMeasurement("erreur", 1, "deg"),
                    OutcomeMeasurement("erreur", 2, "deg"),
                ),
            )

    def test_action_result_mapping_is_application_owned(self) -> None:
        success = ActionExecutionResult("launch_application", "success", "Lancé.", True)
        self.assertEqual(outcome_from_action_result(success).status, "success")
        validation = ActionExecutionResult(
            "launch_application", "error", "Refusé.", False, "validation"
        )
        mapped = outcome_from_action_result(validation)
        self.assertEqual(mapped.status, "not_executed")
        self.assertEqual(mapped.result_code, "action_validation_error")
        execution = ActionExecutionResult(
            "launch_application", "error", "Échec technique.", True, "execution"
        )
        self.assertEqual(outcome_from_action_result(execution).kind, "technical_error")


if __name__ == "__main__":
    unittest.main()
