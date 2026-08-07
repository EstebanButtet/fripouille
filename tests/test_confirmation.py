"""Tests for the explicit assistant confirmation boundary."""

from __future__ import annotations

import unittest

from assistant_ia.security.confirmation import (
    ConfirmationRequest,
    deny_confirmation,
    request_confirmation,
)


class ConfirmationBoundaryTests(unittest.TestCase):
    """Validate explicit and deterministic action confirmation."""

    def test_normalizes_confirmation_request(self) -> None:
        """Confirmation request fields should be normalized."""
        request = ConfirmationRequest(
            action_name="  launch_application  ",
            description="  lancer Bloc-notes  ",
        )

        self.assertEqual(
            request.action_name,
            "launch_application",
        )
        self.assertEqual(
            request.description,
            "lancer Bloc-notes",
        )

    def test_rejects_empty_action_name(self) -> None:
        """A confirmation request requires an action name."""
        with self.assertRaisesRegex(
            ValueError,
            "action name cannot be empty",
        ):
            ConfirmationRequest(
                action_name="   ",
                description="lancer Bloc-notes",
            )

    def test_rejects_empty_description(self) -> None:
        """A confirmation request requires a visible description."""
        with self.assertRaisesRegex(
            ValueError,
            "description cannot be empty",
        ):
            ConfirmationRequest(
                action_name="launch_application",
                description="   ",
            )

    def test_returns_explicit_confirmation(self) -> None:
        """The injected handler should decide the confirmation."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        result = request_confirmation(
            lambda received_request: (
                received_request is request
            ),
            request,
        )

        self.assertTrue(result)

    def test_preserves_explicit_refusal(self) -> None:
        """A negative handler decision should remain negative."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        result = request_confirmation(
            lambda received_request: False,
            request,
        )

        self.assertFalse(result)

    def test_rejects_non_callable_handler(self) -> None:
        """Confirmation requires an injected callable handler."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        with self.assertRaisesRegex(
            TypeError,
            "handler must be callable",
        ):
            request_confirmation(
                "yes",
                request,
            )

    def test_rejects_non_boolean_handler_result(self) -> None:
        """Confirmation handlers must return an actual boolean."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        with self.assertRaisesRegex(
            TypeError,
            "must return a boolean",
        ):
            request_confirmation(
                lambda received_request: "yes",
                request,
            )

    def test_default_confirmation_denies_action(self) -> None:
        """The noninteractive default should fail closed."""
        request = ConfirmationRequest(
            action_name="launch_application",
            description="lancer Bloc-notes",
        )

        self.assertFalse(
            deny_confirmation(request)
        )


if __name__ == "__main__":
    unittest.main()
