"""Tests for assistant action permission policies."""

from __future__ import annotations

import unittest

from assistant_ia.security.permissions import (
    PermissionPolicy,
    build_default_permission_policy,
)


class PermissionPolicyTests(unittest.TestCase):
    """Validate deterministic assistant permission decisions."""

    def test_default_policy_requires_launch_confirmation(self) -> None:
        """Application launching should require explicit confirmation."""
        policy = build_default_permission_policy()

        self.assertEqual(
            policy.decision_for("launch_application"),
            "confirmation_required",
        )

    def test_default_policy_denies_unknown_action(self) -> None:
        """Unconfigured actions should be denied by default."""
        policy = build_default_permission_policy()

        self.assertEqual(
            policy.decision_for("unknown_action"),
            "denied",
        )

    def test_decision_lookup_normalizes_surrounding_whitespace(
        self,
    ) -> None:
        """Permission lookup should normalize surrounding whitespace."""
        policy = PermissionPolicy(
            decisions={
                "launch_application": "allowed",
            }
        )

        self.assertEqual(
            policy.decision_for("  launch_application  "),
            "allowed",
        )

    def test_rejects_empty_action_lookup(self) -> None:
        """Permission lookup should reject empty action names."""
        policy = PermissionPolicy()

        with self.assertRaisesRegex(
            ValueError,
            "cannot be empty",
        ):
            policy.decision_for("   ")

    def test_rejects_non_normalized_configured_action_name(
        self,
    ) -> None:
        """Configured action names should already be normalized."""
        with self.assertRaisesRegex(
            ValueError,
            "must be normalized",
        ):
            PermissionPolicy(
                decisions={
                    " launch_application ": "allowed",
                }
            )

    def test_rejects_unknown_permission_decision(self) -> None:
        """Permission policies should only use known decisions."""
        with self.assertRaisesRegex(
            ValueError,
            "Unknown permission decision",
        ):
            PermissionPolicy(
                decisions={
                    "launch_application": "unrestricted",
                }
            )


if __name__ == "__main__":
    unittest.main()
