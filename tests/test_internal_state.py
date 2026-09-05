"""FRP-IA-08: transitions explicables et intégration sans matériel."""
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

from assistant_ia.internal_state import InternalStateService, StateEvent as E
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.model_client import ModelClientError


class InternalStateTests(unittest.TestCase):
    def test_initial_and_immutable(self):
        s = InternalStateService().snapshot
        self.assertEqual(s.activity, "available")
        with self.assertRaises(FrozenInstanceError):
            s.activity = "engaged"

    def test_repeated_failure_survives_start(self):
        s = InternalStateService()
        s.transition(E.ACTION_FAILED, action="test")
        s.transition(E.TURN_STARTED)
        self.assertEqual(s.snapshot.guidance, "blocked")
        s.transition(E.ACTION_FAILED, action="test")
        self.assertEqual(s.snapshot.guidance, "needs_strategy_change")

    def test_different_action_restarts_streak(self):
        s = InternalStateService()
        s.transition(E.ACTION_FAILED, action="a")
        s.transition(E.ACTION_FAILED, action="b")
        self.assertEqual(s.snapshot.consecutive_failures, 1)

    def test_success_cancel_person_goal_reset_clear(self):
        for event in (E.ACTION_SUCCEEDED, E.CANCELLED, E.PERSON_CHANGED, E.GOAL_COMPLETED, E.RESET):
            with self.subTest(event=event):
                s = InternalStateService()
                s.transition(E.ACTION_FAILED, action="a")
                s.transition(event)
                self.assertIsNone(s.snapshot.guidance)
                self.assertEqual(s.snapshot.consecutive_failures, 0)

    def test_correction(self):
        s = InternalStateService()
        self.assertEqual(s.transition(E.CORRECTION).guidance, "needs_strategy_change")

    def test_wait_and_missing_information(self):
        s = InternalStateService()
        self.assertEqual(s.transition(E.CONFIRMATION_REQUIRED).activity, "waiting")
        self.assertEqual(s.transition(E.INFORMATION_REQUIRED).guidance, "missing_information")

    def test_expiration(self):
        now = [0]
        s = InternalStateService(clock=lambda: now[0])
        s.transition(E.ERROR)
        now[0] = 300
        self.assertEqual(s.snapshot.reason, "idle")

    def test_running_turn_does_not_expire(self):
        now = [0]
        s = InternalStateService(clock=lambda: now[0])
        s.transition(E.TURN_STARTED)
        now[0] = 600
        self.assertEqual(s.snapshot.activity, "engaged")

    def test_rejects_model_text(self):
        with self.assertRaises(TypeError):
            InternalStateService().transition("be happy")

    def test_failure_requires_action(self):
        with self.assertRaises(ValueError):
            InternalStateService().transition(E.ACTION_FAILED)

    def test_core_completion_and_reset(self):
        client = Mock()
        client.generate_response.return_value = ModelResponse(content="Salut", model="test", intent=Intent(name="conversation"))
        core = AssistantCore(model_client=client)
        core.process_message("Bonjour")
        self.assertEqual(core.internal_state.snapshot.reason, "turn_completed")
        core.reset_conversation()
        self.assertEqual(core.internal_state.snapshot.reason, "reset")

    def test_core_model_failure(self):
        client = Mock()
        client.generate_response.side_effect = ModelClientError("offline")
        core = AssistantCore(model_client=client)
        with self.assertRaises(RuntimeError):
            core.process_message("Bonjour")
        self.assertEqual(core.internal_state.snapshot.guidance, "blocked")
