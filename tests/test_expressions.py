"""FRP-IA-10 : aucune primitive de rendu ne vient du modèle."""
import unittest
from unittest.mock import Mock
from assistant_ia.expressions import Expression as E, ExpressiveIntent, ExpressionController, expression_for_state
from assistant_ia.internal_state import InternalStateSnapshot
from assistant_ia.interfaces.face import CanvasFacePresenter, eye_contour
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.runtime import AssistantRuntime


class ExpressionTests(unittest.TestCase):
    def test_default_neutral(self):
        self.assertEqual(ExpressionController().current.expression, E.NEUTRAL)

    def test_expiration(self):
        now = [0]
        c = ExpressionController(clock=lambda: now[0])
        c.show(ExpressiveIntent(E.CURIOUS))
        self.assertEqual(c.current.expression, E.CURIOUS)
        now[0] = 5
        self.assertEqual(c.current.expression, E.NEUTRAL)

    def test_reset(self):
        c = ExpressionController()
        c.show(ExpressiveIntent(E.CONCERNED))
        c.reset()
        self.assertEqual(c.current.expression, E.NEUTRAL)

    def test_policy(self):
        for state, expected in ((InternalStateSnapshot(activity="engaged"), E.FOCUSED),
                                (InternalStateSnapshot(activity="waiting"), E.CURIOUS),
                                (InternalStateSnapshot(guidance="blocked"), E.CONCERNED),
                                (InternalStateSnapshot(guidance="missing_information"), E.CURIOUS)):
            self.assertEqual(expression_for_state(state).expression, expected)

    def test_no_raw_command(self):
        with self.assertRaises(TypeError):
            ExpressiveIntent("PWM 255")
        with self.assertRaises(TypeError):
            ExpressionController().show("<svg onload='evil'>")

    def test_contour_uses_audited_start(self):
        self.assertAlmostEqual(eye_contour(E.NEUTRAL)[0], 18*.43 + 8)
        self.assertEqual(eye_contour(E.NEUTRAL)[1], 70)

    def test_distinct_renderings(self):
        self.assertEqual(len({eye_contour(e) for e in E}), 4)

    def test_canvas_only_receives_local_geometry(self):
        canvas = Mock()
        CanvasFacePresenter(canvas).present_expression(ExpressiveIntent(E.FOCUSED))
        self.assertEqual(canvas.create_line.call_count, 3)
        self.assertTrue(all(isinstance(v, (int, float)) for v in canvas.create_line.call_args.args))

    def test_model_failure_presents_concern_without_action(self):
        client = Mock()
        client.generate_response.side_effect = RuntimeError("offline")
        runtime = AssistantRuntime(AssistantCore(model_client=client))
        with self.assertRaises(RuntimeError):
            runtime.process_message("bonjour")
        self.assertEqual(runtime.expressions.current.expression, E.CONCERNED)
        self.assertIsNone(runtime.assistant.last_action_result)

    def test_runtime_reset_clears_expression(self):
        runtime = AssistantRuntime(AssistantCore(model_client=Mock()))
        runtime.expressions.show(ExpressiveIntent(E.CURIOUS))
        runtime.reset_conversation()
        self.assertEqual(runtime.expressions.current.expression, E.NEUTRAL)
