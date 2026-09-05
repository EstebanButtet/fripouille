"""FRP-IA-14 : bruit, hystérésis, fin et borne d'énoncé simulés."""
import unittest
from scripts.voice_vad import UtteranceGate, VadSettings


class VadTests(unittest.TestCase):
    def test_short_noise_does_not_start(self):
        gate = UtteranceGate()
        for probability in [.9]*7+[.1]*30:
            self.assertIsNone(gate.accept(probability))
        self.assertFalse(gate.active)

    def test_start_hysteresis_and_end(self):
        gate = UtteranceGate()
        for _ in range(7): self.assertIsNone(gate.accept(.9))
        self.assertEqual(gate.accept(.9), "started")
        for _ in range(40): self.assertIsNone(gate.accept(.5))
        for _ in range(21): self.assertIsNone(gate.accept(.1))
        self.assertEqual(gate.accept(.1), "ended")

    def test_maximum_utterance(self):
        gate = UtteranceGate(VadSettings(maximum_frames=10))
        for _ in range(8): gate.accept(.9)
        self.assertIsNone(gate.accept(.9))
        self.assertEqual(gate.accept(.9), "ended")

    def test_invalid_probability(self):
        for value in (-1, 2, float("nan")):
            with self.assertRaises(ValueError): UtteranceGate().accept(value)
