"""FRP-IA-11 : temps, géométrie et séparation d'identité sans caméra."""
import unittest
from unittest.mock import Mock
from assistant_ia.social_vision import SocialVisionService, FaceDetection, VisionFrame


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.now = [10.0]
        self.service = SocialVisionService(clock=lambda: self.now[0])
        self.face = FaceDetection(.5, .5, .2, .3)

    def frame(self, face=True):
        return VisionFrame(self.now[0], self.face if face else None)

    def test_disabled_by_default(self):
        self.assertFalse(self.service.accept(self.frame()))
        self.assertEqual(self.service.snapshot.status, "unavailable")

    def test_arrival(self):
        self.service.start()
        self.service.accept(self.frame())
        s = self.service.snapshot
        self.assertEqual((s.status, s.event, s.position, s.orientation), ("present", "arrived", "center", "unknown"))
        self.assertFalse(hasattr(s, "person_id"))

    def test_continuity(self):
        self.service.start()
        self.service.accept(self.frame())
        track = self.service.snapshot.track_id
        self.now[0] += .1
        self.service.accept(self.frame())
        self.assertEqual(self.service.snapshot.track_id, track)
        self.assertEqual(self.service.snapshot.event, "continued")

    def test_departure(self):
        self.service.start()
        self.service.accept(self.frame())
        self.now[0] += .1
        self.service.accept(self.frame(False))
        self.assertEqual(self.service.snapshot.event, "departed")
        self.assertIsNone(self.service.snapshot.track_id)

    def test_expiration_is_not_departure(self):
        self.service.start()
        self.service.accept(self.frame())
        self.now[0] += 2
        self.assertEqual(self.service.snapshot.status, "expired")
        self.assertIsNone(self.service.snapshot.event)

    def test_reappearance_has_new_track(self):
        self.service.start()
        self.service.accept(self.frame())
        track = self.service.snapshot.track_id
        self.now[0] += 2
        self.service.accept(self.frame())
        self.assertNotEqual(self.service.snapshot.track_id, track)

    def test_stale_future_and_duplicate_rejected(self):
        self.service.start()
        self.service.accept(self.frame())
        for stamp in (8, 9, 10, 11):
            self.assertFalse(self.service.accept(VisionFrame(stamp, None)))
        self.assertEqual(self.service.snapshot.status, "present")

    def test_stop_clears_and_disables(self):
        self.service.start()
        self.service.accept(self.frame())
        self.service.stop()
        self.assertEqual(self.service.snapshot.status, "unavailable")
        self.assertFalse(self.service.accept(self.frame()))

    def test_invalid_geometry(self):
        for x in (float('nan'), float('inf'), -1, True, "0.5"):
            with self.assertRaises((ValueError, TypeError)):
                FaceDetection(x, .5, .2, .3)

    def test_no_identity_or_emotion_payload(self):
        with self.assertRaises(TypeError):
            self.service.accept({"person_id": 1, "emotion": "happy"})

    def test_orientation_is_only_geometry(self):
        self.service.start()
        self.service.accept(VisionFrame(10, FaceDetection(.1, .5, .2, .3, 15)))
        self.assertEqual(self.service.snapshot.orientation, "approximately_frontal")
        self.assertEqual(self.service.snapshot.position, "left")

    def test_provider_failure_is_unavailable_not_absent(self):
        self.service.start()
        provider = Mock()
        provider.read.side_effect = OSError("camera disconnected")
        self.assertFalse(self.service.poll(provider))
        self.assertEqual(self.service.snapshot.status, "unavailable")

    def test_provider_not_read_when_disabled(self):
        provider = Mock()
        self.assertFalse(self.service.poll(provider))
        provider.read.assert_not_called()
