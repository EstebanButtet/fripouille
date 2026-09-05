"""FRP-IA-14 : dialogue continu et interruption sans tours concurrents."""
from threading import Event
import unittest
from unittest.mock import Mock, patch
from assistant_ia.voice import VoiceController, AudioCancelled, AudioError


class ContinuousVoiceTests(unittest.TestCase):
    def test_cli(self):
        from assistant_ia.__main__ import main
        with patch("assistant_ia.__main__.run_terminal") as run:
            main(["--continuous-voice", "--barge-in"])
            run.assert_called_once_with(debug=False,voice=True,continuous_voice=True,barge_in=True)

    def test_barge_requires_continuous(self):
        from assistant_ia.__main__ import main
        with patch("sys.stderr"), self.assertRaises(SystemExit): main(["--barge-in"])

    def test_loop_and_stop_preserve_single_runtime(self):
        runtime, stt, tts = Mock(), Mock(), Mock()
        runtime.process_message.return_value = "réponse"
        stt.listen.side_effect = [None, "bonjour", "suite"]
        controller = VoiceController(runtime, stt, tts)
        turns = []
        def on_turn(turn):
            turns.append(turn)
            if len(turns) == 3: controller.stop()
        controller.run_continuous(on_turn)
        self.assertEqual(runtime.process_message.call_count, 2)
        self.assertEqual(tts.speak.call_count, 2)

    def test_response_is_visible_before_tts(self):
        runtime, stt, tts, show = Mock(), Mock(), Mock(), Mock()
        stt.listen.return_value = "bonjour"
        runtime.process_message.return_value = "salut"
        tts.speak.side_effect = lambda *args: show.assert_called_once_with("salut")
        self.assertEqual(VoiceController(runtime,stt,tts,on_response=show).run_once().status,"completed")

    def test_barge_cancels_speech_then_processes_captured_phrase_once(self):
        runtime, stt, tts = Mock(), Mock(), Mock()
        runtime.process_message.return_value = "réponse"
        stt.listen.return_value = "première phrase"
        speaking, stopped = Event(), Event()
        def speak(text, cancel):
            speaking.set()
            if not cancel.wait(2): raise AssertionError("TTS not cancelled")
            stopped.set()
            raise AudioCancelled()
        def capture(cancel, on_speech):
            self.assertTrue(speaking.wait(2))
            on_speech()
            self.assertTrue(stopped.wait(2))
            return "nouvelle phrase"
        tts.speak.side_effect = speak
        stt.listen_interruptible.side_effect = capture
        controller = VoiceController(runtime,stt,tts,barge_in=True)
        self.assertEqual(controller.run_once().status,"completed")
        self.assertEqual(runtime.process_message.call_count,1)
        controller.barge_in=False
        tts.speak.side_effect=None
        controller.run_once()
        self.assertEqual(runtime.process_message.call_args.args,("nouvelle phrase",))
        stt.listen.assert_called_once()

    def test_no_barge_does_not_capture_during_speech(self):
        runtime, stt, tts = Mock(), Mock(), Mock()
        stt.listen.return_value="bonjour"
        runtime.process_message.return_value="salut"
        VoiceController(runtime,stt,tts).run_once()
        stt.listen_interruptible.assert_not_called()

    def test_engine_error_ends_loop_without_retry(self):
        stt=Mock(); stt.listen.side_effect=AudioError("offline")
        runtime=Mock()
        VoiceController(runtime,stt,Mock()).run_continuous()
        stt.listen.assert_called_once()
        runtime.process_message.assert_not_called()
