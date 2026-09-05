"""FRP-IA-09 : tours vocaux bornés sans périphérique réel."""
import unittest
from threading import Event
from unittest.mock import Mock, patch
from assistant_ia.voice import VoiceController, AudioError, AudioCancelled
from assistant_ia.windows_speech import WindowsSpeechToText, WindowsTextToSpeech, _run


class VoiceTests(unittest.TestCase):
    def test_voice_cli(self):
        from assistant_ia.__main__ import main
        with patch("assistant_ia.__main__.run_terminal") as run:
            main(["--voice"])
            run.assert_called_once_with(debug=False, voice=True)

    def test_voice_gui_rejected(self):
        from assistant_ia.__main__ import main
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            main(["--voice", "--gui"])

    def setUp(self):
        self.runtime, self.stt, self.tts = Mock(), Mock(), Mock()
        self.stt.listen.return_value = " Bonjour "
        self.runtime.process_message.return_value = "Salut"
        self.voice = VoiceController(self.runtime, self.stt, self.tts)

    def test_same_runtime_and_final_response(self):
        turn = self.voice.run_once()
        self.assertEqual(turn.status, "completed")
        self.runtime.process_message.assert_called_once_with("Bonjour")
        self.assertEqual(self.tts.speak.call_args.args[0], "Salut")

    def test_silence_never_reaches_core(self):
        for value in (None, "", "   "):
            self.stt.listen.return_value = value
            self.assertEqual(self.voice.run_once().status, "silence")
        self.runtime.process_message.assert_not_called()

    def test_invalid_transcription(self):
        for value in (42, "a" * 2001):
            self.stt.listen.return_value = value
            self.assertEqual(self.voice.run_once().status, "error")
        self.runtime.process_message.assert_not_called()

    def test_missing_microphone(self):
        self.stt.listen.side_effect = AudioError("Microphone absent")
        self.assertEqual(self.voice.run_once().status, "error")
        self.runtime.process_message.assert_not_called()

    def test_cancel_capture(self):
        def listen(cancel):
            self.voice.stop()
            return "ignore"
        self.stt.listen.side_effect = listen
        self.assertEqual(self.voice.run_once().status, "cancelled")
        self.runtime.process_message.assert_not_called()

    def test_cancel_after_core_does_not_speak_or_retry(self):
        def process(text):
            self.voice.stop()
            return "Fait"
        self.runtime.process_message.side_effect = process
        turn = self.voice.run_once()
        self.assertEqual((turn.status, turn.response), ("cancelled", "Fait"))
        self.tts.speak.assert_not_called()
        self.runtime.process_message.assert_called_once()

    def test_tts_failure_preserves_text(self):
        self.tts.speak.side_effect = AudioError("sortie absente")
        turn = self.voice.run_once()
        self.assertEqual(turn.response, "Salut")
        self.assertEqual(turn.status, "error")

    def test_tts_cancel_preserves_text(self):
        self.tts.speak.side_effect = AudioCancelled()
        self.assertEqual(self.voice.run_once().response, "Salut")

    def test_no_overlapping_turns(self):
        nested = []
        self.stt.listen.side_effect = lambda cancel: nested.append(self.voice.run_once().status)
        self.voice.run_once()
        self.assertEqual(nested, ["busy"])

    def test_restart_after_error(self):
        self.stt.listen.side_effect = [AudioError(), "Bonjour"]
        self.voice.run_once()
        self.assertEqual(self.voice.run_once().status, "completed")

    def test_windows_stt_json(self):
        with patch("assistant_ia.windows_speech._run", return_value='{"text":"bonjour"}'):
            self.assertEqual(WindowsSpeechToText().listen(Event()), "bonjour")

    def test_windows_stt_rejects_invalid_output(self):
        for value in ('oops', '{}', '{"text":123}'):
            with patch("assistant_ia.windows_speech._run", return_value=value):
                with self.assertRaises(AudioError):
                    WindowsSpeechToText().listen(Event())

    def test_tts_text_is_data_and_bounded(self):
        with patch("assistant_ia.windows_speech._run") as run:
            WindowsTextToSpeech().speak('$(Remove-Item x) ' * 200, Event())
            self.assertEqual(len(run.call_args.kwargs["text"]), 2000)
            self.assertNotIn('Remove-Item', run.call_args.args[0])

    def test_pre_cancel_does_not_start_process(self):
        cancel = Event()
        cancel.set()
        with patch("subprocess.Popen") as popen:
            with self.assertRaises(AudioCancelled):
                _run("ignored", cancel)
            popen.assert_not_called()

    def test_process_cancel_kills_and_reaps(self):
        cancel = Event()
        process = Mock()
        process.poll.return_value = None
        def spawn(*args, **kwargs):
            cancel.set()
            return process
        with patch("subprocess.Popen", side_effect=spawn):
            with self.assertRaises(AudioCancelled):
                _run("fixed script", cancel)
        process.kill.assert_called_once()
        process.communicate.assert_called_once()
