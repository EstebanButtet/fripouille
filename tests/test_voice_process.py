"""FRP-IA-14 : IPC réel sans modèle, matériel ou accès réseau."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event, Timer
import unittest
from unittest.mock import Mock

from assistant_ia.voice import AudioError, AudioCancelled
from assistant_ia.voice_process import VoiceConfig, VoiceProcess, LocalSpeechToText, FallbackSpeechToText

WORKER = '''import json, sys, time
for line in sys.stdin.buffer:
 r=json.loads(line); op=r['op']
 if op=='hang': time.sleep(10)
 if op=='invalid': print('x'*17000, flush=True); continue
 if op=='listen': print(json.dumps({'id':r['id'],'event':'speech_started'}),flush=True)
 print(json.dumps({'id':r['id'],'status':'ok','text':"$(Remove-Item x)"}),flush=True)
'''


class VoiceProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        worker = Path(self.tmp.name)/"worker.py"
        worker.write_text(WORKER, encoding="utf-8")
        self.process = VoiceProcess(VoiceConfig(Path(sys.executable), worker, timeout=1))
        self.addCleanup(self.process.close)

    def test_persistent_process_and_transcription_is_only_data(self):
        provider = LocalSpeechToText(self.process)
        callback = Mock()
        self.assertEqual(provider.listen_interruptible(Event(), callback), "$(Remove-Item x)")
        pid = self.process._process.pid
        self.assertEqual(provider.listen(Event()), "$(Remove-Item x)")
        self.assertEqual(pid, self.process._process.pid)
        callback.assert_called_once()

    def test_timeout_reaps_and_can_restart(self):
        with self.assertRaises(AudioError): self.process.request("hang", Event())
        self.assertIsNone(self.process._process)
        self.assertEqual(self.process.request("listen", Event())["status"], "ok")

    def test_cancellation_of_stuck_worker_reaps(self):
        cancel = Event()
        timer = Timer(.1, cancel.set)
        timer.start()
        try:
            with self.assertRaises(AudioCancelled): self.process.request("hang", cancel)
        finally: timer.join()
        self.assertIsNone(self.process._process)

    def test_pre_cancel_never_starts(self):
        cancel = Event(); cancel.set()
        with self.assertRaises(AudioCancelled): self.process.request("listen", cancel)
        self.assertIsNone(self.process._process)

    def test_oversize_protocol_fails_closed(self):
        with self.assertRaises(AudioError): self.process.request("invalid", Event())
        self.assertIsNone(self.process._process)

    def test_fallback_error_but_not_silence_or_cancel(self):
        primary, fallback, notify = Mock(), Mock(), Mock()
        provider = FallbackSpeechToText(primary, fallback, notify)
        primary.listen.return_value = None
        self.assertIsNone(provider.listen(Event()))
        fallback.listen.assert_not_called()
        primary.listen.side_effect = AudioCancelled()
        with self.assertRaises(AudioCancelled): provider.listen(Event())
        fallback.listen.assert_not_called()
        primary.listen.side_effect = AudioError()
        fallback.listen.return_value = "répété"
        self.assertEqual(provider.listen(Event()), "répété")
        notify.assert_called_once()

    def test_config_validation(self):
        for kwargs in ({"model":"anything"}, {"timeout":0}, {"microphone":True}):
            with self.assertRaises(ValueError): VoiceConfig(Path("python"),Path("worker"),**kwargs)
