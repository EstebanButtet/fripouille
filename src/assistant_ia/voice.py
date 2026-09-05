"""FRP-IA-09 : entrée/sortie locale interchangeable, un seul cerveau."""
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Protocol

from assistant_ia.runtime import AssistantRuntime


class AudioError(RuntimeError):
    """Périphérique, moteur ou échange audio indisponible."""


class AudioCancelled(AudioError):
    """Arrêt explicite du tour audio."""


class SpeechToTextProvider(Protocol):
    def listen(self, cancel: Event) -> str | None:
        """Capturer un seul énoncé borné ; None représente le silence.

        Le backend possède capture, détection de parole et transcription.
        Il doit libérer le micro à la fin et respecter cancel.
        """
        ...


class TextToSpeechProvider(Protocol):
    def speak(self, text: str, cancel: Event) -> None:
        """Synthétiser et jouer localement, avec annulation."""
        ...


@dataclass(frozen=True, slots=True)
class VoiceTurn:
    status: str
    transcript: str | None = None
    response: str | None = None
    error: str | None = None


class VoiceController:
    """Un appel = un tour ; stop est utilisable depuis un autre thread.

    Demi-duplex : pas de réécoute du haut-parleur. L'annulation pendant
    Ollama supprime la parole suivante mais n'annule pas une action engagée.
    Aucun retry automatique ne pourrait rejouer une action.
    """
    def __init__(self, runtime: AssistantRuntime, stt: SpeechToTextProvider,
                 tts: TextToSpeechProvider, *, on_response=None, on_transcript=None,
                 barge_in: bool = False) -> None:
        self.runtime, self.stt, self.tts = runtime, stt, tts
        self._cancel = Event()
        self._lock = Lock()
        self._session_lock = Lock()
        self._stop_loop = Event()
        self._pending_transcript = None
        self._output_cancel = Event()
        self._capture_cancel = Event()
        self.on_response, self.on_transcript = on_response, on_transcript
        self.barge_in = barge_in
        self.state = "idle"
        self.last_metrics = {}

    def stop(self) -> None:
        self._stop_loop.set()
        self._cancel.set()
        self._output_cancel.set()
        self._capture_cancel.set()

    def run_continuous(self, on_turn=None):
        """Retour automatique à l'écoute ; erreur -> retour texte, aucun retry métier."""
        if not self._session_lock.acquire(blocking=False):
            raise AudioError("Session vocale déjà active.")
        self._stop_loop.clear()
        try:
            while not self._stop_loop.is_set():
                turn = self.run_once()
                if on_turn is not None: on_turn(turn)
                if turn.status in {"error", "cancelled", "busy"}: break
        finally:
            self.stop()
            self._pending_transcript = None
            self._session_lock.release()

    def _speak(self, response):
        self.state = "speaking"
        if not self.barge_in:
            self.tts.speak(response, self._cancel)
            return
        # Ce mode nécessite un casque/une isolation acoustique explicitement choisie.
        # Aucune prétention d'annulation d'écho acoustique par simple VAD.
        interrupted, finished = Event(), Event()
        self._output_cancel.clear()
        self._capture_cancel.clear()
        failures = []
        def speech_started():
            interrupted.set()
            self._output_cancel.set()
            self.state = "listening"
        def speak():
            try: self.tts.speak(response, self._output_cancel)
            except AudioCancelled: pass
            except Exception as error: failures.append(error)
            finally:
                finished.set()
                if not interrupted.is_set(): self._capture_cancel.set()
        speaker = Thread(target=speak, name="frp-voice-output", daemon=True)
        speaker.start()
        try:
            while not self._cancel.is_set():
                try:
                    text = self.stt.listen_interruptible(self._capture_cancel, speech_started)
                except AudioCancelled:
                    break
                if text is not None:
                    self._pending_transcript = text
                    break
                if finished.is_set(): break
        finally:
            self._output_cancel.set()
            self._capture_cancel.set()
            speaker.join(timeout=2)
            if speaker.is_alive():
                self.stop()
                raise AudioError("Le fournisseur TTS n'a pas respecté l'annulation.")
        if failures: raise AudioError("Sortie vocale interrompue par une erreur.") from failures[0]

    def run_once(self) -> VoiceTurn:
        if not self._lock.acquire(blocking=False):
            return VoiceTurn("busy")
        transcript = response = None
        self._cancel.clear()
        try:
            if self._session_lock.locked() and self._stop_loop.is_set():
                return VoiceTurn("cancelled")
            self.state = "listening"
            transcript, self._pending_transcript = self._pending_transcript, None
            if transcript is None: transcript = self.stt.listen(self._cancel)
            if self._cancel.is_set():
                return VoiceTurn("cancelled")
            if transcript is None:
                return VoiceTurn("silence")
            if not isinstance(transcript, str) or len(transcript) > 2000:
                raise AudioError("Transcription invalide ou trop longue.")
            transcript = transcript.strip()
            if not transcript:
                return VoiceTurn("silence")
            if self.on_transcript is not None: self.on_transcript(transcript)
            self.state = "processing"
            start = monotonic()
            response = self.runtime.process_message(transcript)
            self.last_metrics = {"runtime_ms": (monotonic()-start)*1000}
            if self.on_response is not None: self.on_response(response)
            if self._cancel.is_set():
                return VoiceTurn("cancelled", transcript, response)
            self._speak(response)
            return VoiceTurn("cancelled" if self._cancel.is_set() else "completed", transcript, response)
        except AudioCancelled:
            return VoiceTurn("cancelled", transcript, response)
        except AudioError as error:
            return VoiceTurn("error", transcript, response, str(error))
        finally:
            self.state = "idle"
            self._cancel.set()
            self._lock.release()
