"""FRP-IA-09 : entrée/sortie locale interchangeable, un seul cerveau."""
from dataclasses import dataclass
from threading import Event, Lock
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
                 tts: TextToSpeechProvider) -> None:
        self.runtime, self.stt, self.tts = runtime, stt, tts
        self._cancel = Event()
        self._lock = Lock()

    def stop(self) -> None:
        self._cancel.set()

    def run_once(self) -> VoiceTurn:
        if not self._lock.acquire(blocking=False):
            return VoiceTurn("busy")
        transcript = response = None
        self._cancel.clear()
        try:
            transcript = self.stt.listen(self._cancel)
            if self._cancel.is_set():
                return VoiceTurn("cancelled")
            if transcript is None:
                return VoiceTurn("silence")
            if not isinstance(transcript, str) or len(transcript) > 2000:
                raise AudioError("Transcription invalide ou trop longue.")
            transcript = transcript.strip()
            if not transcript:
                return VoiceTurn("silence")
            response = self.runtime.process_message(transcript)
            if self._cancel.is_set():
                return VoiceTurn("cancelled", transcript, response)
            self.tts.speak(response, self._cancel)
            return VoiceTurn("cancelled" if self._cancel.is_set() else "completed", transcript, response)
        except AudioCancelled:
            return VoiceTurn("cancelled", transcript, response)
        except AudioError as error:
            return VoiceTurn("error", transcript, response, str(error))
        finally:
            self._cancel.set()
            self._lock.release()
