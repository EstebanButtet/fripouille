"""FRP-IA-14 : IPC vocal local JSON borné, sans dépendance ML dans le cerveau."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Queue, Empty, Full
import subprocess
from threading import Event, Lock, Thread
from time import monotonic

from assistant_ia.voice import AudioError, AudioCancelled

MAX_IPC_LINE = 16384


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    python: Path
    worker: Path
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    cache: Path = Path("models/whisper")
    timeout: float = 90
    microphone: int | None = None

    def __post_init__(self):
        if self.model not in {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}:
            raise ValueError("Modèle Whisper non reconnu.")
        if self.device not in {"cpu", "cuda"} or self.compute_type not in {"int8", "float32", "float16", "int8_float16"}:
            raise ValueError("Configuration de calcul vocale invalide.")
        if isinstance(self.timeout, bool) or not 1 <= self.timeout <= 300:
            raise ValueError("Délai vocal : 1 à 300 secondes.")
        if self.microphone is not None and (type(self.microphone) is not int or self.microphone < 0):
            raise ValueError("Identifiant microphone invalide.")

    @classmethod
    def local(cls):
        root = Path(__file__).resolve().parents[2]
        return cls(Path(os.environ.get("FRIPOUILLE_VOICE_PYTHON", root / ".venv-voice/Scripts/python.exe")),
                   root / "scripts/voice_worker.py",
                   model=os.environ.get("FRIPOUILLE_STT_MODEL", "small"),
                   device=os.environ.get("FRIPOUILLE_STT_DEVICE", "cpu"),
                   compute_type=os.environ.get("FRIPOUILLE_STT_COMPUTE", "int8"),
                   cache=Path(os.environ.get("FRIPOUILLE_WHISPER_CACHE", root / "models/whisper")),
                   microphone=int(os.environ["FRIPOUILLE_MICROPHONE"]) if "FRIPOUILLE_MICROPHONE" in os.environ else None)


class VoiceProcess:
    """Un seul appel actif, processus persistant, annulation puis arrêt borné.

    Le worker ne reçoit jamais le runtime, SQLite, les permissions ou des commandes
    shell. Les événements ne sont que des observations vocales. Aucune socket.
    """
    def __init__(self, config: VoiceConfig):
        self.config = config
        self._process = None
        self._reader = None
        self._lock = Lock()
        self._sequence = 0

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        if not self.config.python.is_file() or not self.config.worker.is_file():
            raise AudioError("Runtime vocal absent ; exécutez scripts/setup_voice.ps1.")
        try:
            self._process = subprocess.Popen(
                [str(self.config.python), "-u", str(self.config.worker),
                 "--model", self.config.model, "--device", self.config.device,
                 "--compute-type", self.config.compute_type, "--cache", str(self.config.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as error:
            raise AudioError("Démarrage du runtime vocal impossible.") from error
        self._events = Queue(maxsize=16)
        process, events = self._process, self._events
        def read():
            try:
                while True:
                    line = process.stdout.readline(MAX_IPC_LINE + 1)
                    if not line or len(line) > MAX_IPC_LINE:
                        break
                    events.put_nowait(json.loads(line))
            except (ValueError, OSError, Full):
                pass
            finally:
                try: events.put_nowait(None)
                except Full: pass
        self._reader = Thread(target=read, name="frp-voice-ipc", daemon=True)
        self._reader.start()

    def _send(self, value):
        raw = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        if len(raw) > MAX_IPC_LINE:
            raise AudioError("Requête vocale trop grande.")
        self._process.stdin.write(raw)
        self._process.stdin.flush()

    def request(self, operation, cancel: Event, *, on_speech=None, **parameters):
        if cancel.is_set():
            raise AudioCancelled("Audio annulé.")
        if not self._lock.acquire(blocking=False):
            raise AudioError("Le moteur vocal est occupé.")
        try:
            self._start()
            self._sequence += 1
            identifier = self._sequence
            self._send({"id": identifier, "op": operation, **parameters})
            deadline = monotonic() + self.config.timeout
            cancelling = None
            while True:
                if cancel.is_set() and cancelling is None:
                    cancelling = monotonic()
                    self._send({"op": "cancel", "id": identifier})
                if cancelling is not None and monotonic() - cancelling >= 1:
                    self.close()
                    raise AudioCancelled("Audio annulé ; moteur arrêté.")
                if monotonic() >= deadline:
                    self.close()
                    if cancelling is not None:
                        raise AudioCancelled("Audio annulé ; moteur arrêté.")
                    raise AudioError("Délai du moteur vocal dépassé.")
                try: event = self._events.get(timeout=.02)
                except Empty: continue
                if not isinstance(event, dict) or event.get("id") != identifier:
                    self.close()
                    raise AudioError("Protocole vocal interrompu ou invalide.")
                if event.get("event") == "speech_started":
                    if on_speech is not None and cancelling is None:
                        on_speech()
                    continue
                if cancelling is not None or event.get("status") == "cancelled":
                    raise AudioCancelled("Audio annulé.")
                if event.get("status") != "ok":
                    raise AudioError("Moteur vocal indisponible : " + str(event.get("error", "erreur"))[:300])
                return event
        except KeyboardInterrupt:
            self.close()
            raise
        except (OSError, ValueError) as error:
            self.close()
            raise AudioError("Échange avec le moteur vocal impossible.") from error
        finally:
            self._lock.release()

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            for stream in (process.stdin, process.stdout):
                if stream is not None: stream.close()
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None


class LocalSpeechToText:
    def __init__(self, process: VoiceProcess):
        self.process = process
        self.last_metrics = {}

    def listen(self, cancel):
        return self.listen_interruptible(cancel)

    def listen_interruptible(self, cancel, on_speech=None):
        result = self.process.request("listen", cancel, on_speech=on_speech,
                                      microphone=self.process.config.microphone)
        self.last_metrics = result.get("metrics", {})
        text = result.get("text")
        if text is not None and (not isinstance(text, str) or len(text) > 2000):
            raise AudioError("Transcription invalide ou trop longue.")
        return text

    def close(self):
        self.process.close()


class FallbackSpeechToText:
    """Bascule annoncée, jamais après une annulation ni sur simple silence."""
    def __init__(self, primary, fallback, notify=print):
        self.primary, self.fallback, self.notify = primary, fallback, notify
        self.degraded = False

    def listen(self, cancel):
        if not self.degraded:
            try: return self.primary.listen(cancel)
            except AudioCancelled: raise
            except AudioError:
                self.degraded = True
                self.primary.close()
                self.notify("STT moderne indisponible. Repli System.Speech : répétez votre phrase.")
        if cancel.is_set(): raise AudioCancelled("Audio annulé.")
        return self.fallback.listen(cancel)

    def close(self):
        self.primary.close()

    def listen_interruptible(self, cancel, on_speech):
        if self.degraded:
            raise AudioError("Interruption vocale indisponible avec System.Speech ; utilisez le mode sans barge-in.")
        return self.primary.listen_interruptible(cancel, on_speech)
