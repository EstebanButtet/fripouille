"""FRP-IA-14 : worker Python 3.10, STT/VAD local, IPC sans autorité métier."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic

from voice_vad import UtteranceGate

MAX_LINE = 16384


class Cancelled(Exception):
    pass


class SpeechEngine:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.vad = None

    def load(self, download=False):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.args.model, device=self.args.device,
                compute_type=self.args.compute_type, cpu_threads=4, num_workers=1,
                download_root=str(self.args.cache), local_files_only=not download)

    def transcribe(self, audio, cancel):
        if cancel.is_set(): raise Cancelled()
        if len(audio) > 480256: raise ValueError("Audio supérieur à 30 secondes")
        self.load()
        start = monotonic()
        segments, _ = self.model.transcribe(audio, language="fr", beam_size=5,
            condition_on_previous_text=False, vad_filter=False)
        parts = []
        size = 0
        for segment in segments:
            if cancel.is_set(): raise Cancelled()
            size += len(segment.text)
            if size > 2000: raise ValueError("Transcription trop longue")
            parts.append(segment.text)
        if cancel.is_set(): raise Cancelled()
        return " ".join(parts).strip() or None, (monotonic()-start)*1000

    def listen(self, cancel, emit, microphone=None):
        import numpy as np
        import sounddevice as sd
        from faster_whisper.vad import get_vad_model
        if self.vad is None: self.vad = get_vad_model()
        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        context = np.zeros((1, 64), dtype="float32")
        gate, preroll, chunks = UtteranceGate(), deque(maxlen=10), []
        start = monotonic()
        last_voice = start
        vad_ms = 0
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=512, device=microphone) as stream:
            while True:
                if cancel.is_set(): raise Cancelled()
                if not gate.active and monotonic()-start > 15:
                    return {"text": None, "metrics": {"vad_ms": vad_ms}}
                audio, overflow = stream.read(512)
                if overflow: raise RuntimeError("Capture audio saturée ; répétez")
                frame = audio[:, 0].copy()
                tick = monotonic()
                output, h, c = self.vad.session.run(None,
                    {"input": np.concatenate((context, frame[None, :]), axis=1), "h": h, "c": c})
                context = frame[None, -64:].copy()
                probability = float(output.reshape(-1)[0])
                vad_ms += (monotonic()-tick)*1000
                was_active = gate.active
                event = gate.accept(probability)
                if not was_active: preroll.append(frame)
                else: chunks.append(frame)
                if probability >= gate.settings.offset: last_voice = monotonic()
                if event == "started":
                    chunks.extend(preroll)
                    emit({"event": "speech_started"})
                if event == "ended": break
        # Le micro est fermé avant STT. Aucun enregistrement sur disque.
        text, stt_ms = self.transcribe(np.concatenate(chunks)[:480000], cancel)
        return {"text": text, "metrics": {"stt_ms": stt_ms, "vad_ms": vad_ms,
            "end_speech_to_transcript_ms": (monotonic()-last_voice)*1000}}

    def run(self, request, cancel, emit):
        operation = request.get("op")
        if operation == "warmup":
            self.load()
            return {"text": None}
        if operation == "listen":
            microphone = request.get("microphone")
            if microphone is not None and (type(microphone) is not int or microphone < 0):
                raise ValueError("Microphone invalide")
            return self.listen(cancel, emit, microphone)
        if operation == "transcribe_file":
            import soundfile as sf
            path = Path(request["path"])
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames > 480000:
                raise ValueError("WAV mono 16 kHz de 30 s maximum requis")
            audio, _ = sf.read(path, dtype="float32")
            text, elapsed = self.transcribe(audio, cancel)
            return {"text": text, "metrics": {"stt_ms": elapsed}}
        raise ValueError("Opération vocale inconnue")


def serve(engine):
    output_lock, state_lock = Lock(), Lock()
    current = None
    def send(identifier, values):
        raw = json.dumps({"id": identifier, **values}, ensure_ascii=False) + "\n"
        if len(raw.encode("utf-8")) > MAX_LINE: raise ValueError("Réponse trop grande")
        with output_lock:
            sys.stdout.buffer.write(raw.encode("utf-8"))
            sys.stdout.buffer.flush()
    def execute(request, cancel):
        nonlocal current
        identifier = request["id"]
        try:
            result = engine.run(request, cancel, lambda value: send(identifier, value))
            result = {"status": "cancelled"} if cancel.is_set() else {"status": "ok", **result}
        except Cancelled: result = {"status": "cancelled"}
        except Exception as error:
            result = {"status": "error", "error": type(error).__name__ + ": " + str(error)[:200]}
        # Libérer l'état avant d'annoncer la fin au parent.
        with state_lock:
            current = None
            send(identifier, result)
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line or len(line) > MAX_LINE: break
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int: break
            with state_lock:
                if request.get("op") == "cancel":
                    if current is not None and current[0] == request["id"]: current[1].set()
                    continue
                if current is not None:
                    send(request["id"], {"status": "error", "error": "busy"})
                    continue
                cancel = Event()
                current = (request["id"], cancel)
                Thread(target=execute, args=(request, cancel), daemon=True).start()
        except (ValueError, TypeError, OSError): break
    if current is not None: current[1].set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small", choices=("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="int8", choices=("int8", "float32", "float16", "int8_float16"))
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parents[1] / "models/whisper")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    engine = SpeechEngine(args)
    if args.prepare:
        engine.load(download=True)
        print("Modèle vocal prêt.")
    else:
        # Windows/Python 3.10 : initialiser les DLL numériques sur le thread
        # principal avant qu'il attende stdin (import NumPy sinon bloquant).
        import numpy
        import faster_whisper
        import sounddevice
        serve(engine)


if __name__ == "__main__": main()
