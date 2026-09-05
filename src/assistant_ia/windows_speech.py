"""System.Speech local via Windows PowerShell, sans paquet Python externe.

Scripts fixes ; les données passent en base64 par l'environnement privé du
processus. Aucune transcription/réponse ne devient du code PowerShell.
"""
import base64
import json
import os
import subprocess
from math import isfinite
from threading import Event
from time import monotonic

from assistant_ia.voice import AudioCancelled, AudioError

_PRELUDE = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Speech
"""
_LISTEN = _PRELUDE + """
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine([Globalization.CultureInfo]'fr-FR')
try {
    $settings = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:FRIPOUILLE_SPEECH_TEXT)) | ConvertFrom-Json
    $engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $engine.SetInputToDefaultAudioDevice()
    $engine.BabbleTimeout = [TimeSpan]::FromSeconds(8)
    $engine.EndSilenceTimeout = [TimeSpan]::FromMilliseconds(700)
    $result = $engine.Recognize([TimeSpan]::FromSeconds($settings.initial_silence))
    if ($null -eq $result -or $result.Confidence -lt $settings.minimum_confidence) {
        @{text=$null} | ConvertTo-Json -Compress
    } else {
        @{text=$result.Text} | ConvertTo-Json -Compress
    }
} finally { $engine.Dispose() }
"""
_SPEAK = _PRELUDE + """
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet,
        [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [Globalization.CultureInfo]'fr-FR')
    $speaker.SetOutputToDefaultAudioDevice()
    $text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:FRIPOUILLE_SPEECH_TEXT))
    $speaker.Speak($text)
} finally { $speaker.Dispose() }
"""


def _run(script: str, cancel: Event, *, text: str = "", timeout: float = 20) -> str:
    if cancel.is_set():
        raise AudioCancelled("Audio annulé.")
    if os.name != "nt":
        raise AudioError("System.Speech nécessite Windows ; utilisez le texte.")
    env = os.environ.copy()
    env["FRIPOUILLE_SPEECH_TEXT"] = base64.b64encode(text.encode("utf-8")).decode("ascii")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        process = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as error:
        raise AudioError("Le moteur vocal local n'a pas pu démarrer.") from error
    deadline = monotonic() + timeout
    try:
        while True:
            if cancel.is_set():
                raise AudioCancelled("Audio annulé.")
            if monotonic() >= deadline:
                raise AudioError("Délai audio dépassé ; le texte reste disponible.")
            try:
                output, _ = process.communicate(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise AudioError("Microphone, sortie audio ou moteur français indisponible.")
        return output.decode("utf-8-sig").strip()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


class WindowsSpeechToText:
    def __init__(self, *, initial_silence: float = 10, minimum_confidence: float = .5):
        for value, low, high in ((initial_silence, 1, 15), (minimum_confidence, 0, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not low <= value <= high:
                raise ValueError("Configuration System.Speech invalide.")
        self.settings = {"initial_silence": initial_silence, "minimum_confidence": minimum_confidence}

    def listen(self, cancel: Event) -> str | None:
        try:
            result = json.loads(_run(_LISTEN, cancel, text=json.dumps(self.settings)))
            text = result["text"]
            if text is not None and (not isinstance(text, str) or len(text) > 2000):
                raise ValueError("Invalid transcription")
            return text
        except (ValueError, KeyError, TypeError) as error:
            raise AudioError("Réponse de transcription invalide.") from error


class WindowsTextToSpeech:
    def speak(self, text: str, cancel: Event) -> None:
        if not isinstance(text, str) or not text.strip():
            raise AudioError("Texte vocal vide.")
        # La réponse complète reste visible ; on borne seulement la sortie audio.
        _run(_SPEAK, cancel, text=text[:2000], timeout=120)
