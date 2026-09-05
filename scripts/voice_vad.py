"""FRP-IA-14 : segmentation déterministe, indépendante du modèle VAD."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VadSettings:
    onset: float = .65
    offset: float = .35
    minimum_frames: int = 8  # 256 ms : ignore une impulsion brève.
    silence_frames: int = 22  # 704 ms, hystérésis de fin de phrase.
    maximum_frames: int = 938  # environ 30 s à 16 kHz / 512.

    def __post_init__(self):
        if not 0 <= self.offset < self.onset <= 1:
            raise ValueError("Seuils VAD invalides")
        if any(type(x) is not int or x < 1 for x in
               (self.minimum_frames, self.silence_frames, self.maximum_frames)):
            raise ValueError("Durées VAD invalides")
        if self.maximum_frames > 938 or self.minimum_frames >= self.maximum_frames:
            raise ValueError("Capture VAD trop longue")


class UtteranceGate:
    def __init__(self, settings=VadSettings()):
        self.settings = settings
        self.active = False
        self.high = self.low = self.frames = 0

    def accept(self, probability):
        if not isinstance(probability, (int, float)) or not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilité VAD invalide")
        cfg = self.settings
        if not self.active:
            self.high = self.high + 1 if probability >= cfg.onset else 0
            if self.high >= cfg.minimum_frames:
                self.active = True
                self.frames = self.high
                return "started"
        else:
            self.frames += 1
            self.low = self.low + 1 if probability < cfg.offset else 0
            if self.low >= cfg.silence_frames or self.frames >= cfg.maximum_frames:
                return "ended"
        return None
