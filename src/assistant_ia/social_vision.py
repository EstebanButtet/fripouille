"""FRP-IA-11 : perceptions géométriques éphémères, sans identité sociale.

Le producteur est applicatif : ce module ne reçoit ni sortie LLM, ni image.
Il n'ouvre pas implicitement une caméra. Les timestamps utilisent l'horloge
monotone locale ; un adaptateur externe doit convertir son temps de capture.
"""
from dataclasses import dataclass
from collections.abc import Callable
from math import isfinite
from threading import RLock
from time import monotonic
from typing import Protocol

PERCEPTION_TTL = 2.0


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("A finite numeric geometry value is required.")
    if not isfinite(value) or not low <= value <= high:
        raise ValueError("Geometry value outside its allowed range.")


@dataclass(frozen=True, slots=True)
class FaceDetection:
    center_x: float
    center_y: float
    width: float
    height: float
    yaw_degrees: float | None = None

    def __post_init__(self):
        for value in (self.center_x, self.center_y, self.width, self.height):
            _number(value, 0, 1)
        if self.width == 0 or self.height == 0:
            raise ValueError("Face bounds must have a positive size.")
        if self.yaw_degrees is not None:
            _number(self.yaw_degrees, -90, 90)


@dataclass(frozen=True, slots=True)
class VisionFrame:
    captured_at: float
    detection: FaceDetection | None

    def __post_init__(self):
        _number(self.captured_at, 0, float("inf"))
        if self.detection is not None and not isinstance(self.detection, FaceDetection):
            raise TypeError("Frame requires a validated detection or explicit absence.")


@dataclass(frozen=True, slots=True)
class SocialPerception:
    status: str = "unavailable"
    event: str | None = None
    track_id: int | None = None
    position: str | None = None
    orientation: str | None = None
    captured_at: float | None = None


class SocialVisionProvider(Protocol):
    def read(self) -> VisionFrame:
        """Retourner au plus un visage sélectionné, appel obligatoirement borné."""
        ...

    def close(self) -> None:
        """Libérer capture et ressources."""
        ...


class SocialVisionService:
    """Pas de boucle autonome ni persistance. Une piste n'est jamais Person.

    Le tracking V1 ne promet que la continuité géométrique d'un visage unique.
    Changement de position brutal, absence ou expiration créent une autre piste.
    Erreur caméra et absence de visage sont deux résultats différents.
    """
    def __init__(self, *, clock: Callable[[], float] = monotonic):
        self._clock, self._lock = clock, RLock()
        self._enabled = False
        self._snapshot = SocialPerception()
        self._last_frame = -1.0
        self._detection = None
        self._next_track = 0

    def start(self) -> None:
        with self._lock:
            self._enabled = True
            self._snapshot = SocialPerception()
            self._last_frame, self._detection = -1.0, None

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            self._snapshot, self._detection = SocialPerception(), None

    @property
    def snapshot(self) -> SocialPerception:
        with self._lock:
            if (self._snapshot.captured_at is not None
                    and self._clock() - self._snapshot.captured_at >= PERCEPTION_TTL):
                self._snapshot, self._detection = SocialPerception("expired"), None
            return self._snapshot

    def accept(self, frame: VisionFrame) -> bool:
        if not isinstance(frame, VisionFrame):
            raise TypeError("Only application vision frames are accepted.")
        with self._lock:
            if not self._enabled:
                return False
            age = self._clock() - frame.captured_at
            if not 0 <= age < PERCEPTION_TTL or frame.captured_at <= self._last_frame:
                return False
            old = self.snapshot
            detection = frame.detection
            if detection is None:
                new = SocialPerception("absent", "departed" if old.status == "present" else None,
                                       captured_at=frame.captured_at)
            else:
                previous = self._detection
                continuous = (old.status == "present" and previous is not None
                              and abs(previous.center_x - detection.center_x) <= .25
                              and abs(previous.center_y - detection.center_y) <= .25)
                if not continuous:
                    self._next_track += 1
                position = "left" if detection.center_x < .35 else "right" if detection.center_x > .65 else "center"
                yaw = detection.yaw_degrees
                orientation = "unknown" if yaw is None else "approximately_frontal" if abs(yaw) <= 20 else "turned"
                new = SocialPerception("present", "continued" if continuous else "arrived",
                                       self._next_track, position, orientation, frame.captured_at)
            self._snapshot, self._detection = new, detection
            self._last_frame = frame.captured_at
            return True

    def poll(self, provider: SocialVisionProvider) -> bool:
        with self._lock:
            if not self._enabled:
                return False
        try:
            return self.accept(provider.read())
        except (OSError, ValueError, TypeError):
            with self._lock:
                self._snapshot, self._detection = SocialPerception(), None
            return False
