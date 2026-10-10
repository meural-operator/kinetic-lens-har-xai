"""The small shared boundary between models, explanations and the dashboard."""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class InputSpec:
    fps: float = 5.0
    max_frames: int = 50
    resolution: int = 172

    def __post_init__(self):
        if type(self.max_frames) is not int or type(self.resolution) is not int or isinstance(self.fps,bool):
            raise ValueError("max_frames/resolution must be integers, fps must be numeric")
        if not np.isfinite(self.fps) or not 0 < self.fps <= 120:
            raise ValueError("fps must be finite and in (0,120]")
        if not 2 <= self.max_frames <= 256 or not 8 <= self.resolution <= 1024:
            raise ValueError("Invalid frame count or input resolution")


def validate_clip(frames, spec):
    """Adapters receive sampled RGB uint8 [T,H,W,3]; resizing is adapter-owned."""
    x = np.asarray(frames)
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 [T,H,W,3]")
    if not 1 <= len(x) <= spec.max_frames or min(x.shape[1:3]) < 1:
        raise ValueError("Empty clip or clip exceeds this model's max_frames")
    return x


def validate_logits(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (len(labels),) or len(labels) < 2 or not np.isfinite(z).all():
        raise ValueError("Adapter must return finite raw logits, one per label")
    return z


def prediction(logits, labels):
    z = validate_logits(logits, labels)
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)
    return {"class_id": int(order[0]), "label": labels[order[0]], "logits": z.tolist(),
            "top3": [{"class_id": int(i), "label": labels[i], "score": float(p[i])}
                     for i in order[:3]], "scores_calibrated": False}


@dataclass
class Explanation:
    method: str
    target: int
    data: dict
    limitations: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"method": self.method, "target": self.target, "data": self.data,
                "limitations": self.limitations}


class HARModel(Protocol):
    name: str
    labels: list[str]
    input_spec: InputSpec
    capabilities: frozenset[str]

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Independent clips: reset recurrent/streaming state on every call."""

    def provenance(self) -> dict:
        """Identity, artifacts, preprocessing and actual execution device."""
