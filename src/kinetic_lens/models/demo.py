"""Deterministic plumbing fixture, explicitly NOT a trained HAR model."""
from dataclasses import asdict
import numpy as np
from ..contracts import InputSpec, validate_clip


class DemoModel:
    name = "Synthetic RGB fixture — not HAR"
    labels = ["Red channel", "Green channel", "Blue channel"]
    capabilities = frozenset({"predict"})

    def __init__(self, config):
        self.input_spec = InputSpec(**config.get("input", {"resolution": 64, "max_frames": 20}))

    def predict(self, frames):
        return validate_clip(frames, self.input_spec).mean(axis=(0,1,2))/32

    def provenance(self):
        return {"name": self.name, "trained": False, "runtime": "NumPy CPU", "input": asdict(self.input_spec)}
