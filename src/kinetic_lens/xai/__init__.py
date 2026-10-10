"""Each method declares required adapter capabilities and interpretation limits."""
from ..contracts import validate_clip, validate_logits


def check(model, frames, requires, target=None):
    missing = set(requires)-set(model.capabilities)
    if missing: raise ValueError(f"Model lacks explanation capabilities: {sorted(missing)}")
    frames = validate_clip(frames, model.input_spec)
    logits = validate_logits(model.predict(frames), model.labels)
    target = int(logits.argmax()) if target is None else int(target)
    if not 0 <= target < len(model.labels): raise ValueError("Target class is out of range")
    return frames, logits, target


def margin(logits, target):
    import numpy as np
    return float(logits[target]-np.max(np.delete(logits,target)))
