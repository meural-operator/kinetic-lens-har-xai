"""Deterministic offline sampling and bounded, timestamp-aware live buffers."""
from collections import deque
from pathlib import Path
import math
import cv2
import numpy as np


def sample_indices(source_fps, count, spec):
    if not np.isfinite(source_fps) or source_fps <= 0 or count <= 0:
        raise ValueError("Video must have a valid frame rate and nonempty frame count")
    if source_fps < spec.fps:
        raise ValueError("Source fps is below required sampling fps; do not duplicate frames silently")
    start = max(0., (count/source_fps - spec.max_frames/spec.fps)/2)
    targets = np.arange(start, count/source_fps, 1/spec.fps)[:spec.max_frames]
    ids = np.ceil(targets*source_fps-1e-8).astype(int)
    return ids[ids < count]


def read_video(path, spec):
    cap = cv2.VideoCapture(str(Path(path)))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sample_indices(fps, count, spec)
        frames = []; index = 0; target = 0
        while target < len(wanted):
            ok, frame = cap.read()
            if not ok: break
            if index == wanted[target]:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); target += 1
            index += 1
        if len(frames) != len(wanted) or not frames:
            raise ValueError("Video decoding ended before selected frames")
        return np.stack(frames), {"frame_indices": wanted.tolist(), "source_fps": fps,
                                  "sampled_frames": len(frames), "target_fps": spec.fps}
    finally:
        cap.release()


class RollingWindow:
    def __init__(self, spec, window_seconds):
        self.fps = spec.fps
        self.capacity = int(round(window_seconds*spec.fps))
        if not 2 <= self.capacity <= spec.max_frames:
            raise ValueError("Window must contain 2..max_frames samples")
        self.items = deque(maxlen=self.capacity)
        self.last_received = None
        self.next_sample = None
        self.shape = None

    def clear(self):
        self.items.clear(); self.last_received = self.next_sample = self.shape = None

    def add(self, timestamp, frame):
        if not math.isfinite(timestamp) or (self.last_received is not None and timestamp <= self.last_received):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        if self.shape is not None and (self.shape != frame.shape or timestamp-self.last_received > 2/self.fps):
            self.items.clear(); self.next_sample = None
        self.shape = frame.shape; self.last_received = timestamp
        if self.next_sample is None: self.next_sample = timestamp
        if timestamp+1e-7 >= self.next_sample:
            self.items.append((timestamp, frame.copy()))
            self.next_sample += (int((timestamp-self.next_sample)*self.fps+1e-7)+1)/self.fps
            return True
        return False

    def snapshot(self):
        if len(self.items) < self.capacity: return None
        return np.stack([x[1] for x in self.items]), [x[0] for x in self.items]
