"""The original desktop camera preview, using portable KineticLens configs."""
import argparse
import json
from pathlib import Path
import queue
import time
import cv2
import numpy as np
from kinetic_lens.contracts import prediction
from kinetic_lens.registry import load_model
from live_preview import live, put_latest

ROOT = Path(__file__).resolve().parents[2]


class WindowClassifier:
    def __init__(self, config):
        self.config = Path(config).resolve()
        self.model = load_model(self.config)
        self.labels = self.model.labels
        spec = self.model.input_spec
        self.protocol = {"target_fps": spec.fps, "max_frames": spec.max_frames, "resolution": spec.resolution}
        info = self.model.provenance()
        self.runtime_name = info["runtime"] + " " + info.get("precision", "").upper()

    def classify(self, frames):
        start = time.perf_counter()
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
        logits = self.model.predict(rgb)
        result = prediction(logits, self.labels)
        return {"top3": result["top3"], "processing_ms": 1000 * (time.perf_counter() - start),
                "sampled_frames": len(frames)}, logits

    def warmup(self, count):
        if hasattr(self.model, "shape") and count != self.model.shape[1]:
            raise ValueError(f"ONNX export requires exactly {self.model.shape[1]} sampled frames")
        blank = np.zeros((172, 172, 3), np.uint8)
        for _ in range(2):
            self.classify([blank] * count)


def benchmark(model, video, window_seconds, update_seconds, output):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    fps = model.protocol["target_fps"]
    if not frames or not np.isfinite(source_fps) or source_fps < fps:
        raise ValueError("Video needs a valid frame rate at or above the sampling rate")
    count = int(round(window_seconds * fps))
    stride = int(round(update_seconds * fps))
    ids = np.ceil(np.arange(0, len(frames) / source_fps, 1 / fps) * source_fps - 1e-8).astype(int)
    sampled = [frames[i] for i in ids if i < len(frames)]
    if len(sampled) < count:
        raise ValueError("Video is shorter than the analysis window; no padding is used")
    windows = [sampled[j:j + count] for j in range(0, len(sampled) - count + 1, stride)]
    model.warmup(count)
    timings = []
    predictions = []
    for repeat in range(3):
        for i, window in enumerate(windows):
            result, _ = model.classify(window)
            timings.append(result["processing_ms"])
            if repeat == 0:
                predictions.append({"window": i, **result})
    _, a = model.classify(windows[0])
    _, b = model.classify(windows[0])
    error = float(np.max(np.abs(a - b)))
    if error >= 1e-6:
        raise RuntimeError(f"State-reset repeat mismatch: {error}")
    mailbox = queue.Queue(maxsize=1)
    put_latest(mailbox, "old")
    put_latest(mailbox, "latest")
    assert mailbox.qsize() == 1 and mailbox.get_nowait() == "latest"
    result = {"model": model.model.provenance(), "target_fps": fps, "window_seconds": window_seconds,
              "measurements": len(timings), "mean_ms": float(np.mean(timings)),
              "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
              "max_ms": float(np.max(timings)), "repeat_after_state_reset_max_logit_error": error,
              "bounded_latest_mailbox_verified": True, "predictions": predictions, "live_camera_tested": False,
              "scope": "BGR conversion, stacking, adapter preprocessing, CUDA transfers and prediction formatting; excludes decode, camera/display and cold loading."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("BENCHMARK_COMPLETE", json.dumps({k: v for k, v in result.items() if k != "predictions"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/movinet_kard_cuda_fp16.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=4.)
    parser.add_argument("--update-seconds", type=float, default=1.)
    parser.add_argument("--duration-seconds", type=float, default=0.)
    parser.add_argument("--benchmark-video", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/desktop_benchmark.json")
    args = parser.parse_args()
    model = WindowClassifier(args.config)
    fps = model.protocol["target_fps"]
    if not 0 < args.window_seconds <= model.protocol["max_frames"] / fps:
        parser.error("window exceeds the configured frame count")
    if args.update_seconds <= 0 or args.duration_seconds < 0:
        parser.error("update must be positive and duration nonnegative")
    if int(round(args.window_seconds * fps)) < 2 or int(round(args.update_seconds * fps)) < 1:
        parser.error("window/update must span sufficient samples")
    if args.benchmark_video:
        benchmark(model, args.benchmark_video, args.window_seconds, args.update_seconds, args.output)
    else:
        live(model, args.camera, args.window_seconds, args.update_seconds, args.duration_seconds)


if __name__ == "__main__":
    main()
