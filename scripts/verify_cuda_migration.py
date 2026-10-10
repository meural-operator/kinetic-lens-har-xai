"""Check published CUDA adapters against original SavedModel on archived windows.

Only saved video files are read; no camera is opened or dataset redistributed.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from kinetic_lens.registry import load_model

ROOT = Path(__file__).resolve().parents[1]


def sampled_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read {path}")
        frames = []
        index = 0
        wanted = set(indices)
        while index <= indices[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        if len(frames) != len(indices):
            raise ValueError("Saved video does not contain all benchmark frame indices")
        return np.stack(frames)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cuda_migration.json")
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    archived = json.loads((ROOT / "docs/benchmarks/cuda/benchmark.json").read_text())
    manifest = json.loads((ROOT / "models/kard/onnx/manifest.json").read_text())
    import hashlib
    for filename, key in [("saved_model.pb", "source_saved_model_pb_sha256"),
                          ("variables/variables.data-00000-of-00001", "source_variables_sha256")]:
        digest = hashlib.sha256((args.experiment_dir / "saved_model" / filename).read_bytes()).hexdigest()
        if digest != manifest[key]:
            raise ValueError("Original checkpoint differs from the exported source")
    loaded = tf.saved_model.load(str(args.experiment_dir / "saved_model"))
    serving = loaded.signatures["serving_default"]
    models = {precision: load_model(ROOT / f"configs/movinet_kard_cuda_{precision}.json")
              for precision in ("fp32", "fp16")}
    rows = []
    for record in archived["comparisons"]:
        relative = record["video"].replace("\\", "/")
        frames = sampled_frames(args.dataset / relative, record["frame_indices"])
        images = tf.cast(tf.image.resize_with_pad(tf.convert_to_tensor(frames), 172, 172)[None], tf.float32) / 255.
        reference = serving(image=images)["logits"].numpy()[0]
        row = {"clip": relative, "class": record["class"], "backends": {}}
        for precision, model in models.items():
            pixel_error = float(np.max(np.abs(images.numpy() - model.images(frames).numpy())))
            logits = model.predict(frames)
            error = float(np.max(np.abs(reference - logits)))
            agreement = bool(reference.argmax() == logits.argmax())
            assert pixel_error == 0 and np.isfinite(logits).all() and agreement
            assert error < (1e-4 if precision == "fp32" else 0.1)
            row["backends"][precision] = {"pixel_max_abs_error": pixel_error,
                "max_abs_logit_error": error, "top1_agreement": agreement}
        rows.append(row)
        print("MIGRATION_CHECKED", relative, flush=True)
    result = {"sample_count": len(rows), "checks": rows, "live_camera_tested": False,
              "models": {key: model.provenance() for key, model in models.items()},
              "scope": "Published adapter/checksum/preprocessing/prediction migration; not a new accuracy evaluation or placement trace."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("CUDA_MIGRATION_PASSED", len(rows), flush=True)


if __name__ == "__main__":
    main()
