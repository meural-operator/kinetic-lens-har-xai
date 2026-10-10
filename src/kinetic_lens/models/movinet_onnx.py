"""Fixed-window MoViNet/KARD CUDA inference with the verified pixel pipeline."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from ..contracts import InputSpec, validate_clip, validate_logits
from ..registry import artifact_path


class MoViNetONNXModel:
    capabilities = frozenset({"predict"})

    def __init__(self, config):
        import onnxruntime as ort
        import tensorflow as tf
        self.tf = tf
        self.input_spec = InputSpec(**config["input"])
        self.name = config.get("name", "MoViNet KARD CUDA")
        self.precision = config.get("precision", "fp16")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        manifest_path = artifact_path(config, "manifest")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.labels = json.loads(artifact_path(config, "labels").read_text(encoding="utf-8-sig"))
        if self.labels != self.manifest["labels"]:
            raise ValueError("ONNX label order differs from manifest")
        size = self.input_spec.resolution
        self.shape = [1, self.input_spec.max_frames, size, size, 3]
        if self.shape != self.manifest["input_shape"] or self.input_spec.fps != self.manifest["fps"]:
            raise ValueError("Input policy differs from the fixed-window export")
        artifact = self.manifest[self.precision]
        self.path = (manifest_path.parent / artifact["file"]).resolve()
        if self.path.parent != manifest_path.parent:
            raise ValueError("ONNX artifact must reside beside its manifest")
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.hash != artifact["sha256"]:
            raise ValueError("ONNX artifact checksum mismatch")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Install official onnxruntime-gpu with compatible CUDA/cuDNN dependencies")
        # ORT searches installed PyTorch and NVIDIA runtime wheels for CUDA DLLs.
        # Torch is used only to locate libraries when available, never for inference.
        try:
            import torch
        except ImportError:
            ort.preload_dlls()
        else:
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.provider_options = {"device_id": 0, "use_tf32": "1" if self.precision == "fp16" else "0",
                                 "cudnn_conv_algo_search": "HEURISTIC"}
        self.session = ort.InferenceSession(str(self.path), options,
            providers=[("CUDAExecutionProvider", self.provider_options)])
        self.session.disable_fallback()
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA provider did not load")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != self.shape:
            raise ValueError("ONNX input contract does not match manifest")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].shape != [1, len(self.labels)]:
            raise ValueError("ONNX output contract does not match labels")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.version = ort.__version__

    def images(self, frames):
        frames = validate_clip(frames, self.input_spec)
        if len(frames) != self.shape[1]:
            raise ValueError(f"This export requires exactly {self.shape[1]} sampled frames; do not pad")
        tf = self.tf
        with tf.device("/CPU:0"):
            return tf.cast(tf.image.resize_with_pad(tf.convert_to_tensor(frames),
                self.shape[2], self.shape[3])[None], tf.float32) / 255.

    def predict(self, frames):
        array = np.ascontiguousarray(self.images(frames).numpy(), dtype=np.float32)
        values = self.session.run([self.output_name], {self.input_name: array})[0]
        if values.shape != (1, len(self.labels)):
            raise ValueError("ONNX output must be [1,num_classes] logits")
        return validate_logits(values[0], self.labels)

    def provenance(self):
        return {"name": self.name, "runtime": "ONNX Runtime CUDA", "precision": self.precision,
                "onnxruntime_version": self.version, "sha256": self.hash,
                "providers": self.session.get_providers(), "provider_options_requested": self.provider_options,
                "cpu_execution_fallback": False, "input": asdict(self.input_spec),
                "preprocessing": "RGB; original TensorFlow CPU bilinear resize_with_pad; float32 [0,1]",
                "state_policy": "fresh initial states per independent fixed window",
                "source_saved_model_pb_sha256": self.manifest["source_saved_model_pb_sha256"],
                "source_variables_sha256": self.manifest["source_variables_sha256"],
                "provider_note": "Registered providers are not a placement audit; saved trial profiles are documented separately"}
