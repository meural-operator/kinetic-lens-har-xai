"""Official A0-Stream backbone plus the verified KARD affine classifier.

Explanation hooks use restored internal layers and verify replay equality.
These hooks are specific to this archived backbone; other models use adapters.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from ..contracts import InputSpec, validate_clip, validate_logits
from ..registry import artifact_path


class MoViNetModel:
    capabilities = frozenset({"predict", "linear_head", "spatial_gradients", "channels"})

    def __init__(self, config):
        import tensorflow as tf
        self.tf = tf
        self.input_spec = InputSpec(**config["input"])
        if self.input_spec.resolution != 172:
            raise ValueError("This A0 adapter requires resolution 172")
        self.name = config.get("name", "MoViNet A0 + KARD head")
        self.labels = json.loads(artifact_path(config, "labels").read_text(encoding="utf-8-sig"))
        self.backbone_path = artifact_path(config, "backbone")
        self.head_path = artifact_path(config, "head")
        self.xai_assets = {k: (config["_base"] / v).resolve() for k,v in config.get("xai_assets",{}).items()}
        self.hashes = {}
        for path, expected in [(self.head_path, config.get("head_sha256")),
                               (self.backbone_path/"variables/variables.data-00000-of-00001", config.get("backbone_sha256"))]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing model artifact {path.name}. Run python scripts/setup_models.py")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected and actual != expected: raise ValueError(f"Model hash mismatch: {path.name}")
            self.hashes[path.name] = actual
        with np.load(self.head_path, allow_pickle=False) as a:
            self.mean = a["mean"].astype(np.float32); self.scale = a["scale"].astype(np.float32)
            self.weights = a["weights"].astype(np.float32); self.bias = a["bias"].astype(np.float32)
        if self.weights.shape != (480,len(self.labels)) or self.mean.shape != (480,) or self.scale.shape != (480,) or self.bias.shape != (len(self.labels),):
            raise ValueError("Invalid MoViNet linear head dimensions")
        if not all(np.isfinite(x).all() for x in (self.weights,self.bias,self.mean,self.scale)) or np.any(self.scale <= 0):
            raise ValueError("Invalid head values or normalization scale")
        self.loaded = tf.saved_model.load(str(self.backbone_path))
        self.backbone = self.loaded._backbone
        self.state_shape = tf.constant([1,1,172,172,3])
        self.project = getattr(self.backbone, "layer_with_weights-16")
        self.channel_layers = [getattr(self.backbone, f"layer_with_weights-{i}") for i in (14,15,16)]
        if not self.project.variables[0].name.startswith("head/project/"):
            raise ValueError("Backbone internal hook layout does not match this adapter")
        self._run = tf.function(self._forward, input_signature=[tf.TensorSpec([1,None,172,172,3],tf.float32)])
        self._channel_run = tf.function(self._channel_forward, input_signature=[tf.TensorSpec([1,None,6,6,104],tf.float32)])

    def images(self, frames):
        tf = self.tf; x = validate_clip(frames, self.input_spec)
        return tf.cast(tf.image.resize_with_pad(tf.convert_to_tensor(x),172,172)[None],tf.float32)/255.

    def _forward(self, images):
        endpoints, _ = self.backbone({**self.loaded.init_states(self.state_shape), "image": images})
        return endpoints

    def _head(self, raw):
        tf = self.tf
        return tf.matmul((raw-self.mean)/self.scale, self.weights)+self.bias

    def linear_evidence(self, frames):
        raw = self.tf.reshape(self._run(self.images(frames))["head"],[1,480])
        return {"features": ((raw.numpy()[0]-self.mean)/self.scale).astype(np.float64),
                "weights": self.weights.astype(np.float64), "bias": self.bias.astype(np.float64),
                "logits": self._head(raw).numpy()[0]}

    def predict(self, frames):
        raw = self.tf.reshape(self._run(self.images(frames))["head"],[1,480])
        return validate_logits(self._head(raw).numpy()[0],self.labels)

    def spatial_map(self, frames, target, rival=None):
        tf = self.tf; ep = self._run(self.images(frames)); activation = tf.identity(ep["b4/l3"])
        full = self._head(tf.reshape(ep["head"],[1,480]))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(activation)
            pooled, _ = self.project(activation,states=self.loaded.init_states(self.state_shape),training=False)
            logits = self._head(tf.reshape(pooled,[1,480]))
            objective = logits[0,target] if rival is None else logits[0,target]-logits[0,rival]
        gradient = tape.gradient(objective,activation)
        if gradient is None: raise RuntimeError("Disconnected spatial gradient")
        error = float(tf.reduce_max(tf.abs(logits-full)))
        if error > 1e-4: raise RuntimeError(f"Spatial replay does not match inference: {error}")
        signed = tf.reduce_sum(activation*tf.reduce_mean(gradient,axis=[1,2,3],keepdims=True),axis=-1)[0].numpy()
        if not np.isfinite(signed).all(): raise RuntimeError("Non-finite attribution")
        return signed, full.numpy()[0], error

    def channel_activation(self, frames):
        ep = self._run(self.images(frames))
        return ep["b4/l1"].numpy()[0], self._head(self.tf.reshape(ep["head"],[1,480])).numpy()[0]

    def _channel_forward(self, activation):
        states = self.loaded.init_states(self.state_shape)
        for layer in self.channel_layers: activation, states = layer(activation,states=states,training=False)
        return self._head(self.tf.reshape(activation,[1,480]))

    def channel_predict(self, activation):
        x = np.asarray(activation,dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (6,6,104) or not np.isfinite(x).all():
            raise ValueError("Expected finite [T,6,6,104] activation")
        return self._channel_run(x[None]).numpy()[0]

    def provenance(self):
        return {"name": self.name, "runtime": "TensorFlow", "tensorflow_version": self.tf.__version__,
                "devices": [x.name for x in self.tf.config.list_logical_devices()], "hashes": self.hashes,
                "input": asdict(self.input_spec), "state_policy": "fresh state per independent clip/window",
                "preprocessing": "RGB; TensorFlow bilinear resize_with_pad; float32 [0,1]",
                "trained": True, "training_scope": "KARD head; official pretrained backbone frozen"}
