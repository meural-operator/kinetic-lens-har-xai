"""Simple registries; third-party plugins use module:Class config entries."""
import importlib
from pathlib import Path
import json
from .contracts import InputSpec


def resolve(name, builtins):
    path = builtins.get(name, name)
    if ":" not in path:
        raise ValueError(f"Unknown plugin {name!r}; use a registered name or module:Class")
    module, attribute = path.split(":", 1)
    return getattr(importlib.import_module(module), attribute)


MODELS = {
    "movinet": "kinetic_lens.models.movinet:MoViNetModel",
    "onnx": "kinetic_lens.models.onnx:ONNXModel",
    "torchscript": "kinetic_lens.models.torchscript:TorchScriptModel",
    "demo": "kinetic_lens.models.demo:DemoModel",
}
METHODS = {
    "spatial": "kinetic_lens.xai.spatial:SpatialAttribution",
    "temporal": "kinetic_lens.xai.temporal:TemporalAttribution",
    "concept": "kinetic_lens.xai.concept:ConceptAttribution",
    "contrastive": "kinetic_lens.xai.contrastive:ContrastiveExplanation",
    "mechanistic": "kinetic_lens.xai.mechanistic:ChannelIntervention",
}


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    cfg["_base"] = path.parent
    return cfg


def load_model(config):
    cfg = read_config(config) if isinstance(config, (str, Path)) else dict(config)
    cfg.setdefault("_base", Path.cwd())
    cls = resolve(cfg["adapter"], MODELS)
    model = cls(cfg)
    if not isinstance(model.input_spec, InputSpec) or len(model.labels) < 2:
        raise ValueError("Invalid adapter input specification or labels")
    if not all(isinstance(x,str) and x.strip() for x in model.labels) or len(set(model.labels))!=len(model.labels):
        raise ValueError("Labels must be unique nonempty strings in output order")
    return model


def make_method(name):
    return resolve(name, METHODS)()


def methods_for(model):
    return [{"id": key, "available": set(make_method(key).requires) <= set(model.capabilities),
             "requires": sorted(make_method(key).requires)} for key in METHODS]


def artifact_path(config, key):
    p = Path(config[key]).expanduser()
    return (config["_base"] / p).resolve() if not p.is_absolute() else p.resolve()
