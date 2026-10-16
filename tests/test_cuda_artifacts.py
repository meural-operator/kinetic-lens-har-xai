"""Distribution integrity for the trained, fixed-window CUDA artifacts."""
import hashlib
import json
from pathlib import Path
import pytest
from kinetic_lens.registry import read_config, artifact_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_published_export_matches_evidence_and_input_policy(precision):
    config = read_config(ROOT / f"configs/movinet_kard_cuda_{precision}.json")
    manifest_path = artifact_path(config, "manifest")
    manifest = json.loads(manifest_path.read_text())
    evidence = json.loads((ROOT / "docs/benchmarks/cuda/benchmark.json").read_text())
    labels = json.loads(artifact_path(config, "labels").read_text())
    weights = manifest_path.parent / manifest[precision]["file"]
    private = json.loads((ROOT / "models/private_artifacts.json").read_text())
    record = next(r for r in private["files"] if r["path"] == weights.relative_to(ROOT).as_posix())
    assert record["sha256"] == manifest[precision]["sha256"]
    assert private["visibility_requirement"] == "private"
    assert evidence["artifact_manifest"][precision]["sha256"] == manifest[precision]["sha256"]
    assert len(labels) == 18 and labels == manifest["labels"]
    assert manifest["input_shape"] == [1, 20, 172, 172, 3]
    assert config["input"] == {"fps": 5., "max_frames": 20, "resolution": 172}
    assert evidence["placement"][precision]["executed_node_providers"].keys() == {"CUDAExecutionProvider"}


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_authenticated_local_export_hash(precision):
    manifest = json.loads((ROOT / "models/kard/onnx/manifest.json").read_text())
    weights = ROOT / "models/kard/onnx" / manifest[precision]["file"]
    if not weights.is_file():
        pytest.skip("Private ONNX weights require authenticated setup")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == manifest[precision]["sha256"]
