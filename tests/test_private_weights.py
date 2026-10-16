import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from kinetic_lens.weights import ensure_artifacts, load_manifest


def fixture_manifest(root, payload=b"learned parameters"):
    entry = {"path": "models/kard/head.bin", "hf_path": "kinetic-lens/models/kard/head.bin",
             "group": "default", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {"schema_version": 1, "repo_id": "owner/private", "repo_type": "model",
                "revision": "a" * 40, "visibility_requirement": "private", "files": [entry]}
    (root / "models").mkdir(parents=True)
    (root / "models/private_artifacts.json").write_text(json.dumps(manifest))
    return manifest, entry, payload


def test_verify_only_does_not_download_and_rejects_mismatch(tmp_path):
    _, entry, payload = fixture_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        ensure_artifacts(tmp_path, verify_only=True)
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    assert ensure_artifacts(tmp_path, verify_only=True)["verified_files"] == 1
    target.write_bytes(b"unexpected weights")
    with pytest.raises(ValueError, match="unexpected hash"):
        ensure_artifacts(tmp_path)
    assert target.read_bytes() == b"unexpected weights"


@pytest.mark.parametrize("field,value", [("path", "../outside.bin"), ("hf_path", "../outside.bin"),
                                       ("path", "C:\\outside.bin")])
def test_manifest_rejects_path_escape(tmp_path, field, value):
    manifest, _, _ = fixture_manifest(tmp_path)
    manifest["files"][0][field] = value
    (tmp_path / "models/private_artifacts.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(tmp_path)


def test_manifest_requires_commit_pin(tmp_path):
    manifest, _, _ = fixture_manifest(tmp_path)
    manifest["revision"] = "main"
    (tmp_path / "models/private_artifacts.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full commit"):
        load_manifest(tmp_path)


def test_verified_local_backup_copy(tmp_path):
    root = tmp_path / "repo"
    _, entry, payload = fixture_manifest(root)
    source = tmp_path / "private_weight_migration/backup" / entry["path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    result = ensure_artifacts(root, from_workspace=tmp_path)
    assert result["verified_files"] == 1
    assert (root / entry["path"]).read_bytes() == payload


@pytest.mark.parametrize("private", [True, False])
def test_download_checks_privacy_revision_and_hash(tmp_path, monkeypatch, private):
    manifest, entry, payload = fixture_manifest(tmp_path)
    cached = tmp_path / "download.bin"
    cached.write_bytes(payload)
    calls = []
    class Api:
        def model_info(self, repo, revision):
            assert repo == manifest["repo_id"] and revision == manifest["revision"]
            return SimpleNamespace(private=private, sha=revision)
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == manifest["revision"] and kwargs["filename"] == entry["hf_path"]
        return str(cached)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api, hf_hub_download=download))
    if private:
        assert ensure_artifacts(tmp_path)["verified_files"] == 1
        assert (tmp_path / entry["path"]).read_bytes() == payload and len(calls) == 1
    else:
        with pytest.raises(RuntimeError, match="non-private"):
            ensure_artifacts(tmp_path)
        assert not calls and not (tmp_path / entry["path"]).exists()
