"""Reproduce the portable trained head/XAI assets from an audited local workspace.

Requires TensorFlow and the previous research artifacts. Normal users only need
authenticated setup_models.py. Generated learned files are ignored by Git and
must be uploaded privately with a reviewed revision/hash manifest when updated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--workspace",type=Path,required=True); args=p.parse_args()
    import tensorflow as tf
    base=args.workspace/"movinet_evaluation"; model=base/"kard_a0_head_100pct"
    source_sha=sha(model/"kard_head.keras")
    if source_sha!="e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70": raise ValueError("Expected the selected original head")
    head=tf.keras.models.load_model(model/"kard_head.keras"); norm=head.get_layer("train_only_standardization")
    mean=norm.mean.numpy().ravel(); scale=np.maximum(np.sqrt(norm.variance.numpy().ravel()),tf.keras.backend.epsilon())
    weights,bias=head.get_layer("kard_classifier").get_weights()
    destination=ROOT/"models/kard"; destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/"linear_head.npz",mean=mean,scale=scale,weights=weights,bias=bias)
    converted_sha=sha(destination/"linear_head.npz")
    (destination/"labels.json").write_text((model/"labels.json").read_text(),encoding="utf-8")
    probe=np.load(base/"kard_concept_xai/probe_directions.npz",allow_pickle=False)
    directions=probe["directions"]
    if directions.shape==(480,4): directions=directions.T
    if directions.shape!=(4,480): raise ValueError("Unexpected concept direction dimensions")
    np.savez_compressed(destination/"concept_probes.npz",directions=directions,names=probe["proxy_names"].astype(str),head_sha256=np.array(converted_sha))
    selection=json.loads((base/"kard_mechanistic_xai_extended/selection.json").read_text())
    selection["head_sha256"]=converted_sha
    selection["source_selection_sha256"]=sha(base/"kard_mechanistic_xai_extended/selection.json")
    (destination/"channel_groups.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    provenance={"source_head_sha256":source_sha,"portable_head_sha256":converted_sha,
        "conversion":"Lossless export of normalization, Dense weights and bias; dropout disabled; no retraining or quantization",
        "backbone_data_sha256":"f35edf9b7b747b7fd5b65f127a34c988bec7c6602b1618af123b970d5847d6d2",
        "backbone_archive_sha256":"2453a3d224b8afb6503d845a4220373f7972575e23c9681aaded28b5a12f611f",
        "backbone_handle":"https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification/3",
        "backbone_reference":"https://github.com/tensorflow/models/tree/master/official/projects/movinet",
        "training_clips":324,"validation_clips":108,"test_clips":108,
        "train_subjects":[1,3,4,6,7,8],"validation_subjects":[5,10],"test_subjects":[2,9],
        "validation_correct":96,"test_correct":94,"class_count":18,
        "limitations":["Researcher-defined subject split, not an official KARD benchmark", "Two validation and two test subjects", "Validation reused for checkpoint selection and XAI", "No idle/unknown class; uncalibrated scores", "Concept assets are action-label-derived proxies, not independent semantic annotations"],
        "concept_probes_sha256":sha(destination/"concept_probes.npz"),"channel_groups_sha256":sha(destination/"channel_groups.json")}
    (destination/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    cfg={"schema_version":1,"adapter":"movinet","name":"MoViNet A0-Stream · KARD-18",
         "backbone":"../models/backbones/a0_stream","head":"../models/kard/linear_head.npz","labels":"../models/kard/labels.json",
         "head_sha256":converted_sha,"backbone_sha256":provenance["backbone_data_sha256"],
         "input":{"fps":5.,"max_frames":50,"resolution":172},
         "xai_assets":{"probes":"../models/kard/concept_probes.npz","channels":"../models/kard/channel_groups.json"}}
    (ROOT/"configs/movinet_kard.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    rng=np.random.default_rng(42); x=rng.normal(size=(16,480)).astype(np.float32)
    original=head(x,training=False).numpy(); portable=((x-mean)/scale)@weights+bias
    error=float(np.max(np.abs(original-portable)))
    if error>1e-4: raise RuntimeError("Portable head does not match original")
    print(json.dumps({"portable_head_sha256":converted_sha,"original_vs_portable_max_logit_error":error}))


if __name__=="__main__": main()
