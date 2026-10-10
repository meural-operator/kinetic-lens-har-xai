"""Independent deployment equivalence and XAI smoke check using existing KARD data."""
import argparse
import json
from pathlib import Path
import numpy as np
from kinetic_lens.registry import load_model, make_method
from kinetic_lens.video import read_video

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",required=True,type=Path);args=p.parse_args()
    import tensorflow as tf
    base=args.workspace/"movinet_evaluation/kard_a0_head_100pct"
    original=tf.keras.models.load_model(base/"kard_head.keras")
    model=load_model(ROOT/"configs/movinet_kard.json")
    manifest=json.loads((base/"manifest.json").read_text())
    validation=sorted([r for r in manifest["records"] if r["split"]=="validation"],key=lambda r:r["relative_path"])
    maximum=0.;correct=0;rows=[]
    for i,r in enumerate(validation):
        relative=r["relative_path"].replace("\\","/")
        frames,meta=read_video(args.workspace/"datasets/KARD"/relative,model.input_spec)
        z=model.predict(frames)
        cache=base/"feature_cache"/("validation_"+relative.replace("/","_").removesuffix(".mp4")+".npz")
        with np.load(cache) as a: raw=a["features"]
        old=original(raw.reshape(1,480),training=False).numpy()[0]
        error=float(np.max(np.abs(z-old))); maximum=max(maximum,error);correct+=int(z.argmax()==r["class_id"])
        if error>1e-4:raise AssertionError(f"Prediction differs on {Path(relative).stem}: {error}")
        rows.append({"clip_id":Path(relative).stem,"class_id":r["class_id"],"predicted":int(z.argmax()),"max_logit_error":error})
        if i%18==17:print(f"Verified {i+1}/108 clips",flush=True)
    methods={}
    relative="Two hand wave/a03_s05_e01.mp4"
    frames,_=read_video(args.workspace/"datasets/KARD"/relative,model.input_spec)
    first=model.predict(frames)
    for method in ("spatial","temporal","concept","contrastive","mechanistic"):
        result=make_method(method).explain(model,frames,target=2)
        text=json.dumps(result.to_dict(),allow_nan=False)
        methods[method]={"completed":True,"output_bytes":len(text),"target":result.target}
        if method=="contrastive": methods[method]["reconstruction_error"]=result.data["reconstruction_error"]
        if method=="spatial": methods[method]["replay_error"]=result.data["replay_max_logit_error"]
        print("Verified method",method,flush=True)
    replay=model.predict(frames)
    result={"validation_clips":len(validation),"validation_correct":correct,"max_logit_error_vs_original_head_cache":maximum,
            "repeat_after_explanations_max_logit_error":float(np.max(np.abs(first-replay))),"methods":methods,
            "provenance":model.provenance(),"checks":rows,
            "scope":"Deployment equivalence and execution; not new semantic/faithfulness validation or a new blind accuracy benchmark"}
    (ROOT/"docs/verification_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert correct==96 and result["repeat_after_explanations_max_logit_error"]<1e-4
    print("ALL_MODEL_CHECKS_PASSED",maximum,flush=True)


if __name__=="__main__":main()
