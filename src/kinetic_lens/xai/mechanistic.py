from pathlib import Path
import json
import numpy as np
from ..contracts import Explanation, validate_logits
from . import check, margin


class ChannelIntervention:
    requires = frozenset({"predict","channels"})

    def explain(self, model, frames, target=None, groups=None, size=12, reference="training_mean", **options):
        frames,base,target=check(model,frames,self.requires,target)
        groups=groups or getattr(model,"xai_assets",{}).get("channels")
        if not groups: raise ValueError("Supply training-selected channel groups; see docs/adding-xai.md")
        profile=json.loads(Path(groups).read_text())
        if profile["head_sha256"]!=model.provenance().get("hashes",{}).get(model.head_path.name): raise ValueError("Channel selection/model mismatch")
        if reference not in ("training_mean","zero"): raise ValueError("Unknown reference")
        activation,logits=model.channel_activation(frames)
        if np.max(np.abs(base-logits))>1e-4: raise RuntimeError("Channel capture does not reproduce inference")
        replay=validate_logits(model.channel_predict(activation),model.labels)
        if np.max(np.abs(base-replay))>1e-4: raise RuntimeError("Channel replay does not reproduce inference")
        channels=activation.shape[-1]
        if not 1<=int(size)<=channels: raise ValueError("Invalid channel group size")
        key=str(target)
        if key not in profile["rankings"]: raise ValueError("No training-selected channel ranking for this class")
        mu=np.asarray(profile["training_mean"],np.float32)
        if mu.shape!=(channels,) or not np.isfinite(mu).all(): raise ValueError("Invalid training reference")
        replacement=mu if reference=="training_mean" else np.zeros(channels,np.float32)
        ranking=profile["rankings"][key]; rows=[]
        for name in ("activation","functional"):
            order=ranking[name]
            if sorted(order)!=list(range(channels)): raise ValueError("Ranking must be a complete channel permutation")
            ids=order[:int(size)]; mask=np.zeros(channels,bool); mask[ids]=True
            for operation,changed_mask in (("remove",mask),("retain",~mask)):
                changed=activation.copy(); changed[...,changed_mask]=replacement[changed_mask]
                z=validate_logits(model.channel_predict(changed),model.labels)
                rows.append({"ranking":name,"operation":operation,"channels":ids,"prediction":int(z.argmax()),
                             "target_margin_drop":margin(base,target)-margin(z,target),"logits":z.tolist()})
        return Explanation("mechanistic",target,{"size":size,"reference":reference,"baseline_logits":base.tolist(),"interventions":rows},
            ["Groups are training-selected candidates for Drink, Two hand wave and Forward Kick only.",
             "Internal intervention establishes a computational effect under this replacement, not a complete semantic circuit.",
             "Inspect collateral effects on other actions; target retention alone is insufficient."])
