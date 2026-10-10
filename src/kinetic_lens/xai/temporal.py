import numpy as np
from ..contracts import Explanation, validate_logits
from . import check


class TemporalAttribution:
    requires = frozenset({"predict"})

    def explain(self, model, frames, target=None, bins=5, seed=42, **options):
        frames, base, target = check(model,frames,self.requires,target)
        if not 1 <= bins <= min(10,len(frames)): raise ValueError("bins must be 1..min(10,T)")
        rng=np.random.default_rng(seed); rows=[]
        def measure(name,changed,**extra):
            z=validate_logits(model.predict(changed),model.labels)
            rows.append({"intervention":name,"target_logit_drop":float(base[target]-z[target]),
                         "predicted_class":int(z.argmax()),"logits":z.tolist(),**extra})
        measure("reverse",frames[::-1].copy())
        measure("shuffle",frames[rng.permutation(len(frames))])
        measure("static_middle",np.repeat(frames[len(frames)//2:len(frames)//2+1],len(frames),axis=0))
        references={"first_frame":frames[0],"middle_frame":frames[len(frames)//2],
                    "mean_frame":np.rint(frames.mean(axis=0)).astype(np.uint8)}
        for i,ids in enumerate(np.array_split(np.arange(len(frames)),bins)):
            for reference,value in references.items():
                changed=frames.copy(); changed[ids]=value
                measure("segment_replace",changed,bin=i,frame_indices=ids.tolist(),reference=reference)
        return Explanation("temporal",target,{"baseline_logits":base.tolist(),"seed":seed,"measurements":rows},
            ["Replacement and shuffling create artificial videos; effects are reference-dependent.",
             "Target defaults to the model prediction, which can be incorrect.",
             "These are perturbation effects, not independently identified causal action phases."])
