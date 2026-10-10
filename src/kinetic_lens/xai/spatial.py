import numpy as np
from ..contracts import Explanation
from . import check


class SpatialAttribution:
    requires = frozenset({"predict","spatial_gradients"})

    def explain(self, model, frames, target=None, rival=None, **options):
        frames,base,target=check(model,frames,self.requires,target)
        if rival is not None and (not 0 <= int(rival) < len(model.labels) or int(rival)==target):
            raise ValueError("Rival must be a different valid class")
        signed,replayed,error=model.spatial_map(frames,target,rival)
        if np.max(np.abs(base-replayed))>1e-4: raise RuntimeError("Explanation hook does not reproduce prediction")
        maps=np.maximum(signed,0) if rival is None else signed
        scale=float(np.max(np.abs(maps)))
        return Explanation("spatial",target,{"rival":rival,"maps":maps.tolist(),"shape":list(maps.shape),
            "display_scale":scale,"replay_max_logit_error":error,"baseline_logits":base.tolist(),
            "signed":rival is not None,"layer":"b4/l3"},
            ["Coarse 6x6 Grad-CAM maps contain temporal context, not independent frame importance.",
             "Visual plausibility is not a faithfulness test; run matched input perturbations."])
