import numpy as np
from ..contracts import Explanation
from . import check


class ContrastiveExplanation:
    requires = frozenset({"predict","linear_head"})

    def explain(self, model, frames, target=None, rival=None, **options):
        frames,base,target=check(model,frames,self.requires,target)
        if rival is None: rival=int(np.argsort(base)[-2]) if target==base.argmax() else int(base.argmax())
        rival=int(rival)
        if not 0<=rival<len(model.labels) or target==rival: raise ValueError("Invalid rival")
        evidence=model.linear_evidence(frames); z=evidence["features"]; w=evidence["weights"]; b=evidence["bias"]
        terms=(w[:,target]-w[:,rival])*z; bias=float(b[target]-b[rival])
        reconstructed=float(terms.sum()+bias); residual=abs(reconstructed-float(base[target]-base[rival]))
        if residual>1e-4: raise RuntimeError("Linear evidence does not reproduce the class margin")
        return Explanation("contrastive",target,{"rival":rival,"rival_label":model.labels[rival],
            "actual_winner":int(base.argmax()),"margin":float(base[target]-base[rival]),"bias_difference":bias,
            "contributions":terms.tolist(),"reconstruction_error":residual,
            "largest_positive_indices":np.argsort(-terms)[:20].tolist(),
            "largest_negative_indices":np.argsort(terms)[:20].tolist()},
            ["Exact final-head arithmetic does not explain feature extraction or human concepts.",
             "Beating the chosen rival does not imply winning among every class.",
             "Learned coordinates and their contribution allocation depend on the reference/basis."])
