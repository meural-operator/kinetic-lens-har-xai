"""Training-only ridge probes and bounded, explicit classifier-space interventions."""
from pathlib import Path
import numpy as np
from ..contracts import Explanation
from . import check


def fit_probe(features, positives, alpha=10.):
    x=np.asarray(features,np.float64); y=np.asarray(positives,bool)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or len(np.unique(y))!=2 or not np.isfinite(alpha) or alpha<=0:
        raise ValueError("Provide finite training features, both binary groups, and positive alpha")
    # Balanced family weights; an intercept is recovered after centering.
    sw=np.where(y,.5/y.sum(),.5/(~y).sum())*len(y)
    center=np.average(x,axis=0,weights=sw); yc=np.average(y.astype(float),weights=sw)
    xc=x-center; target=y.astype(float)-yc
    w=np.linalg.solve((xc*sw[:,None]).T@xc+alpha*np.eye(x.shape[1]),(xc*sw[:,None]).T@target)
    norm=np.linalg.norm(w)
    if norm<1e-12: raise ValueError("Degenerate probe direction")
    return {"weights":w,"bias":float(yc-center@w),"direction":w/norm,"alpha":alpha}


class ConceptAttribution:
    requires = frozenset({"predict","linear_head"})

    def explain(self, model, frames, target=None, probes=None, **options):
        frames,base,target=check(model,frames,self.requires,target)
        probes=probes or getattr(model,"xai_assets",{}).get("probes")
        if not probes: raise ValueError("Provide training-fitted concept probes; see docs/adding-xai.md")
        with np.load(Path(probes),allow_pickle=False) as a:
            directions=a["directions"].astype(float); names=a["names"].astype(str).tolist()
            expected=str(a["head_sha256"].item())
        if expected!=model.provenance().get("hashes",{}).get(model.head_path.name):
            raise ValueError("Concept probes do not match this model head")
        e=model.linear_evidence(frames); z=e["features"]; w=e["weights"]; b=e["bias"]
        if directions.shape!=(len(names),len(z)) or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-5):
            raise ValueError("Expected finite unit concept directions aligned to classifier features")
        rows=[]
        for name,v in zip(names,directions):
            projection=float(z@v); changed=(z-projection*v)@w+b
            rows.append({"name":name,"feature_projection":projection,"target_directional_derivative":float(w[:,target]@v),
                         "target_logit_drop":float(e["logits"][target]-changed[target]),"prediction_after_deletion":int(changed.argmax())})
        return Explanation("concept",target,{"concepts":rows,"probe_identity":"action-label-derived proxies"},
            ["Bundled probes use action-family labels; independent semantic validation remains partial.",
             "At an affine head, directional derivatives are constant across videos; TCAV sign fractions degenerate.",
             "Feature deletion need not correspond to a realizable video or natural removal of a concept."])
