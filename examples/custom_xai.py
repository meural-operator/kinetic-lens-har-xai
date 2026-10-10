"""Run with --method examples.custom_xai:ReverseSensitivity."""
from kinetic_lens.contracts import Explanation
from kinetic_lens.xai import check


class ReverseSensitivity:
    requires=frozenset({"predict"})

    def explain(self,model,frames,target=None,**options):
        frames,base,target=check(model,frames,self.requires,target)
        changed=model.predict(frames[::-1].copy())
        return Explanation("custom_reverse",target,{"logit_drop":float(base[target]-changed[target])},
                           ["Frame reversal is an artificial intervention, not a causal action annotation."])
