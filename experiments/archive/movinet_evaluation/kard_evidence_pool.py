"""Baseline-preserving, low-capacity temporal reweighting of action evidence.

This is a research prototype, not a claim of improved unseen accuracy.
The baseline logits are passed through directly; zero output initialization
therefore preserves them bitwise in this module's dtype.
"""
import torch
from torch import nn


class EvidencePool(nn.Module):
    def __init__(self, descriptor_mean, descriptor_scale, blend=.25):
        super().__init__()
        self.register_buffer('descriptor_mean',torch.as_tensor(descriptor_mean,dtype=torch.float32))
        self.register_buffer('descriptor_scale',torch.as_tensor(descriptor_scale,dtype=torch.float32))
        self.hidden=nn.Linear(36,8)
        self.score=nn.Linear(8,1,bias=False)
        nn.init.zeros_(self.score.weight)
        self.blend=blend
        assert sum(p.numel() for p in self.parameters())==304

    def descriptors(self, evidence):
        # Center over classes to remove the arbitrary shared logit offset.
        centered=evidence-evidence.mean(-1,keepdim=True)
        delta=torch.cat([torch.zeros_like(centered[:,:1]),centered[:,1:]-centered[:,:-1]],dim=1)
        return torch.cat([centered,delta],dim=-1)

    def forward(self, baseline_logits, evidence, lengths, return_weights=False):
        descriptor=(self.descriptors(evidence)-self.descriptor_mean)/self.descriptor_scale
        scores=1.5*torch.tanh(self.score(torch.tanh(self.hidden(descriptor))).squeeze(-1)/1.5)
        mask=torch.arange(evidence.shape[1],device=evidence.device)[None]<lengths[:,None]
        weights=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=1)
        uniform=mask.to(evidence.dtype)/lengths[:,None]
        # Computing a zero-centered correction protects the exact baseline
        # at initialization even if per-frame averaging has rounding error.
        correction=((weights-uniform)[...,None]*evidence).sum(1)
        output=baseline_logits+self.blend*correction
        return (output,weights) if return_weights else output
