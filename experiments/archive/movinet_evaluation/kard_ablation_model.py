"""Prespecified frozen-feature KARD ablations; no attention mechanisms."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PARAMETERS = {'A': 8658, 'B': 16018, 'C': 16434, 'D': 16050}
NAMES = {'A': '480D affine', 'B': '104D mean + MLP',
         'C': '104D causal temporal + MLP', 'D': '480D capacity MLP'}


class AblationHead(nn.Module):
    def __init__(self, variant, mean, scale):
        super().__init__()
        self.variant = variant
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.dropout = nn.Dropout(.25)
        # Common modules are constructed before the isolated temporal RNG.
        if variant in ('B', 'C'):
            self.project = nn.Linear(104, 128)
            self.norm = nn.LayerNorm(128)
            self.classifier = nn.Linear(128, 18)
            if variant == 'C':
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(torch.initial_seed() + 1000003)
                    self.temporal = nn.Conv1d(104, 104, 3, groups=104)
        elif variant == 'D':
            self.project = nn.Linear(480, 32)
            self.norm = nn.LayerNorm(32)
            self.classifier = nn.Linear(32, 18)
        elif variant == 'A':
            self.classifier = nn.Linear(480, 18)
        else:
            raise ValueError(variant)
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS[variant]

    def temporal_features(self, sequence, lengths):
        z = (sequence - self.mean) / self.scale
        mask = torch.arange(z.shape[1], device=z.device)[None] < lengths[:, None]
        z = z * mask[..., None]
        if self.variant == 'C':
            delta = self.temporal(F.pad(z.transpose(1, 2), (2, 0))).transpose(1, 2)
            z = z + F.hardswish(delta)
        return z, mask

    def forward(self, features, lengths=None):
        if self.variant in ('B', 'C'):
            z, mask = self.temporal_features(features, lengths)
            z = (z * mask[..., None]).sum(1) / lengths[:, None]
        else:
            z = (features - self.mean) / self.scale
        if self.variant != 'A':
            z = F.hardswish(self.norm(self.project(z)))
        return self.classifier(self.dropout(z))


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(variant, mean, scale, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return AblationHead(variant, mean, scale)


def kernel_l2(model):
    return sum(p.square().sum() for name, p in model.named_parameters()
               if name.endswith('weight') and p.ndim > 1)
