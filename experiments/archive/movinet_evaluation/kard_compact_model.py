"""Trainable compact causal head over frozen MoViNet per-frame features."""
import torch
from torch import nn
from torch.nn import functional as F


class CompactTemporalHead(nn.Module):
    def __init__(self, mean, scale, classes=18, width=128):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.temporal = nn.Conv1d(104, 104, 3, groups=104)
        self.project = nn.Linear(104, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(.25)
        self.classifier = nn.Linear(width, classes)

    def forward(self, sequence, lengths):
        # [B,T,104]. Right-padding is excluded from aggregation; causal kernels
        # cannot see a padded future. Every real frame appears exactly once.
        z = (sequence - self.mean) / self.scale
        mask = torch.arange(z.shape[1], device=z.device)[None] < lengths[:, None]
        z = z * mask[..., None]
        delta = self.temporal(F.pad(z.transpose(1, 2), (2, 0))).transpose(1, 2)
        z = z + F.hardswish(delta)
        pooled = (z * mask[..., None]).sum(1) / lengths[:, None]
        representation = F.hardswish(self.norm(self.project(pooled)))
        return self.classifier(self.dropout(representation))


def load_compact_head(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = CompactTemporalHead(saved['state_dict']['mean'], saved['state_dict']['scale'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved['labels']
