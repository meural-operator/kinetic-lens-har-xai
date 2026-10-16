"""Mergeable classifier-subspace adaptation of MoViNet's original projection.

Research prototype, not an empirically demonstrated accuracy improvement.
Tensor convention: channels last; original kernel is [104, 480].
"""
import torch
from torch import nn


def classifier_basis(classifier_weight, normalization_scale, bn_gain):
    """Fixed class-contrast span, including normalization and frozen BN gain.

    This is a structural prior, not a proof that the span contains the optimum:
    the following nonlinear activation has input-dependent derivatives.
    """
    effective = classifier_weight.double() / normalization_scale.double()[:, None]
    effective = effective * bn_gain.double()[:, None]
    effective = effective - effective.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(effective, full_matrices=False)
    tolerance = max(effective.shape) * torch.finfo(effective.dtype).eps * s[0]
    q = int((s > tolerance).sum())
    return u[:, :q].float()


class ProjectionAdapter(nn.Module):
    """W' = W + (alpha/r) A B S^T; B=0 preserves the baseline at init.

    Merge once for deployment: same kernel shape and operation graph as before.
    Batch normalization, activation, pooling and the trained head stay intact.
    """
    def __init__(self, kernel, basis, rank=4, alpha=4):
        super().__init__()
        if rank <= 0 or basis.shape[0] != kernel.shape[1]:
            raise ValueError('Invalid rank or basis shape')
        self.register_buffer('kernel', kernel.detach().clone())
        self.register_buffer('basis', basis.detach().clone())
        self.factor_in = nn.Parameter(torch.empty(kernel.shape[0], rank))
        self.factor_out = nn.Parameter(torch.zeros(rank, basis.shape[1]))
        nn.init.normal_(self.factor_in, std=0.01)
        self.multiplier = alpha / rank

    def delta(self):
        return self.multiplier * self.factor_in @ self.factor_out @ self.basis.T

    def merged_kernel(self):
        return self.kernel + self.delta()

    def forward(self, x):
        return x @ self.kernel + self.multiplier * ((x @ self.factor_in) @ self.factor_out) @ self.basis.T


def hard_swish(x):
    return x * torch.clamp(x + 3, 0, 6) / 6


def frozen_projection(x, kernel, gamma, beta, mean, variance, epsilon=0.001):
    return hard_swish((x @ kernel - mean) * gamma / torch.sqrt(variance + epsilon) + beta)
