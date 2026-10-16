# Preserving pretrained semantics while adapting to KARD

Date: 10 October 2026. Bounded architecture investigation; no new full training
run and no new test-set evaluation. Proposed accuracy improvement is untested.

## What the completed experiment says

The original 480D affine control achieved test macro F1 91.93 ± 2.03%.
The compact mean head achieved 87.19 ± 2.35%, and adding its temporal
convolution achieved 86.17 ± 4.35%. These are mean ± sample SD across three
training seeds on one previously observed subject split, not confidence
intervals or evidence about an independent population. The temporal addition
changed macro F1 by −1.02 ± 2.00 percentage points relative to compact mean.
The original-feature MLP control also underperformed (83.03 ± 4.97%).

Thus the temporal module is not the only suspect. The compact route also
replaced a pretrained nonlinear 104→480 projection with spatially averaged
104D inputs and a newly initialized larger classifier. Representation,
pooling order, capacity, normalization and optimization changed together
relative to the original affine control. No single cause is established.

## New train/validation-only diagnostic

Used 324 training and 108 validation clips, all with verified activation-cache
and source hashes. Replayed the original final two blocks and original head.
Per-frame projected features averaged over time reproduce the original
pooled 480D cache with maximum absolute discrepancy 1.19e-6. All pretrained
backbone variables remained unchanged.

An intervention keeps the pretrained projection, activation and trained
original classifier fixed, but moves spatial averaging ahead of the nonlinear
projection. Validation accuracy changes from 88.89% to 5.56%; training
accuracy changes from 100% to 5.56%. Mean absolute standardized feature
difference is 1.482 on validation. This is a representation-distribution
intervention with an unchanged classifier, **not** the performance of the
retrained compact heads. It supports retaining the original nonlinear
transformation order; it does not prove the cause of their generalization gap.

In general, mean(hard_swish(Wx)) does not equal hard_swish(W mean(x)).
Pretrained channel expansion encodes nonlinear spatial statistics rather
than merely making an arbitrary vector longer.

## Recommended prototype: classifier-guided mergeable projection adapter

Preserve the original stem, all 15 blocks, original pointwise projection,
frozen BN, hard-swish, global pooling and trained KARD affine classifier.
During training add a low-rank update to the original projection kernel:

    W' = W + (alpha/r) A B S^T

W is 104×480; A is 104×4; B is 4×17. S is a fixed orthonormal 480×17
basis of the existing classifier's class-contrast directions, including
training normalization and frozen BN gain. Its dimension is at most C−1
because directions shared by all class logits cancel in their contrasts.
No validation/test labels are used to construct S. Only A and B train:
4×(104+17) = **484 parameters**, compared with 2,336 for unrestricted rank-4
LoRA and 16,434 for the failed compact temporal head. Freeze the existing
KARD head as well as the backbone and BN statistics in this experiment.

Initialize A randomly and B to zero. The update initially vanishes and the
baseline is preserved. B receives a gradient immediately; A's initial zero
gradient is expected, and becomes nonzero after B changes. After training,
merge the update into W and discard the factors and basis. The deployed
graph has the same kernel shape, parameters, tensor shapes and operation
count as the original model. This is a training-time adapter architecture;
it does not introduce a new deployed temporal block or reduce the original
model's deployed size. Wall-clock latency still requires measurement.

The basis is a task-informed capacity constraint, not an optimality theorem.
The intervening activation has input-dependent derivatives, so the true
useful update need not lie in this fixed subspace. Low rank can also constrain
adaptation too much. There is no guarantee of higher unseen accuracy.

The low-rank, zero-initialized, mergeable mechanism is established in
[LoRA](https://arxiv.org/abs/2106.09685). Classifier-informed adaptation for
streaming HAR is a candidate hypothesis, not a verified original contribution.
The backbone itself is [MoViNet](https://arxiv.org/abs/2103.11511).

## Verification and scope

`verification.json` records real pretrained projection and trained classifier
checks: exact zero-update identity, finite nonzero gradients, agreement between
explicit adapter and merged kernel after a synthetic nonzero update, and
agreement with the original restored projection graph. Synthetic updates are
numerical tests, not a learned checkpoint. Original weights are restored.
Projection checks use one real training clip; do not claim full-dataset or
end-to-end GPU parity from them.
These numerical checks use the existing historical TensorFlow KARD head,
not the freshly trained PyTorch A checkpoints. The next accuracy comparison
must start from each corresponding A seed's selected checkpoint to match
the strongest controlled baseline, rather than exploit the weaker historical
baseline. Do not select the best seed retrospectively.

A secondary 304-parameter temporal evidence-pooling prototype preserves the
baseline logits bitwise on all 432 training/validation clips at zero output
initialization. It reweights class evidence without discarding the original
projection. It adds runtime operations and has no accuracy results; it is
not the primary recommendation for the unchanged-inference-cost objective.

Files: `diagnosis.json`, `verification.json`, vector PDF/SVG and PNG figure,
train/validation projected-feature NPZ, and an adapter initialization NPZ.
Prototype implementations and reproducible verification scripts are in the
parent directory. They are not integrated into the live dashboard.

## Next controlled experiment

Compare unrestricted rank-4 LoRA, classifier-subspace rank-4 adaptation,
and rank-4 adaptation with a fixed random orthonormal 17D output subspace.
The latter has the same 484 trainable-parameter budget and tests whether
classifier guidance helps rather than merely restricting capacity.
Include the unchanged baseline as a candidate at epoch zero. Freeze BN and
head; use identical seeds, input clips, minibatches, tuning budgets and
checkpoint-selection rules. Start with low LR 1e-4 and select exclusively
by validation macro F1, with validation cross entropy as a tie-breaker.
Keep a bounded perturbation/regularization policy fixed before running.

Retaining epoch zero guarantees that selected validation macro F1 is no
lower than the starting model on that validation set. It cannot guarantee
test accuracy or new-subject generalization. This dataset split has already
been observed: establish additional prespecified subject splits or obtain
an independent evaluation before claiming generalization. Report negative
results and actual end-to-end CUDA/ONNX latency as well as parameter budgets.
