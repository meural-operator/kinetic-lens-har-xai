# KARD controlled architecture ablations

Four prespecified variants were trained with three prespecified seeds (42, 123,
2026). Every variant/seed received the same three-candidate L2 tuning budget.
All primary runs use PyTorch CUDA on the same frozen cached representations.
The main controlled comparison is B versus C: the causal temporal convolution
is their only architectural difference. Neither model uses attention.

| Variant | New trainable parameters | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| A: 480D affine | 8,658 | 91.03 ± 0.92 | 91.93 ± 2.03 | 92.28 ± 2.14 |
| B: 104D mean + MLP | 16,018 | 91.28 ± 1.91 | 87.19 ± 2.35 | 87.96 ± 2.45 |
| C: 104D causal temporal + MLP | 16,434 | 91.83 ± 0.44 | 86.17 ± 4.35 | 87.04 ± 4.04 |
| D: 480D capacity MLP | 16,050 | 89.98 ± 0.88 | 83.03 ± 4.97 | 83.64 ± 4.18 |

Values are percentages, mean ± sample standard deviation across **training
seeds**, not population confidence intervals. Each seed uses the same 108 test
clips. This split was observed in earlier studies, so test results are
descriptive and are not a newly blind benchmark.

The paired test macro F1 difference C−B is -1.02 ± 2.00
percentage points and is mixed or tied across seeds. Per-seed differences are
42: -0.06, 123: +0.33, 2026: -3.32 percentage points.
This small experiment supports a conditional comparison on this fixed split;
three seeds do not establish definitive superiority or subject generalization.
Negative and mixed results are retained without changing the protocol.

| Variant | Seed | L2 | Best epoch | Validation macro F1 | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| A | 42 | 0.01 | 43 | 90.93% | 93.10% |
| A | 123 | 0.0001 | 13 | 91.99% | 89.59% |
| A | 2026 | 0.01 | 50 | 90.16% | 93.10% |
| B | 42 | 0.01 | 21 | 89.71% | 88.18% |
| B | 123 | 0.01 | 36 | 90.74% | 88.88% |
| B | 2026 | 0.01 | 17 | 93.40% | 84.51% |
| C | 42 | 0.01 | 46 | 91.64% | 88.12% |
| C | 123 | 0.01 | 17 | 91.51% | 89.21% |
| C | 2026 | 0.01 | 9 | 92.34% | 81.19% |
| D | 42 | 0.0001 | 8 | 90.94% | 78.14% |
| D | 123 | 0.0001 | 11 | 89.20% | 82.87% |
| D | 2026 | 0.0001 | 7 | 89.81% | 88.09% |

## Architecture and experimental controls

A is an original-style standardized480D affine PyTorch control with dropout.25.
B uses equal-clip train-only standardized104D per-frame features, masked mean,
Linear104→128, LayerNorm, hard-swish, dropout.25 and Linear128→18.
C inserts a causal depthwise Conv1d104→104,k3 with hard-swish residual before
masked mean. B/C common initial weights and normalization are bitwise identical
per seed, verified in preflight.json. C's temporal weights use an isolated RNG.
D uses the original pooled480D endpoint, Linear480→32, LayerNorm, hard-swish,
dropout.25 and Linear32→18. D approximately matches the changed-part parameter
budget; it retains the original frozen104→480 projection and does not match
B/C full-model compute. Original stem and all15 bottlenecks remain frozen.

Adam uses lr.001, batch16, global gradient norm clip1, maximum200 epochs,
patience25. Loss adds explicit kernel L2 (bias and LayerNorm excluded).
The L2 grid is .0001,.001,.01; checkpoints and L2 are selected independently
per variant/seed by validation macro F1, then lower validation cross entropy.
No best seed is selected. All candidates share deterministic per-seed shuffled
minibatches and reset dropout streams. Atomic epoch checkpoints include model,
optimizer, all RNG, history, best weights and early-stop state. Exact short
continuation equivalence, gradients, updates, reload, padding and causality
were checked on real training caches.

Input sampling is5fps, resize with padding172×172, maximum50 frames, centered
10-second window when needed. Training subjects1,3,4,6,7,8 (324 clips),
validation5,10 (108), test2,9 (108); all18 labels are retained.
480D normalization was independently reconstructed and compared bitwise to
the historical training-only reference.104D normalization uses equal clip
weights and also matches the saved reference.

All12 selections and their checkpoint hashes were persisted before any new
test cache reads. Only selected checkpoints were evaluated. No post-test
tuning occurred. Cache inventories verify video/model/manifest provenance,
stored104D cache/source hashes, frame agreement and finite array shapes.
Historical480D NPZ sidecars do not contain stored cache checksums; independently
created SHA256 inventories verify immutability from this audit onward and
cannot establish unavailable historical checksums retrospectively.

## Historical reference and limits

The historical TensorFlow CPU affine baseline had validation macro F1
88.77%, test macro F1 86.80%, and test accuracy
87.04%. It is contextual. Fresh A is not an exact reproduction:
PyTorch initialization, Adam defaults and gradient clipping differ.
See historical_context.json for the original result and selection.

Seed variation measures training randomness, not variability across new
subjects. Two subjects in each held-out partition and a previously observed
split constrain inference. The main B/C pair isolates the added temporal
module within this prespecified training recipe; comparisons to A/D also
change representation and/or head capacity. No whole-model latency,
end-to-end speed, artifact-size, novelty or generic-superiority claim is made.

## Outputs

per_seed_metrics.csv, summary_metrics.csv, paired_C_minus_B.csv,
per_class_per_seed.csv and per_class_summary.csv preserve numerical results.
Per-seed confusion CSVs and plots retain each run. Aggregate confusion
matrices are **elementwise mean counts across runs**, with fractional cells;
they are not pooled324 independent clips. All18 classes map to indices1–18
in plots using the ordering in labels.json. Figures are supplied as PDF, SVG
and320dpi PNG. Histories and candidate/selected checkpoints are retained.
Use figures_for_paper.tex as a snippet; existing paper files are untouched.
