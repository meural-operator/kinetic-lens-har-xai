# KARD compact temporal architecture: first controlled experiment

The official MoViNet A0-Stream stem and all 15 bottlenecks remain frozen.
The original spatial 104-to-480 projection and pooled classifier are bypassed.
New path: b4/l3 -> spatial average per frame -> train-only 104-channel
standardization -> causal depthwise temporal convolution (kernel 3), hard-swish
and residual -> masked temporal average -> Linear(104,128) -> LayerNorm ->
hard-swish -> dropout 0.25 -> Linear(128,18).

Trainable parameters: 16,434. Only the new module
and classifier are optimized; the original backbone is unchanged.
Preprocessing and subject split match the original full-training-pool baseline.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Original | 88.89% | 88.77% | 87.04% | 86.80% |
| Compact temporal | 90.74% | 90.57% | 93.52% | 93.23% |

The selected candidate uses L2 0.01, epoch
29. Selection uses validation only. The selected
checkpoint is evaluated on test once. See protocol.json, preflight.json,
selection.json, histories, evaluation.json and predictions.csv for evidence.

## Interpretation limits

This is one seed and one previously observed researcher-defined subject split,
with two validation and two test subjects. It is not a blind benchmark,
significance result or a demonstrated novel method. Model, normalization,
optimizer and regularization differ from the original TensorFlow linear head;
this is a complete architecture comparison, not an isolated temporal-module
ablation. No improved latency, smaller end-to-end artifact or real-time behavior
is claimed until both complete execution graphs are exported and benchmarked.
The frozen extraction here is TensorFlow CPU; new module training is PyTorch CUDA.
The new inference helper requires both runtimes. Dataset and old artifacts are preserved.
