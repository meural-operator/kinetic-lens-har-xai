# KARD mergeable projection-adapter controlled comparison

Each adapter started from the corresponding seed's strong, validation-selected
480D affine baseline from the earlier controlled ablation. Original projection,
BN, backbone and classifier weights were frozen; only adapter factors trained.
All nine model selections were locked before test cache preparation or reads.

| Variant | Adapter parameters trained | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Frozen strong baseline | 0 | 91.03 ± 0.92 | 91.93 ± 2.03 | 92.28 ± 2.14 |
| Ordinary rank-4 LoRA | 2,336 | 91.30 ± 1.37 | 92.52 ± 1.01 | 92.90 ± 1.07 |
| Classifier-guided rank-4 | 484 | 91.03 ± 0.92 | 92.19 ± 1.58 | 92.59 ± 1.60 |
| Random-subspace rank-4 | 484 | 91.03 ± 0.92 | 92.46 ± 1.10 | 92.90 ± 1.07 |

Values are percentages, mean ± sample SD across three **training seeds**.
All seeds use the same 108 test clips on a previously observed subject split.
These are descriptive results, not independent-subject confidence intervals.

## Paired test macro F1 differences

- lora_minus_baseline: +0.59 ± 1.02 percentage points; per seed [0.0, 1.76, 0.0]
- guided_minus_baseline: +0.26 ± 0.45 percentage points; per seed [0.0, 0.78, 0.0]
- random_minus_baseline: +0.53 ± 0.92 percentage points; per seed [0.0, 1.6, 0.0]
- guided_minus_random: -0.27 ± 0.47 percentage points; per seed [0.0, -0.82, 0.0]
- guided_minus_lora: -0.33 ± 0.56 percentage points; per seed [0.0, -0.98, 0.0]

## Selected checkpoints

- lora_seed_42: selected epoch 2, λ=0.0001.
- guided_seed_42: selected epoch 6, λ=0.0001.
- random_seed_42: selected epoch 9, λ=0.0001.
- lora_seed_123: selected epoch 61, λ=0.01.
- guided_seed_123: selected epoch 15, λ=0.01.
- random_seed_123: selected epoch 100, λ=0.0001.
- lora_seed_2026: selected epoch 0, λ=0.0001.
- guided_seed_2026: selected epoch 0, λ=0.0001.
- random_seed_2026: selected epoch 0, λ=0.0001.

Epoch zero means the unchanged baseline was selected. Do not interpret such
a selection as a learned architectural improvement. All models kept the
baseline as a validation fallback. This ensures the selected validation
macro F1 does not decrease on this validation set, not that test accuracy
or unseen-subject generalization improves.

## Controls and limitations

Three seeds (42,123,2026), identical seeded minibatches and dropout starts,
three identical L2 candidates (.0001,.001,.01), Adam lr1e-4, batch16,
gradient clipping1, maximum100 epochs, patience15. The update is constrained
to 2% of the original kernel's Frobenius norm. BN and classifier stay frozen.
The penalty is λ times summed squared entries of the actual kernel update.
Guided and random adapters each train484 parameters; ordinary LoRA trains2336.
The random-subspace variant controls for reduced capacity. No best seed is chosen.

Preflight records original baseline prediction agreement on all432 train/val
clips per variant/seed, correct padding, real-data gradients and updates,
immutable frozen buffers, merged-kernel equivalence, and exact short resume.
The test gate is verified to reject before any test read. Cache sidecars and
inventories verify source videos and original backbone activations. Model
and code hashes are stored in protocol/preflight files.

The adapter merges into the same104→480 projection shape with unchanged
deployed parameter count and operation graph. No new end-to-end GPU/ONNX
latency or accuracy robustness claim follows from this experiment. Low-rank
adaptation is established; novelty of classifier-guided HAR adaptation needs
a separate literature review and independent evaluation. The fixed guidance
subspace need not contain the optimal update because activation derivatives
depend on the input. Negative results are retained.

Figures are exported as PDF/SVG/PNG; CSV and JSON preserve per-seed values.
