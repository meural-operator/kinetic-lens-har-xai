# Temporal input sampling: recognition quality and benchmark coverage

The 5/10-fps comparison preserves a maximum ten-second context and changes both sampling density and frame cap. It does not hold the number of frames fixed.

| FPS | Frame cap | Maximum context | Validation accuracy (%) | Validation macro F1 (%) | Test accuracy (%) | Test macro F1 (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 50 | ≤10 s | 91.73 ± 2.89 | 91.56 ± 2.67 | 87.78 ± 4.90 | 87.02 ± 5.35 |
| 10 | 100 | ≤10 s | 90.06 ± 3.16 | 89.73 ± 3.22 | 85.43 ± 4.41 | 85.03 ± 4.39 |

Mean ± sample SD across five fold means, with three seeds averaged within each fold. The 30 rate/fold/seed combinations used 90 L2 candidate fits. All selections preceded new-fold test scoring. These are exploratory, researcher-defined subject folds, not a new blind external benchmark.

Ten fps changes mean test accuracy by −2.35 percentage points and macro F1 by −1.99 points. It wins macro F1 in 4/15 paired fold/seed runs and 1/5 seed-averaged folds. This does not prove that 5 fps is universally optimal.

## Actual input lengths

| FPS | Minimum real frames | Median real frames | Maximum real frames | Mean real frames |
|---:|---:|---:|---:|---:|
| 5 | 9 | 23 | 50 | 23.91 |
| 10 | 17 | 45 | 100 | 47.21 |

Statistics cover all 540 KARD clips. A cap of 50/100 does not mean every clip contained that many frames. No frame repetition or padding was used; the source timestamp audit verifies real distinct samples.

## What the existing timing records can and cannot say

- Official pretrained MoViNets have retained-state **one-frame** and fresh-state **eight-frame** CPU timing records. The state policies differ, so this is not a pure fixed-length scaling experiment.
- ViT–BiLSTM precision benchmarks use **eight-frame** windows.
- Our historical trained KARD MoViNet CPU/CUDA benchmarks use **20-frame** windows at 5 fps (four seconds of context).
- The 5/10-fps recognition study did **not** benchmark latency. Its missing timings must remain unreported.
- We have not recorded a matched sweep of one unchanged checkpoint/runtime/precision at, for example, 4/8/16/20/32 frames. Existing 1/8/20-frame results cannot be pooled to infer that curve.
- Sampling density and input length affect activation/computation workload; they do not create new weight-precision or parameter-count variants.

For a paper, use the verified sampling-quality table separately from the size/latency tables. A future frame-count trade-off experiment must specify whether it varies context duration at fixed fps, or sampling density at fixed context, and measure accuracy and latency under that same protocol.
