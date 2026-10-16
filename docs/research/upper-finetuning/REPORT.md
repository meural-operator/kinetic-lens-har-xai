# KARD: careful upper-backbone fine-tuning

Initialized from the official pretrained A0-Stream backbone and the validation-selected head trained on the full 324-clip training pool. The final two residual layers (`b4/l2`, `b4/l3`) and backbone projection (`head`) are fine-tuned together with the KARD head. This is partial backbone adaptation, not another frozen-feature head-only run.

## Declared protocol

- 498,850 trainable upper-backbone parameters plus 8,658 classifier parameters: 507,508 total.
- Adam learning rate 1e-5; gradient global-norm clipping at 1.0; maximum 20 epochs, patience five.
- Seeded exact-length video buckets with batches of at most eight; no frame padding, repetition or new temporal truncation. Every training clip contributes once per epoch.
- Keep all earlier backbone weights and every batch-normalization variable fixed, including affine gamma/beta. Run backbone layers in inference mode; gradients still update the explicitly selected upper weights.
- Retain the original train-only standardization statistics, dropout 0.25 and head L2 0.01. No new backbone weight decay, augmentation, person cropping, learning-rate search or layer-scope search.
- Keep the same RGB [0,1], 5 fps, aspect-preserving 172x172 padding and maximum 50 frames/centered 10-second window. Reset states per clip.
- Cache only the frozen b4/l1 endpoint in float32. Recompute and differentiate the trainable upper layers each step; verified against the full model.
- Checkpoint selection uses highest validation macro F1, then lower validation cross entropy. Include epoch zero as a fallback. New test activations are created only after selection.json is written.
- Native Windows TensorFlow CPU in the xai environment. Training subjects 1,3,4,6,7,8; validation 5,10; test 2,9. Source hashes unchanged.

| Model | Train accuracy | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Frozen head | 100.00% | 88.89% | 88.77% | 87.04% | 86.80% |
| Epoch-0 fallback | 100.00% | 88.89% | 88.77% | 87.04% | 86.80% |

**Negative result:** none of the five trained epochs surpassed the starting validation macro F1. The best trained epoch was 1, with validation macro F1 88.59% versus 88.77% initially. The selection rule retained epoch zero. Consequently, the exported model and its repeated test scores are the original frozen-head baseline, not a newly improved fine-tuned model. Actual final trained weights remain in `last_weights.npz` for inspection, but were not selected or evaluated on test.

| Epoch | Validation accuracy | Validation macro F1 | Validation cross entropy |
|---:|---:|---:|---:|
| 0 (initialization) | 88.89% | 88.77% | 0.3705 |
| 1 | 88.89% | 88.59% | 0.3536 |
| 2 | 87.96% | 87.66% | 0.3506 |
| 3 | 88.89% | 88.59% | 0.3582 |
| 4 | 88.89% | 88.59% | 0.3584 |
| 5 | 87.96% | 87.66% | 0.3556 |

Selected epoch 0 out of 5 executed epochs; 310 optimizer steps. Test accuracy changed by +0.00 percentage points, correcting 0 earlier errors and introducing 0 errors on previously correct clips.

Maximum selected backbone weight change: 0. Selected changes by layer group: `{"b4/l2/": 0.0, "b4/l3/": 0.0, "head/": 0.0}`. Frozen and batch-normalization digests remained identical. Starting checkpoint selected as fallback: True.

Training, evaluation on train/validation and checkpoint selection took 62.5 seconds, excluding activation extraction, model loading, final test and export. This is not a total training-cost or inference-latency benchmark.

## Verification and limitations

- Real finite, nonzero gradients were verified for all 17 selected backbone variables on a training clip. Equal-length batch equivalence passed.
- Actual last-epoch weights change all three selected layer groups and change full-model logits on a training clip. The cached-upper versus full-model maximum logit error after those updates is 0. This proves the backbone training path works even when the validation rule rejects the trained checkpoint.
- Three test clips were checked through the complete model and reloaded SavedModel; cached-upper and complete-model predictions agree within numerical tolerance.
- All original head-only models, downloaded weights and paper assets are preserved. No post-test tuning was performed.
- One seed and one researcher-defined subject split; only two test subjects. Earlier test results were already observed, so this is not a newly blind evaluation. No statistical significance, cross-validation, dataset-wide benchmark, privacy guarantee or methodological novelty is established.

## Artifacts and reuse

`protocol.json`, `source_verification.json`, `gradient_verification.json`, activation provenance, `history.json` and `selection.json` document the training procedure. `selected_weights.npz` stores the selected upper weights and head arrays; `last_weights.npz` stores the final actually trained weights. A head file alone cannot reproduce an updated backbone. Use the complete `saved_model/` for the selected inference model; when the epoch-zero fallback wins, this model is equivalent to the original frozen-head baseline.

`paper_assets/` contains PDF/SVG/600-dpi PNG plots, the numerical LaTeX table, figure insertion snippets and exact plot data. Existing head-only paper assets are unchanged.

Reproduce or resume completed stages with `movinet_evaluation/run_kard_finetune_pipeline.ps1`. An interrupted training history requires inspection rather than silently starting over.

```powershell
& 'python' -B movinet_evaluation\predict_kard_video.py --experiment-dir movinet_evaluation\kard_a0_upper_finetune --video 'C:\path\to\clip.mp4'
```

The low learning rate and inference-mode normalization follow the [TensorFlow transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning). The original backbone is the [official MoViNet model](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet).