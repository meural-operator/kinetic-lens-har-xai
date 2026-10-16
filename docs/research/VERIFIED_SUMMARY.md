# Independently recomputed study summary

Generated with `python experiments/report_results.py --output docs/research/VERIFIED_SUMMARY.md`.
This verifies public archived results and hash records; it does not retrain models or test a camera.
Private checkpoints are excluded from the default offline verification. Use `--require-private` after authenticated setup to verify their actual bytes; otherwise their manifest identities alone are checked.

## Temporal sampling

Five fold means; three seeds averaged within each fold. Percentages, mean ± fold SD.

| FPS | Test accuracy | Test macro F1 |
|---|---:|---:|
| 5 | 87.78 ± 4.90 | 87.02 ± 5.35 |
| 10 | 85.43 ± 4.41 | 85.03 ± 4.39 |

## Architecture ablations

Percentages, mean ± training-seed SD on the same fixed split.

| Variant | Test accuracy | Test macro F1 |
|---|---:|---:|
| A | 92.28 ± 2.14 | 91.93 ± 2.03 |
| B | 87.96 ± 2.45 | 87.19 ± 2.35 |
| C | 87.04 ± 4.04 | 86.17 ± 4.35 |
| D | 83.64 ± 4.18 | 83.03 ± 4.97 |

## Projection adapters

| Variant | Test accuracy | Test macro F1 |
|---|---:|---:|
| baseline | 92.28 ± 2.14 | 91.93 ± 2.03 |
| lora | 92.90 ± 1.07 | 92.52 ± 1.01 |
| guided | 92.59 ± 1.60 | 92.19 ± 1.58 |
| random | 92.90 ± 1.07 | 92.46 ± 1.10 |

SDs are descriptive and are not confidence intervals. Sampling-fold and seed SDs capture different variation. The default deployed head is the historical 94/108-test-correct model; these later controlled heads and adapters do not replace it. Previously observed data limit independent generalization claims.

Verification counts:

```json
{
  "published_files_verified": 231,
  "private_publication_records_checked": 21,
  "private_artifact_records_checked": 26,
  "private_files_verified": 0,
  "sampling_runs": 30,
  "ablation_selected_runs": 12,
  "adapter_rows_including_baselines": 12,
  "selected_checkpoint_hash_records_checked": 21,
  "serialized_selected_checkpoints_verified": 0,
  "raw_vit_timing_calls_verified_including_superseded_gpu_session": 2025,
  "size_latency_forms": 31,
  "serialized_inference_artifact_sizes": 26,
  "training_or_camera_executed": false
}
```
