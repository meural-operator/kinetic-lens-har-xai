# KARD: paired sampling comparison across subject folds and seeds

Completed five subject folds, three seeds per fold, and two sampling rates: **30 fresh heads and 90 L2 candidate fits**. The official A0-Stream backbone remains frozen. All 30 checkpoint/L2 selections and 15 validation-based sampling decisions were locked before new-fold test scoring.

| FPS | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| 5 | 91.73 ± 2.89% | 91.56 ± 2.67% | 87.78 ± 4.90% | 87.02 ± 5.35% |
| 10 | 90.06 ± 3.16% | 89.73 ± 3.22% | 85.43 ± 4.41% | 85.03 ± 4.39% |

Values are means ± sample SD across five fold means; three seeds are averaged within each fold. Seeds are not counted as independent subject groups. SDs are descriptive, not confidence intervals.

10 minus 5 fps: test accuracy **-2.35 percentage points**, test macro F1 **-1.99 points**. 10 fps has higher test macro F1 in 4/15 paired fold/seed runs and 1/5 seed-averaged folds; 0 paired ties.
Validation selects 10 fps in 5/15 runs and 5 fps in 10/15. The procedure that uses each run's validation-selected rate has descriptive test accuracy 87.90% and macro F1 87.32%. Both fixed-rate arms are still reported; no post-test tuning occurs.

## Protocol and leakage controls

| Fold | Training subjects | Validation subjects | Test subjects |
|---|---|---|---|
| 1 | 5, 6, 7, 8, 9, 10 | 3, 4 | 1, 2 |
| 2 | 1, 2, 7, 8, 9, 10 | 5, 6 | 3, 4 |
| 3 | 1, 2, 3, 4, 9, 10 | 7, 8 | 5, 6 |
| 4 | 1, 2, 3, 4, 5, 6 | 9, 10 | 7, 8 |
| 5 | 3, 4, 5, 6, 7, 8 | 1, 2 | 9, 10 |

## Per-seed out-of-fold scores

Each row pools the five held-out folds, covering all 540 unique clips once for that seed. Pooled macro F1 differs from the mean of fold macro F1 values above.

| Seed | FPS | Out-of-fold test accuracy | Out-of-fold test macro F1 |
|---|---:|---:|---:|
| 42 | 5 | 87.04% | 87.24% |
| 42 | 10 | 85.93% | 86.31% |
| 123 | 5 | 88.70% | 88.85% |
| 123 | 10 | 85.74% | 85.76% |
| 2026 | 5 | 87.59% | 87.84% |
| 2026 | 10 | 84.63% | 84.75% |

## Controls and verification

- Each fold has 324/108/108 clips with disjoint subjects; every action has 18/6/6 clips. Every subject is tested once per seed. Sorted consecutive pairs determine test folds; the next pair cyclically validates. No prediction-based partitioning.
- Seeds 42, 123, 2026; both rates use matched initialization and shuffle seed. RGB [0,1], aspect-preserving padding to 172 square, centered maximum 10-second window, fresh streaming state per clip; caps 50/100. No cropping or augmentation.
- All 1,080 cached feature vectors and 540 source-file hashes are verified. Only raw features from the fixed pretrained backbone are reused. Their former split labels do not restrict their use in a new fold. No old normalization, trained classifier or previous prediction enters training.
- Fresh train-only 480-feature normalization for each fold/rate; floor std 0.001. Dropout 0.25 and Dense(18), 8,658 trainable parameters. Adam 0.001, batch 16, max 200 epochs, patience 25, L2 candidates 0.0001/0.001/0.01. Checkpoint/candidate choice: highest validation macro F1, then lower cross entropy.
- All selected .keras heads reload with maximum logit error below 1e-6. Independent checks confirm all 30 normalizations equal statistics of exactly their training clips, all 90 candidate checkpoints match their best validation history entry, optimizer steps match executed epochs, and confusion-matrix recomputation agrees with all 30 test reports.
- Runs use Miniconda xai and native Windows TensorFlow CPU. This experiment does not benchmark inference latency or GPU execution.

## Interpretation and limits

This study measures stability of two sampling policies for a frozen pretrained representation and a lightweight KARD head. It is an exploratory follow-up after earlier KARD results were observed. Test subjects are excluded from their own model training and normalization; this is not a newly blind external evaluation. Training sets overlap across folds and seeds use the same clips. No p-values, confidence intervals or population-level significance claims are made. Higher fps changes both input density and the physical receptive field of temporal convolutions. Performance alone does not establish why a particular action succeeds or fails.
These are researcher-defined subject folds, not a published official KARD benchmark. No new privacy method, architectural novelty or upper-backbone fine-tuning is claimed. Do not choose the best test-scoring fold checkpoint as a deployment model: the fold checkpoints serve evaluation and do not train on all ten subjects.

## Reproduction and artifacts

Run `run_kard_sampling_cv.py` followed by `write_kard_sampling_cv_report.py` with the xai interpreter. The runner preserves matching completed selections and verifies immutable protocol/input/selection hashes on resume.
`protocol.json`, `feature_provenance.json`, `selection_lock.json`, and per-run candidate histories, normalization, selected heads, predictions, test logits and metrics retain the evidence. Numeric CSVs provide run, fold, subject, seed, action and paired summaries. Action metrics pool the five held-out folds for each seed, then average seeds. `paper_assets/` contains PDF/SVG/600-dpi PNG figures, figure/table LaTeX snippets and exact plot data. Existing earlier checkpoints are preserved.