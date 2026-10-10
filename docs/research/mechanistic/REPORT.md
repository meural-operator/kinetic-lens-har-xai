# Mechanistic channel intervention extension

The original deployed KARD A0-Stream baseline is 96/108 on all validation clips. This extension runs 3,420 interventions, trains no HAR model, and accesses no test examples. The two validation subjects have counts {5: (53, 54), 10: (43, 54)}. All six examples per target are included; Forward Kick is correct on 3/6, and its s10 examples are all originally predicted Walk. Candidate channel effects must therefore distinguish degradation of successful recognition from score changes on already incorrect recognition.

## Model and data

The unchanged official A0-Stream is cut after b4/l1 with cached [T,6,6,104] activations. Original restored layers with weights 14,15,16 produce the 480-dimensional endpoint, followed by the original train-standardized Dense18 head. Input is padded RGB at 172 square, 5 fps, maximum 50 sampled frames. Fresh states and inference normalization are used on every batch. Original head SHA256: `e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70`. Original backbone variable-data SHA256: `f35edf9b7b747b7fd5b65f127a34c988bec7c6602b1618af123b970d5847d6d2`.

## Prespecified training-only selection

PLAN.md was saved before running the experiments. Activation-separation ranks all 104 channels by absolute target-v-rest difference of clip-pooled training activations divided by the all-training clip-mean SD (324 clips). Functional sensitivity uses all 18 target training examples per class: average the spatial/temporal sum of the absolute product of activation deviation from the training mean and gradient of target-minus-fixed-rival logit. The rivals selected from mean target-training logits are Drink versus Draw Tick, Two hand wave versus High arm wave, Forward Kick versus Side Kick. Eighteen fixed non-target training controls per target measure comparison sensitivity but do not influence selection. No validation result chooses rankings, rivals, sizes, seeds, or references. No head fitting occurs.

The gradient heuristic is a local first-order magnitude estimate. It discards effect sign and can select generally influential channels. It is not an optimized minimal circuit. Top12 activation/functional overlaps are {'Drink': 2, 'Two hand wave': 2, 'Forward Kick': 2} channels. Nested random permutations use seeds 6101-6103, shared across targets. They differ from the original pilot's independent size 12 draws and are not exact replications of those controls.

## Interventions and measurement

All 18 target validation clips receive removal/retention at sizes 8,12,24,52,104 under two references (equal-clip-weight training channel mean, or zero). Removal replaces selected channels across all positions/frames; retention replaces their complement. Both references are spatially and temporally constant. Three nested random groups accompany both rankings. This produces 1,800 curve rows. All 108 validation clips also receive size 12 mean-reference removal for all three target groups and five rankings/controls, producing 1,620 collateral rows. The overlapping conditions are intentionally retained in separate study rows, not treated as independent observations.

Target margin is target logit minus the largest other logit. Drop is original minus intervened margin; a negative value increases the target's margin. True-label margin and original-prediction margin are separately recorded. Off-target target-margin drops compare the same target score but do not measure evidence for those clips' own labels. Classification uses all 18 logits. The full raw vectors support alternative analyses without rerunning inference.

## Size 12 findings

| Target | Ranking | Mean removal drop | Random mean (range) | Zero removal drop | Mean retain correct | Zero retain correct | Other correct | Other true-margin drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drink | activation | 1.282 | 0.899 (0.039-1.557) | 2.435 | 0/6 | 0/6 | 81/102 | 1.023 |
| Drink | functional | 4.702 | 0.899 (0.039-1.557) | 3.523 | 0/6 | 0/6 | 41/102 | 3.069 |
| Two hand wave | activation | 1.979 | 1.535 (0.681-2.209) | 1.728 | 0/6 | 0/6 | 90/102 | 0.089 |
| Two hand wave | functional | 5.027 | 1.535 (0.681-2.209) | 3.691 | 0/6 | 0/6 | 65/102 | 1.765 |
| Forward Kick | activation | 0.379 | 0.094 (-0.409-0.928) | 0.412 | 0/6 | 0/6 | 89/102 | 0.516 |
| Forward Kick | functional | 2.455 | 0.094 (-0.409-0.928) | 1.800 | 0/6 | 0/6 | 55/102 | 2.284 |

Random ranges describe three channel permutations, not subject uncertainty, confidence intervals, or significance. Each target accuracy uses only six clips from two subjects.

## Retention across sizes

| Target | Ranking | Reference | Retain8 | Retain12 | Retain24 | Retain52 | Retain104 |
|---|---|---|---:|---:|---:|---:|---:|
| Drink | activation | training_mean | 0/6 | 0/6 | 0/6 | 0/6 | 6/6 |
| Drink | activation | zero | 0/6 | 0/6 | 0/6 | 0/6 | 6/6 |
| Drink | functional | training_mean | 0/6 | 0/6 | 0/6 | 4/6 | 6/6 |
| Drink | functional | zero | 0/6 | 0/6 | 0/6 | 4/6 | 6/6 |
| Two hand wave | activation | training_mean | 0/6 | 0/6 | 0/6 | 6/6 | 6/6 |
| Two hand wave | activation | zero | 0/6 | 0/6 | 0/6 | 6/6 | 6/6 |
| Two hand wave | functional | training_mean | 0/6 | 0/6 | 0/6 | 6/6 | 6/6 |
| Two hand wave | functional | zero | 0/6 | 0/6 | 1/6 | 6/6 | 6/6 |
| Forward Kick | activation | training_mean | 0/6 | 0/6 | 0/6 | 0/6 | 3/6 |
| Forward Kick | activation | zero | 0/6 | 0/6 | 0/6 | 0/6 | 3/6 |
| Forward Kick | functional | training_mean | 0/6 | 0/6 | 0/6 | 4/6 | 3/6 |
| Forward Kick | functional | zero | 0/6 | 0/6 | 0/6 | 6/6 | 3/6 |

Full retention is a no-op and recovers the original predictions, including the original mistakes. Full removal is ranking-independent, independently checked from saved outputs. These boundaries verify implementation behavior but do not establish a semantic mechanism.

## Baseline correctness stratification

| Target | Ranking | Baseline stratum | n | Removal drop mean-reference size 12 |
|---|---|---|---:|---:|
| Drink | activation | correct | 6 | 1.282 |
| Drink | functional | correct | 6 | 4.702 |
| Two hand wave | activation | correct | 6 | 1.979 |
| Two hand wave | functional | correct | 6 | 5.027 |
| Forward Kick | activation | correct | 3 | 0.609 |
| Forward Kick | activation | wrong | 3 | 0.150 |
| Forward Kick | functional | correct | 3 | 4.907 |
| Forward Kick | functional | wrong | 3 | 0.003 |

For Forward Kick, original target-minus-Walk margins by clip are a09_s05_e01 4.078, a09_s05_e02 5.019, a09_s05_e03 3.836, a09_s10_e01 -0.686, a09_s10_e02 -0.620, a09_s10_e03 -0.347. A negative true Kick margin on an originally Walk-predicted clip means the original recognition is already wrong. Dropping that true-class margin is different from explaining the original Walk decision. Original-prediction margin and Kick-Walk margin are available for every intervention.

## Verification

All 432 training/validation source-video and activation-cache hashes pass. Cached activations trace to the original parent manifest, initial head, original backbone file, and identical 5 fps/50 frame/172 RGB input policy. Metadata hashes the upper-experiment protocol, which differs from the deployed head protocol; no upper-tuned checkpoint is restored. The maximum error between downstream cache logits and original feature-cache/head logits across all 108 validation clips is 5.7220459e-06. Independent replay and batched no-op checks have maximum logit error 5.7220459e-06. The same three prior illustration clips independently match full original-backbone and deployed SavedModel predictions within 1e-4; their errors are stored in analysis.json. Full-retention rows match each baseline within 1e-4. Full-removal group-invariance max error is 0. Protected head/backbone/protocol/manifest hashes and in-memory head/backbone variable digests are unchanged. Total execution took 66.6 seconds.

## Interpretation and limitations

See INTERPRETATION.md for accessible worked examples and the necessary/sufficient distinction. The strongest justified statement is local computational causality: replacing a specified internal tensor changes the deterministic downstream model output under this intervention. There is no identification of a real-world action cause, minimal complete circuit, semantic channel meaning, or circuit edge. Reference-dependent damage, nonlinear interaction, distributed information, rank dependence, and subject-specific baseline errors all constrain stronger claims. No perturbation creates a verified plausible counterfactual video. Only two validation subjects are studied; repetitions and random groups are not additional subjects. No validation winner is promoted to a discovered circuit.

## Files and references

Raw data: interventions.json/CSV, intervention_logits.npz, baseline.csv, summary.csv, train_sensitivity.json, selection.json, analysis.json. Reusable run/render scripts are in the parent evaluation folder. Figures are removal_reference_curves, retention_reference_curves, collateral_true_label_effects (PDF/SVG/PNG). Standalone LaTeX and a reusable snippet accompany the reports. Curve figures display prescribed group sizes as equally spaced categories; line segments connect discrete tested conditions and do not interpolate untested sizes.

[Heimersheim and Nanda (2024), How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) discusses what patching supports about circuits and metric pitfalls. [Zhang and Nanda (2023/2024), Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042) demonstrates sensitivity to corruption and measurement choices. These motivate comparison of references and metrics here; language-model results are not assumed to transfer unchanged to this CNN. The local gradient heuristic is an explicitly documented approximation, evaluated by actual forward interventions rather than treated as proof.

## Exploratory retention 52 selectivity follow-up

The observed functional retention 52 Kick success triggered PLAN_AMENDMENT.md. This is a result-triggered exploratory check, not part of the prespecified main study. It fixes all existing groups and references, then applies the three target-specific functional groups plus the same three shared random groups under both references to all 108 validation clips: 1,296 unique additional interventions. Shared random logits support all three target analyses without being counted as independent new model interventions.

| Target | Group | Reference | Target correct | Off-target false positives | Other correct | Overall correct | Other true-margin drop |
|---|---|---|---:|---:|---:|---:|---:|
| Drink | functional_15 | training_mean | 4/6 | 0/102 | 45/102 | 49/108 | 2.947 |
| Drink | functional_15 | zero | 4/6 | 0/102 | 75/102 | 79/108 | 1.473 |
| Drink | random_6101 | training_mean | 0/6 | 0/102 | 14/102 | 14/108 | 7.525 |
| Drink | random_6101 | zero | 0/6 | 0/102 | 16/102 | 16/108 | 6.066 |
| Drink | random_6102 | training_mean | 0/6 | 0/102 | 11/102 | 11/108 | 7.196 |
| Drink | random_6102 | zero | 1/6 | 0/102 | 21/102 | 22/108 | 5.524 |
| Drink | random_6103 | training_mean | 0/6 | 0/102 | 9/102 | 9/108 | 7.525 |
| Drink | random_6103 | zero | 0/6 | 0/102 | 10/102 | 10/108 | 7.544 |
| Two hand wave | functional_2 | training_mean | 6/6 | 0/102 | 27/102 | 33/108 | 3.867 |
| Two hand wave | functional_2 | zero | 6/6 | 0/102 | 54/102 | 60/108 | 2.522 |
| Two hand wave | random_6101 | training_mean | 2/6 | 0/102 | 12/102 | 14/108 | 7.769 |
| Two hand wave | random_6101 | zero | 3/6 | 0/102 | 13/102 | 16/108 | 6.434 |
| Two hand wave | random_6102 | training_mean | 4/6 | 0/102 | 7/102 | 11/108 | 7.301 |
| Two hand wave | random_6102 | zero | 3/6 | 0/102 | 19/102 | 22/108 | 5.438 |
| Two hand wave | random_6103 | training_mean | 0/6 | 0/102 | 9/102 | 9/108 | 7.481 |
| Two hand wave | random_6103 | zero | 0/6 | 0/102 | 10/102 | 10/108 | 7.469 |
| Forward Kick | functional_8 | training_mean | 4/6 | 0/102 | 31/102 | 35/108 | 3.572 |
| Forward Kick | functional_8 | zero | 6/6 | 0/102 | 56/102 | 62/108 | 2.423 |
| Forward Kick | random_6101 | training_mean | 0/6 | 0/102 | 14/102 | 14/108 | 7.733 |
| Forward Kick | random_6101 | zero | 0/6 | 0/102 | 16/102 | 16/108 | 6.398 |
| Forward Kick | random_6102 | training_mean | 0/6 | 0/102 | 11/102 | 11/108 | 7.048 |
| Forward Kick | random_6102 | zero | 0/6 | 0/102 | 22/102 | 22/108 | 5.363 |
| Forward Kick | random_6103 | training_mean | 0/6 | 0/102 | 9/102 | 9/108 | 7.481 |
| Forward Kick | random_6103 | zero | 0/6 | 0/102 | 10/102 | 10/108 | 7.406 |

For each functional group, off-target false positives are 0/102 under both references, so the target success is not a blanket tendency to classify unrelated clips as that target on this cohort. In particular, zero-reference functional retention 52 recognizes all 6 Kick clips while rejecting Kick on all 102 other clips. This provides a stronger bounded target-discrimination result than the target-only retention curve. It does not preserve the full action classifier: other correct counts are only 56/102 (Kick),54/102 (Wave),75/102 (Drink) under zero, compared with baseline93/102,90/102,90/102 respectively. Mean-reference overall correct counts are35/108,33/108,49/108 respectively. All random retention 52 groups reject these targets too, but Kick sensitivity is0/6 for all three under both references, and overall accuracy is severely reduced (9-22/108 across random/reference combinations). Specificity alone is therefore insufficient; a model that never predicts Kick can reject all non-Kick clips without recognizing Kick.

The functional 52 group is a training-selected candidate representation with demonstrated conditional target recognition and rejection under these synthetic reference conditions. This is not proof of minimality, natural-intervention sufficiency, semantic channels, or a complete circuit. Off-target recognition damage and reference dependence remain material. No ranking/size was reselected using validation and no corrected deployed model is proposed. All108 follow-up source/cache checks pass; baseline replay max absolute logit error is 0; protected hashes and in-memory variable digests remain unchanged. Follow-up inference execution after loading took 6.9 seconds. Raw rows, logits, and summary are in retention_selectivity_interventions.json/CSV, retention_selectivity_logits.npz, and retention_selectivity_analysis.json/summary.csv.
