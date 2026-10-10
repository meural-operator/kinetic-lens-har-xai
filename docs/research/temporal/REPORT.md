# Temporal sensitivity of frozen MoViNet on KARD: validation extension

## Question and experimental status

Does the original three-clip result reflect stable chronology dependence or a mixture of dynamic pose coverage, streaming context and replacement reference? This descriptive follow-up keeps the trained model fixed and expands to all 108 saved validation clips for inexpensive controls, and all 18 validation clips in the three pilot true classes for segment/reference/prefix experiments. PLAN.md was written before new inference. No clips were selected by prediction, confidence or effect; no test clips, training, hyperparameter selection or sampling-policy changes were used. The saved validation set already participated in head selection, so these results describe sensitivity on that development split, not an independent generalization estimate.

## Model and preprocessing

Selected frozen official MoViNet A0-Stream plus the full-training 5-fps KARD head. Head file SHA-256: `e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70`. Backbone variable data file SHA-256: `f35edf9b7b747b7fd5b65f127a34c988bec7c6602b1618af123b970d5847d6d2`. Backbone endpoint is 480 pooled features; the fixed KARD head applies train-only standardization and Dense(18) logits, with dropout inactive. The original Kinetics classifier is not used for these KARD scores. RGB float32 [0,1] uses saved 5-fps timestamp sampling, bilinear resize with padding to 172x172, max 50 frames, centered 10-second window for longer clips. Input frame indices and source fps are preserved per clip. CPU: four intra-op and one inter-op threads.

## Cohorts, controls and score definitions

All 108 validation clips: 18 classes x subjects 5 and 10 x three repetitions. Segmentation cohort: Drink, Two hand wave and Forward Kick, six clips each. There are two subjects, not 108 independent people. Repetition dependence and original selection by validation performance rule out population inference here.

For baseline logits z, target t = argmax z is fixed throughout a clip. Drop is z_t - z'_t; positive means target score decreased, negative means it increased. Margin is z'_t - max(k != t) z'_k. Negative margin means the original target is no longer argmax. We also measure true-label accuracy, which can improve when an initially wrong target changes. Raw logits are uncalibrated; across-class logit magnitude is a sensitivity diagnostic, not a common evidence unit. A confidence drop and a label change answer different questions.

Controls preserve sequence length. Reverse preserves the entire frame multiset and reverses transitions and streaming position. Static first/middle/last removes most pose and scene variation, with different retained appearances. Three shuffles use seeds 20261010/11/12 independently per clip; exact permutations are saved. Shuffle retains frame coverage while breaking local transition continuity and changing order/streaming context together. Affine RGB is 0.85*x+0.075: chronology and frame correspondence remain intact, values stay in [0,1], temporal differences scale exactly by 0.85. It tests one global contrast/brightness change, not all appearance invariance.

Five segments use integer linspace edges from zero to sampled length. Four replacements repeat segment-first or global-first/middle/last frames. The repeated anchor can be unchanged; assigned_positions differs from changed_positions. Frames outside the segment and full length are identical. Local reversal preserves frames inside that segment while reversing order and changing boundaries. Three dispersed global-middle controls per segment length use the same assigned-position budget, seeds 20262010+100*j+repeat, and saved position indices. Dispersed and contiguous perturbations differ in boundaries/context, so their comparison does not isolate importance alone.

Five prefixes carry causal buffers and cumulative pooling forward. Independently recomputed fresh-state prefixes must agree. They classify the growing sequence, not individual frames. No Shapley, completeness or additivity property is claimed.

## Results: complete validation controls

| Control | True accuracy | Original target retained | Mean target-logit drop | Mean target margin |
|---|---:|---:|---:|---:|
| baseline | 88.9% | 100.0% | 0.000 | 2.770 |
| reverse | 78.7% | 86.1% | 0.345 | 2.281 |
| static_first | 10.2% | 11.1% | 4.781 | -3.460 |
| static_middle | 15.7% | 16.7% | 4.184 | -2.440 |
| static_last | 9.3% | 11.1% | 4.926 | -3.633 |
| shuffle | 23.1% | 22.2% | 1.999 | -3.263 |
| affine_appearance | 88.9% | 98.1% | 0.004 | 2.807 |


Each shuffle repeat is averaged within each clip before cohort summaries. Its accuracy can have a fractional correct-count equivalent; it is not a 324-subject sample. Baseline is 96/108 correct (88.9%). Subject 5 is 53/54 and subject 10 is 43/54. Forward Kick is only 3/6 correct initially. This imbalance matters: preserving an incorrect original target is not a successful recognition outcome.

| Control | Subject 5 accuracy | Subject 10 accuracy | Subject 5 retention | Subject 10 retention |
|---|---:|---:|---:|---:|
| baseline | 98.1% | 79.6% | 100.0% | 100.0% |
| reverse | 92.6% | 64.8% | 94.4% | 77.8% |
| static_first | 11.1% | 9.3% | 11.1% | 11.1% |
| static_middle | 16.7% | 14.8% | 18.5% | 14.8% |
| static_last | 13.0% | 5.6% | 14.8% | 7.4% |
| shuffle | 28.4% | 17.9% | 29.6% | 14.8% |
| affine_appearance | 100.0% | 77.8% | 98.1% | 98.1% |


Full class-specific retention, accuracy, margin and drop are in class_controls.csv. Six clips per class represent two subjects and three correlated repetitions. Fig. 1 separates accuracy and target preservation and shows score distributions and subjects; Fig. 2 exposes class heterogeneity. The three successful subject-5 pilot clips were a narrow subgroup, not a representative estimate.

Reversal yields 85/108 correct and retains 93/108 original targets; three shuffled trials total 75/324 correct and 72/324 original targets retained, summarized as 23.1% and 22.2% within-clip averages. Static anchors yield only 11/108, 17/108 and 10/108 correct. The affine probe yields 96/108 correct and retains 106/108 targets. The near-zero pooled affine drop (0.004) hides subject-specific means +0.120 and -0.112, so even this limited appearance response is not literally zero. Its two changed answers include one correction in subject 5 and one newly wrong answer in subject 10.

Order direction tolerance is heterogeneous: Stand up retains only 1/6 targets under reversal, while Forward Kick retains all 6/6, including three subject-10 clips originally predicted as Walk. Walk retains 16/18 targets across its three shuffle trials per clip, whereas Drink, Forward Kick, Phone Call, Draw X and Horizontal arm wave retain none. These are descriptive class contrasts, not independent class-level significance tests. They make a universal claim of chronology invariance untenable.

## Results: reference, local order and dispersed controls

Of 18 segmentation clips, 15 are initially correct. The strongest segment agrees across all four references in 9/18 clips, and between local-first and global-middle in 16/18. Mean pairwise within-clip segment rank correlation is 0.664. This is descriptive rank agreement over five bins, not a reliability coefficient with a sampling CI.

| True class | All four anchors agree (of 6 clips) | Local-first / global-middle agree |
|---|---:|---:|
| Drink | 3/6 | 6/6 |
| Two hand wave | 4/6 | 5/6 |
| Forward Kick | 2/6 | 5/6 |


Across 360 segment replacements, 72 alter the original target. Across 90 local reversals, 9 alter it; mean local-reversal drop is 0.707. The global-middle contiguous minus dispersed mean drop is -0.667, with contiguous greater in 28/90 clip-segment comparisons. These 90 comparisons are nested within 18 clips and two subjects. They do not establish a population effect or superiority over a random attribution method. Strongest bins are descriptive extrema over the evaluated bins, not prespecified semantic action phases. The three dispersed repeats estimate sensitivity to a few position draws, not a precise null distribution.

The negative contiguous-minus-dispersed mean is a useful mixed result: dispersed substitutions are more damaging on average, despite equal assigned-position budgets. Numerous scattered discontinuities and streaming interactions are plausible explanations. This weakens any claim that contiguous high-drop regions uniquely identify concentrated action evidence. The two original references agree fairly often (16/18), yet adding global start/end anchors reduces unanimity to 9/18. Agreement under two closely related references is not sufficient evidence of reference-independent localization.

The complete per-clip four-reference drops, strongest-bin indices, random-control averages and rank correlations are in summary.json. Figure 3 retains all 18 clips; Figure 4 shows class means and all prefix trajectories. Relative segment indices need not align with the same movement phase across repetitions.

The Forward Kick class mean combines three correctly recognized subject-5 Kick targets with three subject-10 Walk targets. It is a true-class cohort mean of original-target sensitivity, not a mean Forward Kick evidence curve. Even the Drink class-mean maximum is segment 2 under all four references, while four of six individual clips favor segment 4 under the two original references: large effects in a few clips and phase misalignment can separate means from per-clip rank counts.

## Prefix accumulation and original-pilot continuity

Original targets are retained in [3, 12, 15, 17, 18] of 18 clips at the five endpoints. Scores may rise or fall because new context changes causal features and the pooled representation. Even with a linear final Dense layer, nonlinear causal backbone features and evolving cumulative context prevent interpreting adjacent prefix score differences as isolated-frame contributions. A target becoming stable early indicates prediction stability within this clip; it does not identify movement onset or prove later frames are unused.

All 45 comparable prior-pilot intervention logit vectors agree within 0. The prior maxima remain Drink segment 4 (drops 2.052/2.401), Two hand wave segment 2 (2.488/2.580), Forward Kick segment 4 (2.822/3.585) for local-first/global-middle references. The Kick pilot target changes to Stand up under its strongest two replacements. Those observations are reproduced, but the extension qualifies their representativeness and reference robustness.

## Interpretation and what cannot be claimed

Different frame orders can preserve labels while moving target scores substantially. Reversal tolerance is compatible with useful pose-set coverage, repetitive actions, trajectory symmetry or accumulated features; it does not imply permutation invariance. Shuffling disrupts local motion continuity more drastically and can create artificial high-frequency transitions. Static failures are compatible with reliance on motion, but also with loss of pose diversity, anchor bias and off-distribution stationary inputs. The affine control provides a limited comparison with chronology-preserving appearance change; it cannot subtract all appearance confounds from shuffle/static effects.

A segment replacement measures the trained classifier's response to that particular altered input. It can propagate into later causal receptive fields and pooled evidence. Anchor disagreement demonstrates reference dependence, not a model bug. Negative drops can reveal removal of competing or misleading context, or artifacts that happen to support the target. An intervention that changes an originally wrong prediction is not evidence the removed region carried the correct class's evidence. Interpret initially wrong clips in terms of their actual original target.

These results cannot prove biological motion causation, unique semantic event localization, attention, minimal sufficient frames, explanation fidelity on natural inputs, calibrated confidence, or generalization beyond the saved validation split. No significance tests, independent-repeat CIs or novel-method claim are made. No temporal test tuning or weight changes occurred.

## Verification and reproducibility

Finite states/logits, exact pool counts, fresh-state inference for each full perturbation, video/cache hashes, cache-derived logits, restored baseline, and unchanged head/backbone variable digests and protected file hashes all passed. Maximum baseline cache error 5.72e-06; restored baseline error 0; streaming/fresh prefix error 8.58e-06; full/chunked endpoint error 3.81e-06. An independent process replays the three audit baselines and three new global-last segment-4 interventions; independent_verification.json contains errors. all_logits.npz equals JSON baselines and stores 1,584 intervention vectors. analysis.json includes all exact orders, indices, timestamps and provenance; all_measurements.csv is the complete score table.

Reproduce with the prescribed xai Python executable and -B: run_kard_temporal_extended.py, summarize_kard_temporal_extended.py, verify_kard_temporal_extended.py, document_kard_temporal_extended.py, render_kard_temporal_extended.py. Resuming requires identical plan/script/model hashes. The run log records completion and actual resource limits remain in runtime.py.

## Next falsifiable experiment

On newly held-out subjects, prespecify semantically annotated action phases and compare segment deletion/replacement against matched-budget random segments and temporally plausible interpolations, retaining originally correct and wrong strata. Independently annotate phase alignment before reading model scores. Test time-stride and smooth time-warp controls that preserve natural local transitions more closely; compare fresh-state segment-only predictions with carried-state prefixes to separate contextual accumulation. Before training any robustness variant, freeze the comparison policy and acquire an untouched subject cohort. A separate retraining-based evaluation could study information sufficiency under distribution matching, but it would answer a different question than this fixed-model audit. None of these next tests has been performed here.

## Methodological context and citations

MoViNet uses stream buffers and cumulative pooling for causal streaming; this motivates the strict reset and prefix-consistency checks [Kondratyuk et al., CVPR 2021](https://arxiv.org/abs/2103.11511). Explicit input perturbations support testable classifier sensitivity claims [Fong and Vedaldi, ICCV 2017](https://arxiv.org/abs/1704.03296). Removal can create distribution shift, confounding information loss with artifacts; ROAR addresses a different, retraining-based question [Hooker et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf). These papers provide methodological context, not evidence for the empirical KARD numbers above.
