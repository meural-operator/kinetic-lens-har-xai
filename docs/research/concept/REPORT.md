# Extended KARD concept proxy audit

Semantic validation remains **PARTIAL**. Action-family labels are readable proxies derived from action identities; there are no independent semantic annotations. The study now provides stronger evidence about probe transfer and the deployed head's sensitivity to fitted feature directions. It does not establish that these directions correspond specifically to waving motion, leg motion, objects, or pose.

## Fixed model, cohorts and scope

The deployed model is the unchanged official frozen MoViNet A0-Stream endpoint with the selected KARD 5-fps Dense18 head. Video policy: RGB float 32 [0,1], bilinear resize with padding to 172 x 172, 5 fps, at most 50 frames, entire clip if at most 10 seconds otherwise centered 10-second window, streaming chunks 8. The head uses saved train-only standardization and dropout only during HAR training; dropout is disabled at inference. KARD did not train the backbone or original Kinetics classifier. This matters: proxy separation is measurable in a generic pretrained action representation, with the KARD head providing the task-specific readout.

Probes fit 324 clips from subjects 1/3/4/6/7/8. Validation uses 108 clips from subjects 5/10, each 54. This cohort was previously used for HAR checkpoint/penalty selection and the pilot; it is reused validation, not an untouched independent confirmation set. The 108 test clips from subjects 2/9 are used only for cache/model integrity and saved-logit equivalence. They contribute no probe metrics, choices, or figures. No HAR training, backbone updates, camera access or video resampling took place.

The proxies are arm waving (Horizontal arm wave, High arm wave, Two hand wave), kicking (Forward Kick, Side Kick), object interaction (Catch Cap, High throw, Toss Paper, Take Umbrella, Phone Call, Drink), and posture transition (Bend, Sit down, Stand up). These groups are not independently annotated motion/pose/object labels. Object interaction is especially heterogeneous.

## Design and calibration

PLAN.md was written before execution. This is exploratory follow-up after observed pilot results. All alphas 0.1/1/10/100 are reported, with no winning penalty selected and no test tuning. Dual ridge regresses +/-1 targets on 480 saved-standardized features, using an unpenalized intercept and unweighted squared loss. Alpha 10 is the prior fixed pilot setting, retained for direction/intervention analysis.

For every full-family and leave-positive-action-out fit, six training subject folds produce out-of-fold scores. A threshold maximizes balanced accuracy on those training predictions only; ties prefer closest to zero, then smaller value. The final probe fits all eligible training clips and uses that threshold on validation. This is **decision threshold calibration**, not probability calibration. OOF training scores are used to choose the threshold and their best BA is optimistic as a calibration objective, not an independent performance estimate. Normalization remains the deployed train-only transform, which uses all original 324 training clips, including each held training subject and omitted positive action. Hence folds audit the probe conditional on the fixed deployed representation/normalization, not the full training pipeline. Omitted actions are excluded from auxiliary coefficient fitting and threshold calibration, but not retroactively from the existing HAR normalization/head.

## Family scores and per-subject differences

Alpha 10 results; BA means the average of sensitivity and specificity:

| Proxy | AUC | BA zero | BA train threshold | Sensitivity calibrated | Specificity calibrated |
| --- | --- | --- | --- | --- | --- |
| Arm waving | 0.985 | 0.833 | 0.867 | 0.778 | 0.956 |
| Kicking | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Object interaction | 0.988 | 0.917 | 0.931 | 0.889 | 0.972 |
| Posture transition | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

High AUC can coexist with missed positives at a fixed cutoff. At alpha 10 calibration detects two additional waving positives while admitting four negatives, and one additional object-interaction positive without extra false positives. It does not alter AUC, the classifier head, or model probabilities. At alpha 100 full waving/object AUC reaches 1; these observed results are reported without selecting alpha 100. Across all fixed penalties, full waving AUC ranges 0.909-1.000 and object interaction 0.885-1.000; kicking stays 1 and posture 0.999-1.

External subject-specific metrics use the train-only calibrated alpha 10 threshold:

| Proxy | s 05 AUC | s 10 AUC | s 05 BA | s 10 BA |
| --- | --- | --- | --- | --- |
| Arm waving | 1.000 | 0.956 | 0.989 | 0.744 |
| Kicking | 1.000 | 1.000 | 1.000 | 1.000 |
| Object interaction | 0.994 | 1.000 | 0.972 | 0.889 |
| Posture transition | 1.000 | 1.000 | 1.000 | 1.000 |

Only two validation subjects are available. Do not treat their clips or three repeated takes per action as independent population samples. No subject-level significance or population confidence intervals are reported.

## Action transfer

Each of 14 positive actions is removed completely from auxiliary training and training OOF calibration. Validation positives are its six clips; negatives are all clips from the family's negative actions. Thus each probe is tested on a new positive action identity, but shared appearance, acquisition and motion can still explain transfer. The positive count is six in each row, and negative counts are 90/96/72/90 for waving/kicking/object/posture respectively. All action results, all penalties and both thresholds are in action_transfer.csv.

| Omitted action | AUC alpha 10 | AUC all-alpha range | BA zero | BA calibrated | Sens calibrated | Spec calibrated |
| --- | --- | --- | --- | --- | --- | --- |
| Horizontal arm wave | 0.650 | 0.574-0.793 | 0.500 | 0.583 | 0.167 | 1.000 |
| High arm wave | 0.911 | 0.809-0.957 | 0.750 | 0.683 | 0.500 | 0.867 |
| Two hand wave | 1.000 | 1.000-1.000 | 0.917 | 0.972 | 1.000 | 0.944 |
| Forward Kick | 1.000 | 1.000-1.000 | 0.583 | 0.583 | 0.167 | 1.000 |
| Side Kick | 1.000 | 0.997-1.000 | 0.750 | 0.917 | 0.833 | 1.000 |
| Catch Cap | 0.993 | 0.988-0.998 | 0.903 | 0.972 | 1.000 | 0.944 |
| High throw | 0.940 | 0.875-0.991 | 0.667 | 0.750 | 0.500 | 1.000 |
| Toss Paper | 0.975 | 0.856-0.995 | 0.667 | 0.826 | 0.667 | 0.986 |
| Take Umbrella | 0.956 | 0.523-1.000 | 0.736 | 0.736 | 0.500 | 0.972 |
| Phone Call | 0.884 | 0.708-0.917 | 0.736 | 0.729 | 0.500 | 0.958 |
| Drink | 0.743 | 0.731-0.965 | 0.639 | 0.708 | 0.500 | 0.917 |
| Bend | 1.000 | 1.000-1.000 | 0.500 | 0.750 | 0.500 | 1.000 |
| Sit down | 0.996 | 0.994-1.000 | 0.833 | 0.833 | 0.667 | 1.000 |
| Stand up | 1.000 | 0.998-1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Horizontal arm wave has AUC 0.650 at alpha 10 and 0.574-0.793 across penalties: the failure survives these choices. Drink has alpha 10 AUC 0.743 but 0.731-0.965 across penalties: this failure is strongly estimator-dependent. Calibration raises alpha 10 Horizontal BA 0.500 to 0.583 and Drink 0.639 to 0.708, while leaving ranking unchanged. Bend has perfect ranking at every penalty but its zero cutoff labels all six positives negative; alpha 10 train-only calibration recognizes three (BA 0.750), while other penalties remain BA 0.500. Threshold shift cannot be mistaken for robust transferable semantics. Kicking and posture transfer ranking is consistently high (minimum AUC 0.9965 and 0.9944), which is evidence for broader action-family association conditional on this cohort, not semantic confirmation.

## Direction stability and class-derived label circularity

At alpha 10 we resample six whole training-subject blocks with replacement 32 times (seeds 101/202/303/404; eight per seed). Each block has 54 clips. Direction cosines compare unit-normalized resample coefficients to the original all-training direction. Repeated identical resamples are valid bootstrap draws. Spread describes dependence on this six-subject sample; it is not an inferential population confidence interval.

| Proxy | Median bootstrap cosine | Bootstrap cosine range | Action-span squared norm | Head contrast cosine |
| --- | --- | --- | --- | --- |
| Arm waving | 0.733 | 0.519-0.869 | 0.261 | 0.336 |
| Kicking | 0.734 | 0.549-0.905 | 0.218 | 0.209 |
| Object interaction | 0.716 | 0.461-0.910 | 0.215 | 0.262 |
| Posture transition | 0.789 | 0.585-0.923 | 0.284 | 0.262 |

High ranking is compatible with appreciable direction changes: bootstrap median cosine is 0.716-0.789, and the lowest observed cosine 0.461. Across penalties, cosine with the alpha 10 reference can be as low as 0.786-0.825. Two equally accurate separating directions need not perform the same intervention.

Training action means centered about their mean span rank 17. Alpha 10 directions place 21.5-28.4% of their squared norm in this subspace versus 3.71% for the 128 ambient random directions. This overlap identifies action-related geometry. It cannot discriminate semantic action properties from action-correlated nuisance variables. The remaining coefficient energy is not evidence of semantic independence: ridge applies inverse covariance to between-class signal and can use within-class correlated coordinates to improve separation. Head family contrast is the mean of positive HAR class columns minus mean negative columns; its cosine 0.209-0.336 with a probe direction measures readout agreement, not concept meaning. Pairwise CAV cosines range-0.344 to 0.041 for distinct families; the directions are not an orthogonal independent concept basis.

There is a direct label-identifiability problem. If each training feature is centered within its action, its covariance with each proxy target is numerically zero (max 8.39e-17), because every action's target is constant. Consequently these labels supply **no within-action attribute variation** for discovering whether an object was visible, a particular limb moved, or a posture was actually present. This algebraic fact does not disprove semantic encoding; it shows why present data cannot identify semantic specificity beyond action membership.

## All-validation deletion and matched controls

For unit direction v, standardized vector z, head logit ell_c=z^T W_c+b_c, deletion is z'=z-(z^T v)v. Its exact effect is delta ell_c=-(z^T v)(W_c^T v). This intervention removes a one-dimensional projection anchored at the train mean. It is not a video edit; it can leave the feature data manifold. Standard final-logit TCAV is mathematically degenerate here: its derivative W_c^T v is constant across every clip and its sign fraction is 0 or 1. The pilot verified this by autodiff; this study preserves the same endpoint/model and does not reinterpret that sign fraction as samplewise semantic reliance. Probability sensitivities would differ because softmax couples logits, but are not substituted for standard logit TCAV.

We now use every 108 validation clip, not only the pilot's three demonstrations. The primary target is always the **true action logit**, even where the baseline head is wrong. Baseline validation accuracy is 96/108 (88.89%). This makes drops on incorrectly classified clips evidence about the true-class readout, not necessarily about the originally selected decision. Full CSVs contain both predictions, true probability, projection, margin and accuracy change.

Controls use 128 random unit directions in 480-space (seed 505), and 128 directions confined to the 17-dimensional training action-centroid subspace. For each clip and proxy, match perturbation L 2 exactly to the proxy deletion |z^T v|; signs point toward removing projection on the control direction. The latter is an equal-magnitude directional displacement, not necessarily full removal of its own projection. Radial shrinkage also matches that L 2 and shrinks all feature coordinates toward the train mean. Unmatched ordinary random projection deletions are saved separately to expose magnitude effects. No null p-values are assigned: random directions and repeated clips are not independent subject samples, and none of these controls is verified semantic ground truth. Plotted intervals are the 2.5/97.5 percentiles across 128 fixed control directions, not uncertainty on human concept effects.

| Proxy | Family positive n | Family mean true-logit drop | Matched ambient mean | Matched action-span mean | Changed predictions /108 | Net correct count change /108 |
| --- | --- | --- | --- | --- | --- | --- |
| Arm waving | 18 | 0.397 | 0.025 | 0.180 | 1 | -1 |
| Kicking | 12 | 0.708 | 0.049 | 0.372 | 0 | 0 |
| Object interaction | 36 | 0.129 | 0.013 | 0.097 | 4 | -3 |
| Posture transition | 18 | 0.587 | 0.043 | 0.308 | 1 | -1 |

The proxy drop exceeds the mean ambient random drop for every family, establishing head sensitivity beyond an arbitrary equal-size ambient displacement. Confining controls to action-related structure substantially narrows the gap. The radial control often produces larger true-logit drops, so reducing evidence amplitude is also sufficient to create large logit effects. Controls do not demonstrate that a proxy is unique, necessary, sufficient, or semantically faithful. A changing logit/probability does not imply a changed prediction; competing classes move too. A positive average hides samplewise signs and class differences, provided in validation_deletions.csv.

## Evidence boundaries and next validation

Supported: action-family association in the frozen representation, weak/penalty-sensitive cross-action transfer, threshold-dependent decision performance, nonunique fitted directions, and exact directional sensitivity of the deployed HAR head. Unsupported: independent semantic concept discovery, body-part or object localization, absence of background/identity confounds, human concept causal reliance, completeness, physically realizable deletion, and generalization to a new dataset or population. No new semantic descriptor labels were derived or asserted.

The next informative dataset would independently annotate observable attributes (e.g. visible held object, arm above shoulder, knee extension, torso flexion) within both positive and negative action identities, using raters blind to model outcomes and documented agreement. Hold out entire actions and subjects, retain counterexamples where attribute and action disagree, and evaluate earlier nonlinear layers plus realistic video counterfactuals. Until that variation exists, stronger separability cannot solve the identifiability problem.

## Reproducibility and verification

Run concept_extended_experiment.py then concept_extended_author.py with python -B. No model candidates were trained or selected. Initial failed execution exposed missing SciPy; NumPy eigendecomposition replaces it. A mistaken 100% validation assertion was removed before final execution; it did not change any experimental choice. Final baseline accuracy is recorded honestly. Data manifests/cache label/subject fields are checked, all protected model/source files are SHA 256-equivalent before/after, and standalone TeX is compiled using the built-in compiler. Figures are supplied as vector PDF/SVG and PNG; each final PDF is rendered and visually checked. provenance.json records package versions, plan/script/cache/model hashes and exact numerical errors.

Analytical versus Keras logits max error 5.21e-06; saved test logits error 0; common suite clips error 3.81e-06; deletion identity error 1.33e-15; equal-L 2 matching error 1.78e-15.

Methodological references: Kim et al. use directional derivatives to relate user-defined concepts to predictions ([TCAV, ICML 2018](https://proceedings.mlr.press/v80/kim18d.html)). Pahde et al. distinguish a good classifier's separating direction from the underlying concept signal and show how distractors can change it ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/852870d56e20a22fcd52ea14523fb820-Abstract-Conference.html)). Those general results motivate controls; they do not verify the semantics of KARD proxy directions.
