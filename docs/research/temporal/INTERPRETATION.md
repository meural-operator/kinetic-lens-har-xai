# What the temporal experiments mean

The extended audit shows how the same trained KARD model responds when we change a clip's frame sequence. It separates true action recognition from stability of the model's original answer. The baseline gets 96 of 108 validation clips right. Subject 5 contributes 53 correct clips out of 54, subject 10 only 43. The earlier three correct subject-5 examples therefore show useful local behavior, but they are insufficient to describe the whole model.

## Order and movement are related but different

Playing every sampled frame backward preserves the original target in 86.1% of clips and produces 78.7% true-label accuracy. Mean original-target drop is 0.345. A preserved label does not imply unchanged internal evidence: the logit and margin can move while the same class still wins.

Random reordering retains the original target in 22.2% on average over three fixed shuffles per clip, with 23.1% true accuracy and mean target-logit drop 1.999. This comparison preserves which frames are present. It disrupts both chronology and the smoothness of neighboring poses, creates unnatural transitions, and moves useful frames into different causal contexts. A greater shuffle effect than reversal is consistent with sensitivity to local continuity rather than a single indispensable forward timeline. It does not identify which of those changes caused the score difference.

Repeating one frame is a more severe change. It removes changing poses and scene coverage as well as motion, and the retained frame may depict a neutral or discriminative posture. The different static anchors are separate probes, not three independent clips. Their results in the report quantify dependence on the chosen posture; static failure alone cannot establish that the model recognizes optical motion rather than varied poses.

The affine brightness/contrast probe keeps chronology intact and scales each RGB temporal difference by 0.85. It retains original targets in 98.1%, with mean target drop 0.004. It checks one global appearance change. Even a small effect does not prove invariance to backgrounds, subjects, illumination or color generally. Even a large effect would not show that temporal evidence is irrelevant.

## Why a highlighted time segment is not a unique explanation

We divide each clip into five nearly equal frame-count parts and hold one part still. The resulting target-logit drop answers: how much did this particular substitution change the original answer's score? It does not answer: what percentage of the decision was caused by this time interval? Effects are non-additive, and changing one interval can affect later causal features and cumulative pooling.

In the old Drink clip, the fourth segment reduced Drink most strongly under two anchors. In Two hand wave, the second segment did. In Forward Kick, the fourth segment changed the answer to Stand up under those two anchors. We reproduced those numbers. On the expanded six-clips-per-class cohort, all four anchors identify the same strongest segment in only 9 of 18 clips; the original two anchor types agree in 16. Reference choice is therefore part of the result and must accompany every temporal highlight.

The global-middle contiguous substitution is more damaging than a same-budget dispersed substitution in 28 of 90 clip-segment comparisons. Their mean difference is -0.667. Dispersed replacements create many boundaries while contiguous replacements remove an uninterrupted span. Neither is guaranteed to be natural. This comparison helps diagnose sensitivity to perturbation structure; it is not a random-baseline significance test or proof that the highlighted span is uniquely important.

Local segment reversal removes no frames and changes 9 of 90 answers, versus 72 of 360 frame-repeat replacements. The operations have different severity and those denominators are repeated interventions on 18 clips. We cannot treat them as independent people or directly equate their rates with a controlled causal motion effect.

## Reading prefixes correctly

Each prefix point is the model's prediction after observing more of the clip, retaining causal memory and cumulative pooling. Original-target retention counts at the five endpoints are [3, 12, 15, 17, 18] out of 18. The final prefix agrees with full-clip inference, and resetting/recomputing every prefix gives the same result within numerical tolerance.

A rising curve means the growing representation becomes more favorable to the final original target. A dip can reflect new competing context or dilution of earlier pooled evidence. It cannot locate an event onset precisely: a prefix includes everything before it, and the sampled endpoint is one of five checkpoints. Adjacent score differences are not frame attributions. A positive margin means the target beats all rivals; a negative margin means another class wins.

## Originally wrong clips need a different reading

Targets are fixed to the model's original prediction, including wrong predictions. Only 15 of 18 segment clips are originally correct; Forward Kick is correct on 3 of its 6 validation clips. If a perturbation preserves a wrong target, the model is stable but wrong. If it changes to the true class, original-target preservation declines while accuracy improves. Both metrics are required. Consult class_controls.csv and subject_controls.csv before generalizing any pooled effect.

All three initially wrong segmentation clips are subject-10 Forward Kick clips predicted as Walk. Their segment plots explain changes in Walk scores. Averaging them with correctly recognized subject-5 kicks produces a true-class cohort summary of different original targets; it must not be called a mean Forward Kick evidence curve. Reversal preserves all six original answers for this class, including all three wrong Walk answers. Across classes, Stand up is much less reversal-stable (only 1 of 6 original targets retained), so the strong overall reversal retention has clear exceptions.

## What is established and what remains open

The audit establishes reproducible, controlled sensitivity of this exact frozen model to specified frame-level changes. Video/cache/model hashes, restored predictions, finite states, exact pool counts, streaming/fresh prefix equality and independent numerical replays passed. No retraining or alternate weights were used.

It remains open whether these perturbations reflect natural action evidence, whether important intervals align with human action phases, and whether the same behavior holds for new subjects. The next useful test is a prespecified untouched-subject evaluation with independently annotated action phases, realistic interpolation/time-warp controls, and equal-budget random spans. The findings should be described as temporal perturbation sensitivity and cumulative prediction behavior. They should not be presented as proven motion causation, calibrated confidence, attention or a uniquely correct causal explanation.

See REPORT.md for the exact method, numerical tables, limitations and primary-source references. Figures display all clips or clearly identified cohort means, with random repeats averaged within clip and no population confidence intervals.
