# What the contrastive explanations mean

The selected model answers 96 of 108 validation clips correctly. This extension explains every clip, including all 12 mistakes. The model is unchanged: a frozen official A0 Stream video backbone produces 480 learned features, and the selected 5-fps KARD classifier turns those features into 18 scores. No feature index is a named body part or action concept.

## Read a contrast as a specific question

“Why Drink rather than Phone Call?” concerns only those two scores. A positive margin means Drink has the larger score. It does not mean Drink wins the entire classification, and it is not a calibrated probability. The runner-up question uses the actual second-highest score and is closer to the model's current decision boundary. For an incorrect prediction, this report also asks why the true label lost to the predicted one.

The three semantic comparisons were fixed before the extension: Drink/Phone Call, Two hand wave/High arm wave, and Forward Kick/Side Kick. There are 36 validation clips from those six actions. Every true-class-oriented semantic margin is positive. Yet 5 of those clips are wrong overall: all three subject-10 Forward Kick clips lose to Walk, and two subject-10 High arm wave clips lose to Draw Tick. The semantic comparison therefore exposes a limitation in the question itself. The Forward Kick failures are not evidence that the model cannot separate Forward Kick from Side Kick; the observed failure is that it ranks Walk higher.

## Positive and negative evidence are both active

For class c against d, each coordinate contributes (W_ic - W_id) times z_i. Here z_i measures how far that feature lies above or below its training-pool mean in training-standard-deviation units. A positive product favors c over d relative to this reference. A negative product favors d over c. A below-mean feature can be positive evidence when its weight difference is negative. “Positive” is not the same as “more activation,” a visible object, or a good prediction.

All 480 terms must be added, along with b_c - b_d. That final bias is the pair's score difference at the training-mean feature vector. It is not the class frequency, an independent physical cue, or automatically a harmful statistical bias. It may increase or decrease the final margin. The training-mean feature vector itself need not correspond to an actual video.

The calculation is exact for the inference head: the largest reconstruction error over 156 contrasts is 3.85e-06 logit units. That establishes arithmetic faithfulness to the classifier. It does not establish that the backbone has represented the correct action mechanics.

There is substantial cancellation. Across runner-up contrasts, on average 88.2% of the absolute signed evidence cancels before the net margin is formed. The top 20 absolute terms contain 23.9% of total absolute evidence on average. A short bar chart is therefore a selected subset, not a complete explanation. Ranking by absolute size mixes evidence for both classes; ranking only positive terms answers a different question. A margin can be small despite strong evidence on both sides.

## What deletion and retention establish

Deleting a coordinate here means setting it to the training mean, not removing a visible hand, frame or movement. For this affine head, deleting a collection subtracts exactly its signed term sum. Removing positive terms lowers the c-versus-d margin; removing negative terms raises it. Retention keeps only the selected signed terms plus the bias. The exported curves compare this with 100 seeded same-size random groups, including groups matched to the sign of the selected terms.

Deleting the 10 largest positive terms reverses the runner-up comparison in 61/108 clips. This is useful for understanding score dependence in the chosen coordinates. It follows from the ordering of exact additive terms, so faster deletion than random is not an independent discovery of semantic importance. It does not prove a realizable video counterfactual, and it can create unnatural combinations of correlated coordinates.

Training data contain 28 coordinate pairs with absolute correlation at least 0.8, producing 456 connected groups, the largest containing 6 coordinates. Group totals provide a sensitivity check, not a decomposition into independent causal factors. The group control matches the number of groups; its coordinate counts vary and are exported. The chosen threshold is descriptive, not a learned optimum.

## The reference and coordinate system matter

Changing the reference changes what counts as above or below normal. Replacing the global training mean with a training-subject mean changes 11.4% of coordinate signs on average; training-class means change 20.6%. The corresponding mean top-20 Jaccard overlaps are 0.618 and 0.402. The intercept must change at the same time. With that correction every margin is preserved; only the allocation of evidence changes. A training-class reference asks how this clip differs from an average training example of that class, rather than how it differs from an average training clip.

A harmless mathematical rotation of every consecutive feature pair also preserves the classifier score while changing individual evidence terms. Mean top-20 mass for runner-up contrasts changes from 23.9% to 22.8%. Coordinate evidence belongs to this representation basis. It is not a unique inventory of natural action causes. Cross-subject semantic profiles also vary: signed cosine similarities range from 0.303 to 0.637 and mean top-20 Jaccard overlaps from 0.158 to 0.350 across the six oriented pairs. These are descriptive comparisons of three repetitions from each of two subjects, not independent population estimates.

## A pair boundary is not necessarily a successful counterfactual

The closest standardized-coordinate move to equality between c and d has a closed-form solution. For the 36 semantic contrasts, a third class scores higher at that equality in 16 cases. For the 108 actual runner-up contrasts, no third-class winner appears at the closest pair boundary in this cohort; that empirical result is not a general theorem.

We also solve the small convex head problem that makes the chosen alternative beat every other class by 0.00001 logit units. For error contrasts the chosen target is the true class, so the projection diagnoses the actual error. Mean semantic-pair displacement grows from 3.371 to 3.494 standardized L2 units when all 18 classes must be beaten. Solver feasibility and the primal-dual gap are verified; actual Keras inference checks the reconstructed features. This is a minimal head-coordinate displacement under the stated norm and constraints. It has no video generator, motion constraint, or body-part interpretation. Distances depend on scaling and ignore covariance. Nearest-training-feature distances are recorded as a diagnostic, not proof of realizability.

## Spatial maps need a separate check

The original signed maps use margin gradients at the b4/l3 endpoint, a 6x6 spatial grid across sampled time. Red and blue indicate the signs of a gradient-weighted activation visualization. The map does not partition the exact margin into pixel or body-part contributions. Its channels are averaged over time and space; temporal receptive fields mix frames. Each panel's color scale is separate.

We tested the same three original clips by actually blurring eight spatial cells through every sampled frame, using a 31x31 Gaussian replacement. Random masks match the exact pixel area. On Drink, blurring the top signed cells lowers the Drink-Phone Call margin by 1.880, while bottom signed cells raise it by 3.412. However, both top-matched random controls lower it more (6.124 and 5.934), so the top signed map does not outperform those controls. On Two hand wave, top signed blur lowers the wave margin by 7.265 and reverses it, while the two matching controls lower it by 0.186 and 1.858. But blurring its eight negative-map cells lowers the margin by 0.546, contrary to the simple negative-evidence interpretation. On Forward Kick, top blur lowers the pair margin by 0.192 and bottom blur raises it by 0.804; both top-matched random masks instead raise it by 0.231 and 0.531. The full intervention table includes every control. Read all outcomes together; no selective map-success claim is supported.

A valid altered RGB input is useful evidence about this model under this replacement, but blur can change the input distribution and the backbone's features in nonlinear ways. It does not establish causal real-world action cues. Two random masks per tail and three clips cannot establish significance or validation-wide spatial faithfulness. If a map has no negative time-averaged cells, “bottom signed” means weaker positive visualization values, not negative evidence.

## Use the results for diagnosis

Start with the true and predicted labels and all 18 logits. Inspect true-versus-predicted evidence on errors; inspect runner-up evidence for near decisions. Then examine whether a prespecified semantic question addresses the actual competing class. Compare evidence across subjects, references and repetitions before naming a recurring signal. Finally, use controlled input interventions to challenge a spatial interpretation and keep failures visible.

This study supports exact classifier-coordinate accounting and descriptive error analysis. It does not establish semantic feature names, causal body parts, a full explanation of the backbone, or a novel explanation method. Validation had already selected this deployed head, and these two subjects are not a blind new cohort. No test features were used in this extension.
