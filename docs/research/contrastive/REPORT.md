# Extended contrastive audit of the selected KARD model

## Cohort, protocol and preserved model

PLAN.md was saved before experiments. Reference cohort: all 324 selected training clips from subjects 1, 3, 4, 6, 7 and 8. Evaluation cohort: all 108 validation clips from subjects 5 and 10, 54 per subject, three repetitions per each of 18 actions. Test caches were not loaded. Validation previously selected the deployed head; these are descriptive post-selection results.

Observed accuracy: 96/108 = 0.888889; errors: 12. Subject 5: 53/54; subject 10: 43/54. No correctness filtering. Contrasts: 108 predicted-versus-runner-up, 36 fixed semantic pairs with true-class orientation, 12 true-versus-predicted error contrasts. Some contrasts duplicate the same class comparison and must not be treated as independent clips.

Unmodified deployed model: official frozen A0 Stream, selected 5-fps/max-50-frame preprocessing, training-only normalization, 480-feature affine Dense(18) head. Inference dropout OFF. All loaded head weights remain bytewise equal in memory; backbone digest verified unchanged in the spatial run. No HAR training, weight writes, camera use or external publication.

Head SHA256: `e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70`. Backbone variable-data SHA256: `f35edf9b7b747b7fd5b65f127a34c988bec7c6602b1618af123b970d5847d6d2`. Protocol, manifest, normalization, source scripts, and all 432 loaded train/validation cache hashes are retained.

## Exact head accounting

z_i=(x_i-mu_i)/max(sqrt(v_i),epsilon); m_cd=sum_i[(W_ic-W_id)z_i]+b_c-b_d. Positive and negative signs are relative to the global training-mean reference. Reconstruction uses float64 products of the actual float32 normalization outputs and trained weights. Keras runs in its deployed inference arithmetic.

Maximum explicit-logit versus actual-head absolute error: 3.3245175e-06. Maximum pair-margin residual: 3.8503403e-06. Training standardized mean maximum absolute value: 2.6994388e-06. Normalization parameters match the saved training normalization. Actual head deletion checks at requested k=10,40,480 pass absolute error <1e-4.

| Contrast rule | N | Mean margin | Mean top-20 absolute mass | Mean cancellation | Third class above pair boundary |
|---|---:|---:|---:|---:|---:|
| predicted_vs_runner_up | 108 | 2.769547 | 0.238576 | 0.882180 | 0/108 |
| semantic_pair | 36 | 5.461787 | 0.250847 | 0.762680 | 16/36 |
| true_vs_predicted_error | 12 | -1.034529 | 0.238791 | 0.955647 | 1/12 |

Cancellation is 1-|sum coordinate terms|/sum absolute terms, excluding bias. Top-k concentration divides absolute selected terms by the total absolute coordinate sum. These describe the chosen feature basis; they are not probabilities or semantic importance.

## Random controls and signed deletion/retention

Requested k=[0,1,5,10,20,40,80,120,240,480]. Positive and negative tails are ranked separately. Actual k=min(requested k, number of coordinates of that sign). Deletion sets chosen standardized coordinates to zero; retention sets all other standardized coordinates to zero. Random controls sample without replacement 100 groups at each k, first from all 480 coordinates and separately from the matching-sign pool. Seed=20261010. Numeric audit NPZ retains all random signed sums; CSV retains means, standard deviations, empirical quantiles, actual k, and selected results.

Removing the ten strongest positive terms makes the actual runner-up comparison negative in 61/108 cases. Comparing largest signed terms with sign-matched random terms is an ordering property of exact additive evidence. It is not independent evidence of semantic or pixel faithfulness. Retention can retain positive-only evidence that exceeds the original margin because negative evidence was omitted.

Correlated coordinates: absolute Pearson correlation >=0.8, calculated on 324 standardized training features, connected components. There are 28 edges, 456 groups, largest group size 6. Group concentration uses signed group sums and ranks groups by absolute total. Random group controls match group count, not constituent coordinate count; both dimension counts are exported. Correlations do not establish independent factors, and the threshold was prespecified rather than optimized.

## Reference, basis and cross-subject stability

Alternative references: six training-subject means and the two relevant training-class means per contrast. At reference r, terms become (W_c-W_d)_i(z_i-r_i), and intercept becomes b_c-b_d+(W_c-W_d)^T r. This preserves the score exactly while reallocating evidence. No normalization or deployed model was changed.

Mean coordinate sign-change fraction: training-subject references 0.114296; training-class references 0.206150. Mean absolute top-20 Jaccard: 0.617570 and 0.401889. Maximum reference reconstruction residual: 3.55e-15.

Basis check: fixed orthogonal 45-degree rotations within consecutive coordinate pairs, with the weight difference rotated identically. Maximum preserved-margin residual: 3.55e-15. Runner-up mean top20 mass changes from 0.238576 to 0.227616. This is an analytic reexpression of the same score, not altered backbone weights.

| Semantic orientation | Clips | Cross-subject repetition pairs | Mean signed cosine | Mean absolute top20 Jaccard |
|---|---:|---:|---:|---:|
| Drink / Phone Call | 6 | 9 | 0.303166 | 0.185388 |
| Forward Kick / Side Kick | 6 | 9 | 0.440843 | 0.184619 |
| High arm wave / Two hand wave | 6 | 9 | 0.324575 | 0.174096 |
| Phone Call / Drink | 6 | 9 | 0.433067 | 0.158023 |
| Side Kick / Forward Kick | 6 | 9 | 0.469772 | 0.231039 |
| Two hand wave / High arm wave | 6 | 9 | 0.636790 | 0.350378 |

Each orientation has six clips (three per subject) and nine cross-subject pairings. Repetitions and pairings are dependent. No inferential confidence intervals or population-level stability claims are made. All action accuracy counts and error clips appear in analysis.json.

## Pair boundary versus multiclass head counterfactual

Pair equality: dz=-m_cd (W_c-W_d)/||W_c-W_d||^2. This is the minimum standardized Euclidean displacement to that hyperplane. All 18 logits are evaluated there. Third-class-above uses advantage >1e-7 in float64 head geometry. Actual Keras pair margins are independently checked <1e-4.

For semantic and runner-up contrasts the multiclass target is d. For true-versus-predicted errors the target is c (the true class). Solve min 0.5||dz||^2 subject to (W_target-W_j)^T dz >= 1e-5 - (logit_target-logit_j) for each of the other 17 classes. The target must win all classes by a prespecified tiny advantage. Dual coordinate descent uses 17 nonnegative multipliers, with no new solver dependency. Stationarity, primal feasibility, complementarity and primal-dual gap are exported for every solution. Convex KKT verification certifies the minimum within numerical tolerance, not video realizability.

Maximum feasibility violation 4.91e-11; maximum primal-dual gap 5.71e-11; minimum actual-Keras target advantage 9.54e-07. Maximum actual-Keras pair residual 8.58e-06. The float64 constraint target is 1e-5; small float32 round-off is measured rather than hidden.

| Contrast rule | Mean pair L2 | Mean target-winning multiclass L2 |
|---|---:|---:|
| predicted_vs_runner_up | 1.644244 | 1.644250 |
| semantic_pair | 3.370517 | 3.494465 |
| true_vs_predicted_error | 0.590223 | 0.605615 |

Sixteen of 36 semantic pair boundaries have a third-class winner; none of 108 runner-up boundaries do in this cohort. Error corrections target the true class explicitly. Distances are in standardized learned coordinates, with no covariance metric, image-domain restriction or generator. Nearest-training distances are diagnostic only; coordinates may leave the realizable-video feature manifold.

## Input perturbation crosscheck of signed coarse maps

The same three original validation clips are retained. Prior exact downstream signed Grad-CAM at b4/l3 [1,T,6,6,104] supplies the map. Gradient channel weights are averaged across space and time; no ReLU is applied. Masks rank time-averaged 6x6 values, choose eight top or bottom cells, and apply a 31x31 Gaussian blur in those fixed cells through every sampled frame. Two seeded random masks per tail exactly match pixel area by matching cell-area bucket counts. Original/altered input arrays, extracted features, all 18 logits, cell sets and source frame indices are saved.

All three fresh baselines match deployed logits exactly; prior logits agree within saved residuals. Maximum altered-feature exact-head margin-change residual: 1.0365762e-05. Backbone and loaded head weights remain unchanged. The intervention is a valid RGB tensor transformation, not proof of a physically plausible altered action.

| Clip | Top signed blur | Bottom signed blur | Random top 1 | Random top 2 | Random bottom 1 | Random bottom 2 |
|---|---:|---:|---:|---:|---:|---:|
| Drink | -1.8803 | +3.4120 | -6.1237 | -5.9341 | -5.6962 | -6.8410 |
| Two hand wave | -7.2646 | -0.5460 | -0.1855 | -1.8581 | -2.3415 | -1.3095 |
| Forward Kick | -0.1918 | +0.8043 | +0.2306 | +0.5312 | +0.7503 | +0.6092 |

Entries are new margin minus baseline, in logits. A decrease after top-positive blur is directionally consistent with removing c-over-d evidence; an increase after truly negative-cell blur is directionally consistent with removing d-over-c evidence. Global map sign does not guarantee these effects because the backbone and downstream project are nonlinear, blur can add evidence, and map cells are not exact additive partitions. Both signs of each selected tail and available negative cells are exported. “Bottom signed” is not “negative” when the selected cells are positive.

Drink top blur has the expected sign but less margin drop than both matching random masks; bottom blur raises the margin. Two hand wave top blur reverses the pair margin and exceeds both top-matched random drops, but bottom blur of eight negative-map cells lowers it by 0.546 rather than raising it: the simple map-sign interpretation fails. Forward Kick top blur lowers the margin by 0.192 and bottom blur raises it by 0.804, while top-matched random masks raise it by 0.231 and 0.531. Every control is reported without filtering. All three top selections are eight positive time-averaged cells; all three bottom selections are eight negative cells. The mixed outcomes support limited intervention-specific observations; three clips and two controls per tail do not support significance or dataset-wide map faithfulness.

## Error diagnosis

Eleven of twelve errors occur on subject 10. The true label is runner-up in 10/12 errors. Subject-10 Forward Kick is predicted Walk in all three repetitions despite positive Forward-Kick-versus-Side-Kick margins; true-versus-predicted evidence identifies the relevant failed comparison. Draw X is confused with Draw Tick twice; High arm wave with Draw Tick twice; remaining errors are fully retained in analysis.json. Model errors and semantic-pair successes coexist.

## Limitations and evidence level

- Exactness applies to the unchanged affine inference head and finite learned-coordinate interventions, not the complete frozen backbone mechanism.
- Coordinate signs/ranks depend on reference, scale, basis and correlated features. Feature indices do not name concepts or body parts.
- Pair and multiclass counterfactuals are minimum-norm classifier-coordinate calculations, not realizable-video or semantic-action counterfactuals.
- The deployed head was previously selected with this validation split. Two subjects and dependent repetitions constrain generalization; no test feature use or new blind test claim.
- Spatial maps are local, coarse, gradient-weighted visualizations, not exact spatial decomposition. Blur changes the distribution and has temporal effects.
- Three spatial illustrations, two random masks per tail, no map-success filtering, no significance testing or dataset-wide spatial causality claim.
- Standard affine-head contrasts, gradient localization and convex projection are applied here without a novelty claim.

## Primary literature and method context

The signed-map variant uses the gradient-weighted localization construction of [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391), retaining signs rather than applying the original positive ReLU. [Adebayo et al., Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) motivates checking dependence on model behavior rather than judging a map by appearance; this extension is not a full model-randomization sanity-check suite. [Wachter et al., Counterfactual Explanations](https://arxiv.org/abs/1711.00399) provides context for changing an output through interventions; the present coordinate projections deliberately do not claim actionability or physical plausibility.

## Reproduction and artifacts

Run the uniquely named contrastive_extended_analysis.py, contrastive_extended_spatial.py and contrastive_extended_publish.py with python -B from the workspace. runtime.py uses TensorFlow intra/inter threads 4/1; BLAS/OMP threads are set to 4. Outputs are confined to this new folder. The scripts retain source hashes, deterministic seeds, raw CSV/JSON/NPZ and figures. PDF pages are rasterized for visual QA; standalone TeX compilation is recorded separately.