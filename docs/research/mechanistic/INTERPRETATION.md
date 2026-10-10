# What the channel experiments explain

The experiments show that changing selected internal channels changes this model's predictions, but the results do not identify a small, complete action-recognition circuit. The stronger sensitivity-based groups also damage recognition of many other actions. The original weak Forward Kick result was partly a ranking limitation: a training-only gradient ranking finds a much larger effect on the successful Kick examples. That effect still does not establish a Kick-specific mechanism or explain the original Walk mistakes by itself.

## Picture the intervention as changing an internal control panel

The unchanged backbone turns each video into a sequence of 104 feature channels at a late convolutional layer. Each channel is a whole map across space and time, not a named feature such as "leg motion" or "cup." We replace selected maps, then let the original remaining layers and classifier process them. The video, backbone weights, and head weights stay fixed. This is a direct intervention on the model's computation.

A channel can differ between actions without being used strongly by the remaining network. Conversely, a channel used strongly by the classifier may matter for many actions. Activation separation asks which channels look different for Drink versus other training clips. Functional sensitivity asks which channels locally influence a Drink-versus-fixed-rival score. Those are different questions. Their top 12 groups share only two channels for each of the three actions.

The functional ranking uses gradients of a fixed training-selected rival contrast: Drink versus Draw Tick, Two hand wave versus High arm wave, and Forward Kick versus Side Kick. Validation measurement instead uses each target versus its strongest current alternative. The rival does not change during ranking and is never picked using validation. Thus functional selection is not a gradient ranking explicitly optimized for the validation Kick-versus-Walk failure. Gradient magnitude ignores whether the influence supports or opposes the target, and the gradient is local to an unperturbed activation. Actual removal is a large finite change through a nonlinear downstream network. The forward interventions, rather than the gradient alone, provide the reported evidence.

## Local computational cause is different from the cause of an action

Holding the video and weights fixed while replacing an internal tensor isolates a change in this deterministic computation. The difference in logits is caused by that replacement in the model. This establishes a bounded causal statement about the model under the specified reference, layer, channels, and input.

It does not establish that those channels represent the physical cause of drinking, that a person could naturally perform the same video with those channels removed, or that the model would behave this way on a plausible altered scene. Neither a constant training mean nor zero is a verified natural activation for that video. No real-world action variable is randomized here. A future natural donor intervention could reduce some artificiality, but a donor would bring its own subject, pose, timing, and scene differences; it would not automatically be a causal ground truth.

## Score importance and classification necessity are different

Target margin means the target logit minus the largest competing logit. A positive margin predicts the target. A removal drop means the replacement lowered that margin. A large drop can leave the target prediction correct if it began far ahead; a small drop can flip a borderline prediction. Negative drops mean the target margin improved.

For size 12 training-mean removal, activation-ranked Drink drops 1.282 margin units and retains 4/6 correct Drink predictions, compared with 6/6 originally. The functional group drops 4.702 and leaves 1/6 correct. Two hand wave drops 1.979 with activation ranking (5/6 correct) and 5.027 with functional ranking (3/6 correct). These findings support influence under the specified intervention. Neither group is required on every target example: some predictions survive removal.

The original pilot's 36-clip subset gave Drink and Wave activation groups larger mean drops than all five of its random groups. On all six validation examples per target, with three prespecified nested random permutations, their drops no longer exceed every random draw. Drink's activation drop is 1.282, whereas one random drop is 1.557; Wave's activation drop is 1.979, whereas one random drop is 2.209. The random groups differ from the earlier pilot, so the numerical difference combines cohort expansion with a different descriptive control set. The defensible conclusion is that the original strong comparison is not a blanket guarantee across this extension. No statistical significance is claimed from three permutations.

For Kick, activation-ranked size 12 removal changes the true margin by only 0.379 on average, without changing the 3/6 correct count. Functional removal drops 2.455 and leaves 1/6 correct. Stratification matters: its drop is 4.907 on the three originally correct s05 clips but 0.003 on the three originally incorrect s10 clips. The averaged effect mainly describes degradation of successful recognition. It provides almost no mean true-Kick-margin effect on the original failure cohort. This is evidence against presenting the same group as a complete explanation of those mistakes.

## The model's answer and the dataset label are separate explanation targets

All three s10 Kick examples originally predict Walk. Their Kick-minus-Walk margins are -0.686, -0.620, and -0.347. Asking why the model predicts Walk means tracking the original Walk prediction and its competitors. Asking which computation supports the correct Kick label means tracking the Kick score. These questions can have different answers on the same clip.

For example, some random size 12 removals make all six Kick clips correct, even though they were not selected for Kick. Their target margin drops are negative on average. Removing information can suppress a rival, change downstream interactions, or introduce a new score bias. Improvement after destructive intervention is an interesting computational counterfactual, but it is not proof that the removed channels were the cause of a semantic error. It also cautions against choosing a favorable validation group after seeing results.

Raw outputs include the true-label margin, original-prediction margin, target margin, all 18 logits, and Kick-Walk margin. These distinguish what the model originally believed from what it ought to predict according to the dataset.

## Why collateral effects weaken a specificity claim

For an off-target video, an intervention's Drink-margin change does not tell us whether its actual action becomes harder to recognize. The relevant check also examines its own true-label margin and final prediction.

Size 12 functional removal leaves only 41/102 other clips correct for the Drink group, 65/102 for the Wave group, and 55/102 for the Kick group. Their baseline off-target counts are 90/102 for Drink and Wave, and 93/102 for Kick. Corresponding activation-removal counts are 81/102, 90/102, and 89/102. Thus the sensitivity-based groups are much more destructive beyond their target actions. A stronger target effect does not imply a more target-specific circuit. The fixed non-target training controls also have substantial total gradient sensitivity, particularly Kick (88.663 versus 90.133 for target training clips). The score ranking may emphasize generally influential downstream channels.

In the collateral figure, blue bars measure those other clips' true-label evidence, while pink bars measure the same selected target's score on those clips. They can move differently. For example, replacing a Drink group can improve the Drink margin on unrelated clips while reducing evidence for their actual class. Calling that pattern "selective Drink localization" based only on a target-margin contrast would be misleading.

## Removal and retention test different conditions

Removal asks whether a group contributes when the other channels retain their natural values. Retention asks whether that group supports recognition when every other channel is replaced by the chosen reference. In a nonlinear network, those contexts are different. A group that matters in the intact network may fail by itself because supporting context and interactions have been disrupted. A group that predicts the target after retention may succeed because the replacement suppresses rivals or biases the head toward that target.

All ranked size 12 retention groups fail on all six target examples under both references. This refutes sufficiency for those groups under these two particular replacement conditions. It does not prove that there is no circuit, that those channels contain no relevant information, or that only 104 channels could suffice under every possible intervention.

Larger sets behave differently. With 52 functional channels retained, Drink is correct 4/6 and Wave 6/6 under both references. Kick is correct 4/6 with training-mean reference and 6/6 with zero reference, compared with 3/6 originally. In the mean case, all six Kick-minus-Walk margins become positive, but two s10 clips predict Draw X instead of Kick. Defeating Walk alone therefore does not guarantee correct recognition among all 18 actions. Restoring all 104 channels recovers 3/6 Kick accuracy: the original errors return. This nonmonotonic behavior exposes interaction or competing evidence, but it does not isolate the relevant pathway.

The observed size 52 success triggered an explicitly exploratory all-validation selectivity follow-up, recorded in PLAN_AMENDMENT.md before execution. It checks whether the retained groups still reject other actions rather than merely favoring the target. All original groups, references, and seeds remain fixed; no favorable size is promoted as a discovered circuit. See the follow-up results in REPORT.md.

## Why a reference can change the story

Training-mean replacement preserves each channel's average across equally weighted training clips, but discards video-specific variation and joint spatial/temporal structure. Zero replacement is simple and reproducible, but zero is not necessarily a neutral or typical activation. Both can violate the natural distribution of this late layer.

For Drink activation removal 12, the mean-reference drop is 1.282 and zero-reference drop 2.435. For functional Kick retention 52, the correct count changes from 4/6 to 6/6 across references. These differences are measured behavior, not uncertainty about which run occurred. The conclusion must name the reference. Large-set retention replaces 52 channels, and size 12 retention replaces 92; failures can reflect severe joint-distribution damage. Agreement across two artificial references helps robustness but cannot certify a natural counterfactual.

## What redundancy and distributed processing would mean

If several groups carry interchangeable information, removing one group may have little effect because another route compensates. If a representation is distributed across channels and relies on combinations, small retention can fail despite relevant information in every retained channel. Downstream gates, normalization, pooling, and nonlinear interactions can also mask or amplify finite changes. These mechanisms are possible explanations of the curves, not demonstrated findings of this experiment.

The weak activation-ranked Kick removal and stronger functional removal show rank dependence directly. The retention curves show that intervention behavior changes with group size. They do not distinguish redundancy from poor ranking, joint-distribution damage, or antagonistic rival evidence. Demonstrating redundancy would require coordinated removal/interaction or substitution tests designed to separate those possibilities.

## Candidate groups versus a complete circuit

A set of late-layer channels is a candidate component list. A complete circuit claim would require a specification of the relevant computation and pathways, careful necessity and sufficiency checks, appropriate counterfactual controls, effects on competing tasks, and generalization beyond these two validation subjects. Minimality would additionally require showing which subsets are dispensable. No edges are traced or mapped here, and 104 channels are not exhaustively searched across combinations. Group membership is not a semantic channel annotation.

The current experiments are useful because they show an exact, reproducible dependence in a fixed deployed model and expose the limits of a simple localization story. They support follow-up questions about broad influence, score competition, and reference sensitivity. They do not justify labeling the top 12 or top 52 sets as the model's Drink, Wave, or Kick circuit.

## How much generalization the evidence supports

There are 108 validation clips but only two validation subjects. Each target's six clips are three repetitions from each subject. Three random permutations are three ways of selecting channels, not three additional people. Neither repetitions nor random groups supply population-level confidence intervals. The baseline itself is 53/54 for s05 and 43/54 for s10, demonstrating how important subject differences are on this partition. No test videos were accessed for this extension.

The cached downstream computation agrees with original feature-cache/head predictions within 5.73e-6 across 108 clips; same-three full-backbone/deployed checks match exactly. Fresh-state replay, full retention, and ranking-independent full removal behave as expected, and protected model hashes/variable digests are unchanged. Those checks support implementation fidelity. They cannot make two subjects representative of an action-recognition population or turn synthetic activations into verified real-world causes.

## Literature context

[Heimersheim and Nanda (2024)](https://arxiv.org/abs/2404.15255) discusses how patching evidence should be interpreted, including circuit and metric pitfalls. [Zhang and Nanda (2023/2024)](https://arxiv.org/abs/2309.16042) studies how corruption and metric choices affect localization in language models. Their cautions motivate reference and metric comparisons here. The present model is a video CNN, and no language-model-specific conclusion is transferred as a proved fact.





## What the exploratory selectivity check adds

Functional retention 52 under zero correctly predicts all 6 Kick clips and never predicts Kick for any of the 102 other validation clips. The mean-reference group gets 4/6 Kick clips, again with no off-target Kick predictions. Wave retention 52 gets 6/6 target clips under both references and zero off-target Wave predictions. Drink gets 4/6 under both references with zero off-target Drink predictions. These results count against a simple broad target-class bias on this cohort: the model distinguishes targets from unrelated clips under the retained representations.

The positive result remains conditional. Under zero, the retained Kick group leaves only 56/102 other actions correctly classified; the Wave group leaves 54/102 and Drink 75/102. Their baseline counts are 93/102,90/102,90/102. They have not preserved a complete multiclass action recognizer. All random groups also have zero target false positives, but none recognizes a Kick clip under either reference. Thus high specificity alone could be obtained by never predicting the target; target sensitivity must be considered with it. Functional 52 retention has evidence of conditional target discrimination, above these fixed random retention controls, while remaining a broad and disruptive subset of the network.

This exploratory amendment answers a material confound exposed by the main results without searching for a favorable subset. It does not make 52 channels minimal, identify their semantic content, show the relevant edges, guarantee transfer to natural activations, or establish population generalization. Both the positive discrimination result and the extensive collateral damage belong in the conclusion.
