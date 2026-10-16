# Figure captions

## fig01_training_fraction

Test performance of frozen MoViNet A0-Stream classifiers trained on nested KARD subsets of 81, 162 and 324 clips (25%, 50% and 100% of the 324-clip training pool). Each head is initialized afresh. All runs use the same 108 test clips from subjects 2 and 9; checkpoints and L2 strength are selected using subjects 5 and 10 only. One split and seed are reported; no uncertainty intervals or significance claims are made. Earlier test results were already observed.

## fig02_protocol

Experimental protocol and inference pipeline. Training uses six KARD subjects and nested 81-, 162- and 324-clip subsets; validation and test each use two distinct subjects. RGB clips are sampled at 5 fps and resized with padding to 172 × 172; clips longer than 10 seconds use a centered 10-second window, with at most 50 frames. Fresh MoViNet states are used per clip, retaining state across eight-frame extraction chunks. The official 480-dimensional A0 pooled endpoint is frozen. Training-only standardization precedes dropout (0.25) and an 18-class dense head. No privacy-preservation mechanism is evaluated in these experiments.

## fig03_action_recall

Per-action test recall across the three nested training subsets. Each cell gives the exact number of correctly classified test clips out of six (three repetitions from each of two held-out subjects); shading shows recall from 0% to 100%. More training data improves several actions but does not improve every action monotonically. These cells are small-sample descriptive results, not class-specific estimates with independent uncertainty bounds.

## fig04_confusion_100pct

Test confusion matrix for the head trained on all 324 training clips. Rows are true actions, columns are predictions, and numbers are clip counts (zero entries are omitted). Each row contains six clips; 94 of 108 predictions are correct. Plot action IDs are one-based for readability; implementation class IDs are zero-based. The action key gives the exact manifest label order.

## fig05_training_curves

Unsmoothed training and validation macro F1 trajectories for the validation-selected L2 candidate in each run. Dotted gray lines and orange markers indicate selected checkpoints: epoch 60 (25%, L2 = 0.0001), epoch 63 (50%, L2 = 0.01), and epoch 19 (100%, L2 = 0.01). Panel headings report the selected epoch, not the number of executed epochs. Curves include the subsequent early-stopping patience period. Test results are not used for checkpoint selection.

## fig06_subject_accuracy

Test accuracy separately for subjects 2 and 9, each contributing 54 clips. Lines connect the three measured training fractions for visual guidance only. These two subjects do not establish population-level generalization; no subject-level confidence intervals or statistical significance are claimed.
