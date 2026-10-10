# KARD spatial attribution: three-clip validation pilot

Video Grad-CAM has been implemented for the existing 5-fps full-training KARD model. No model weights were trained or changed. The three fixed validation clips are Drink, Two hand wave and Forward Kick, subject 5 repetition 1. Clips were chosen by action/subject/repetition before attribution, not by visualization quality.

| Ground truth | Prediction | Uncalibrated score | Top-map blur: logit drop | Random blur: mean ± SD |
|---|---|---:|---:|---:|
| Drink | Drink | 93.8% | -0.298 | -0.191 ± 1.602 |
| Two hand wave | Two hand wave | 99.6% | 8.358 | 1.651 ± 1.620 |
| Forward Kick | Forward Kick | 78.1% | 0.035 | 0.292 ± 0.774 |

Highlighted-region blur produces a positive logit drop greater than the random-mask mean in 1/3 pilot clips. In 1/3 clips, highlighted-region blur increases the target logit. The results are mixed and do not establish overall explanation faithfulness.

## What the maps explain

The target is the predicted-class logit, not softmax probability. At the final unpooled b4/l3 endpoint, activations have shape [1,T,6,6,104]. Gradients are averaged over time and space to obtain channel weights. Weighted activation channels are summed and rectified; the resulting T maps are upsampled for display. Each row is normalized using one maximum across the whole clip. Different classes and the randomized head use separate scales, so color brightness must not be compared as an absolute effect size across rows.
The first row is the model input including resize-with-pad. Subsequent rows explain the prediction, its runner-up, and the same target class under one randomly initialized classifier. GIFs show the input alongside the predicted-class overlay at 5 fps. Classifier scores shown in titles are uncalibrated.

## Checks and interventions

- All clips use the saved sampling/resize protocol and fresh streaming state. Source-file hashes match the existing manifest. All maps and gradients are finite, gradients are nonzero, and repeated explanations match exactly.
- The differentiable downstream path reproduces the full backbone-plus-head logits within 1e-4. The first clip also agrees with the deployed SavedModel within 1e-4. The frozen backbone and selected classifier remain bitwise unchanged.
- The random-head panel uses one independently initialized classifier with the same normalization and fixed target class. It is a limited classifier-sensitivity diagnostic, not a complete model/label-randomization sanity test.
- For the blur check, the eight highest cells in the time-averaged 6x6 map are blurred at fixed spatial locations throughout the clip. Twenty seeded random eight-cell masks supply controls. A negative logit decrease means the intervention increased the target logit. SD describes random-mask variability, not confidence intervals. Slight pixel-area variation from unequal grid-edge widths is recorded in the raw JSON.
- Gaussian blur can alter the input distribution, and these controls are only a pilot. Larger target-logit drops for highlighted cells alone do not prove semantic or causal correctness.

## Limits and next use

The spatial resolution is only 6x6. Smooth overlays do not imply pixel-level accuracy. Because MoViNet is causal and temporal, a map at time t reflects activations with earlier-frame context; these panels are not independent temporal importance scores. No person or body-part masks were available to quantify localization. Three clips cannot establish dataset-wide faithfulness, and a heatmap is not evidence of a class-specific subnetwork.
Inspect both successful and mistaken predictions, then expand the fixed evaluation sample across classes and subjects with localization annotations, additional randomization controls, and temporal interventions. These explanations have not yet been integrated into the webcam viewer; gradient explanations can be more expensive than ordinary inference.

`analysis.json`, per-clip JSON and raw NPZ arrays retain exact indices, maps, logits and interventions. `paper_assets/` contains PDF/SVG/PNG contact sheets, animated GIFs, an intervention figure and LaTeX snippets.

Method reference: [Grad-CAM](https://arxiv.org/abs/1610.02391). Explanation validation reference: [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).