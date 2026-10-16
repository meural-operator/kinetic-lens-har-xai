# Person-crop ablation for MoViNet

The user's hypothesis is partly supported: locating the actor changes some predictions toward the intended action. It did not consistently resolve the dataset mismatch in these four demonstration clips. This is a small controlled diagnostic, not evidence of dataset-wide accuracy improvement.

Used the official [Torchvision SSDLite320 MobileNetV3 Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) COCO weights (3,440,060 parameters). The detector ran in FP32 on CUDA; the official MoViNet classifiers ran on CPU in TensorFlow. The detector checkpoint source and SHA-256 are in person_rois.json.

Each video contains one intended actor. On exactly the previously sampled frames, selected the highest-scoring person detection above 0.25, took the union of these boxes, added 15% padding per side, and clamped to the image. This creates one stable crop for the whole clip, avoiding crop jitter in this ablation. For a live camera it uses future information and must be replaced by causal detection/tracking/smoothing. It is not an implemented live multi-person tracker. Missing detections remain in the input sequence; no frames were discarded to make recognition easier.

Kept weights, frame indices, frame count, state initialization, RGB range [0,1] and inference protocol identical. Compared full-frame resize, padded person-crop resize, and an aspect-preserving version with black letterboxing. The two crop preprocessors are separate conditions because directly resizing a tall person ROI to a square distorts body geometry. Each MoViNet state was reset before each condition. Full-clip execution was previously verified equivalent to framewise streaming.

| Model | Video | Full frame | Person crop, resized | Person crop, aspect-preserving letterbox |
|---|---|---|---|---|
| a0_stream | kth-boxing | juggling balls | using a paint roller | punching person (boxing) |
| a0_stream | kth-running | jogging | jogging | jogging |
| a0_stream | kard-drink | front raises | front raises | front raises |
| a0_stream | kard-kick | stretching leg | stretching leg | stretching leg |
| a2_stream | kth-boxing | tai chi | tai chi | punching person (boxing) |
| a2_stream | kth-running | jogging | jogging | jogging |
| a2_stream | kard-drink | front raises | drinking shots | front raises |
| a2_stream | kard-kick | stretching leg | stretching leg | stretching leg |

Boxing: both A0-Stream and A2-Stream produce "punching person (boxing)" with aspect-preserving person crops. This is closer to the demonstrated boxing motion, but KTH shadow boxing is not the same Kinetics class and no second person is present. It is not an exact-label accuracy result.

Drinking: the plain A2 crop changes "front raises" to "drinking shots", whereas the aspect-preserving crop remains "front raises". This result depends on crop preprocessing and does not establish reliable generic drinking recognition. KARD's drinking task differs from drinking shots or sipping a cup.

Kicking: "stretching leg" remains the top class for both crop variants; "side kick" appears in lower-ranked predictions. Cropping does not resolve this example. Running remains "jogging"; these Kinetics weights do not provide KTH's separate generic running and jogging output classes.

The Kinetics head offers "punching bag", "punching person (boxing)", "drinking shots", "sipping cup", "high kick", "side kick", "jogging" and "running on treadmill". It does not implement the same closed-set classifier as the original 6/13/18/12/45-class HAR models. Correct head/label mapping and target-domain training/evaluation are therefore separate from person localization. Google's [MoViNet transfer-learning tutorial](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet) replaces the pretrained classifier head with one for the new dataset.

For deployment, use person detection and identity tracking, retain enough surrounding context for hands/feet/interacted objects, generate temporally stable crops, and keep/reset MoViNet state separately for each track/window. Fine-tune and evaluate on the same crop policy used at inference. A tight object-only crop of a bottle or cup removes body/action cues. Extra context or a whole-frame stream may be useful for interactions, subject to validation.

The detector plus MoViNet needs a separate end-to-end latency measurement; parameter count alone does not guarantee a fast pipeline. This experiment measured CUDA detector calls in Torchvision's eager implementation, not an optimized edge runtime, and did not integrate the detector into the dashboard.

## Files and reproduction

- person_rois.json: per-frame detections, crop coordinates, detector metadata and timing.
- crop_results.json: all 24 condition predictions, including top-five probabilities.
- crop_comparison.csv: compact condition table.
- detect_person_rois.py: official lightweight detector and stable offline ROIs.
- test_person_crops.py: actual full-frame/crop inference comparisons.

```powershell
& 'python' -B '<local-workspace>\movinet_evaluation\detect_person_rois.py'
& 'python' -B '<local-workspace>\movinet_evaluation\test_person_crops.py'
```
