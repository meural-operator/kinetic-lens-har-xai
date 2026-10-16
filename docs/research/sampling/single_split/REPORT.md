# KARD: controlled 5-fps versus 10-fps frozen-head experiment

The new 10-fps head is trained from initialization on the same 324 training clips as the 5-fps baseline. The official A0-Stream backbone remains frozen. Target rate and frame cap change together (5/50 to 10/100), preserving a centered 10-second time window; no cropping, augmentation, split change or backbone fine-tuning is introduced.

| Sampling | Frame cap | Training accuracy | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| 5 fps | 50 | 100.00% | 88.89% | 88.77% | 87.04% | 86.80% |
| 10 fps | 100 | 100.00% | 89.81% | 89.40% | 84.26% | 84.15% |

**Validation recommends 10 fps.** This decision was frozen in `sampling_selection.json` before extracting or evaluating new 10-fps test features. Validation macro F1 changes by +0.64 percentage points. The descriptive test accuracy change is -2.78 points, correcting 7 earlier mistakes and introducing 10 errors on previously correct clips.

## Fixed controls and checks

- Same training subjects 1,3,4,6,7,8; validation 5,10; test 2,9. Same source files, labels, ordering and selected clips: 324/108/108.
- Same RGB [0,1], aspect-preserving padding to 172x172, original frozen official weights, 480-dimensional pooled endpoint and state reset per clip. Eight-frame streaming chunks retain state within each clip.
- New head initialized afresh with seed 42; training-only normalization refitted using the 10-fps training features. Dropout 0.25, Dense(18), 8,658 trainable parameters.
- Same Adam learning rate 0.001, batch 16, maximum 200 epochs, patience 25, L2 candidates 0.0001/0.001/0.01. Checkpoint/candidate selection by validation macro F1, then lower cross entropy.
- No final 5-fps features are reused as 10-fps features. All 540 new pooled feature vectors are extracted and checked; backbone content digests remain unchanged.
- Higher-rate sampling exposed terminal timestamp rounding: 101 clips have one requested timestamp whose rounded frame index would exceed the final real frame. That unavailable terminal target is omitted without padding or repeated frames. The full 540-clip timestamp audit proves all baseline 5-fps indices remain unchanged and are exactly every other selected 10-fps index; window starts match for every clip. The earlier full decoding audit is reused after source hashes are verified.
- The inference CLI reads each experiment protocol; a 10-fps model is not accidentally given the default 5-fps preprocessing.
- Native Windows TensorFlow CPU in the Miniconda xai environment.

| Candidate for 10 fps | Selected epoch | Validation accuracy | Validation macro F1 |
|---:|---:|---:|---:|
| L2 0.0001 | 20 | 89.81% | 89.40% |
| L2 0.001 | 20 | 89.81% | 89.40% |
| L2 0.01 | 20 | 89.81% | 89.39% |

| Action | Validation recall: 5 fps | Validation recall: 10 fps | Test recall: 5 fps | Test recall: 10 fps |
|---|---:|---:|---:|---:|
| Horizontal arm wave | 100.0% | 100.0% | 50.0% | 66.7% |
| High arm wave | 66.7% | 66.7% | 100.0% | 83.3% |
| Two hand wave | 100.0% | 100.0% | 100.0% | 100.0% |
| Catch Cap | 100.0% | 100.0% | 100.0% | 100.0% |
| High throw | 66.7% | 50.0% | 50.0% | 83.3% |
| Draw X | 66.7% | 100.0% | 83.3% | 100.0% |
| Draw Tick | 100.0% | 100.0% | 33.3% | 33.3% |
| Toss Paper | 100.0% | 100.0% | 100.0% | 100.0% |
| Forward Kick | 50.0% | 100.0% | 100.0% | 100.0% |
| Side Kick | 100.0% | 100.0% | 100.0% | 100.0% |
| Take Umbrella | 100.0% | 100.0% | 100.0% | 83.3% |
| Bend | 100.0% | 100.0% | 83.3% | 100.0% |
| Hand Clap | 66.7% | 50.0% | 83.3% | 66.7% |
| Walk | 100.0% | 100.0% | 100.0% | 100.0% |
| Phone Call | 100.0% | 100.0% | 100.0% | 100.0% |
| Drink | 100.0% | 66.7% | 100.0% | 83.3% |
| Sit down | 100.0% | 100.0% | 83.3% | 66.7% |
| Stand up | 83.3% | 83.3% | 100.0% | 50.0% |

## Interpretation and artifacts

This controlled experiment tests whether denser temporal input improves this frozen-backbone adaptation baseline. Changing fps also changes the physical duration spanned by a fixed number of temporal convolution steps; a performance change cannot prove that missing motion was the sole cause of earlier errors. One seed and one researcher-defined subject split, with two validation and two test subjects, are reported. Earlier test results were already observed. No statistical significance, cross-validation, population-level generalization, privacy guarantee or new algorithm is claimed. No post-test tuning was performed.

`protocol.json`, `sampling_protocol_verification.json`, `sampling_reader_verification.json`, feature metadata, candidate histories, `selection.json`, `sampling_selection.json`, `evaluation.json` and paired predictions retain the evidence. `paper_assets/` provides vector PDF/SVG, 600-dpi PNG, figure/table LaTeX snippets and exact plot data. Previous experiments and paper assets are preserved.

Use the complete `saved_model/` plus `protocol.json` and `labels.json` for 10-fps inference. Reproduce/resume declared stages with `movinet_evaluation/run_kard_10fps_pipeline.ps1`.

```powershell
& 'python' -B movinet_evaluation\predict_kard_video.py --experiment-dir movinet_evaluation\kard_a0_head_100pct_10fps --video 'C:\path\to\clip.mp4'
```