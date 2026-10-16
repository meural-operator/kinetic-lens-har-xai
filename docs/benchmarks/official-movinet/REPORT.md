# Official Google MoViNet evaluation

For a live-camera HAR dashboard on a **Windows/Linux CPU or mini-PC, start with A0-Stream**: its measured CPU frame latency is substantially lower than A2's. **A2-Stream** is the higher-capacity candidate when prioritizing its stronger published Kinetics result and when the compute budget permits it; GPU deployment needs an appropriate backend. Both streaming variants can process video clips as well as individual frames. **A2-Base** is a useful offline clip baseline.

The recommendation is an engineering inference from Google's model design and published benchmarks, plus local functional tests. It is not a finding that A2 is most accurate on the user's five HAR datasets.

## Official research findings

MoViNets are mobile-oriented video action classifiers. The streaming architecture uses causal temporal convolutions, retained feature buffers and pooling statistics. Its state is not the BiLSTM used in the existing HAR models. Base variants use noncausal 3D convolutions and process complete clips. Google recommends streaming variants for fast CPU/mobile inference because their (2+1)D implementation uses optimized 2D convolutions.

| Family | Published Kinetics-600 base top-1 (%) | Published streaming top-1 (%) | Evaluation input T×H×W | GFLOPs/video |
|---|---:|---:|---|---:|
| A0 | 72.28 | 72.05 | 50×172×172 | 2.7 |
| A1 | 76.69 | 76.45 | 50×172×172 | 6.0 |
| A2 | 78.62 | 78.40 | 50×224×224 | 10 |
| A3 | 81.79 | 80.09 | 120×256×256 | 57 |
| A4 | 83.48 | 81.49 | 80×290×290 | 110 |
| A5 | 84.27 | 82.37 | 120×320×320 | 280 |

These are Google's single-clip Kinetics-600 benchmark figures from the [official implementation](https://github.com/tensorflow/models/blob/master/official/projects/movinet/README.md#results-and-pretrained-weights), not measurements from this workstation. The [MoViNets paper](https://arxiv.org/abs/2103.11511) describes the architecture and stream buffers. Google's [inference tutorial](https://www.tensorflow.org/hub/tutorials/movinet) demonstrates stateful inference and supplies the reference jumping-jacks clip.

For A2, the published streaming/base accuracy difference is only 0.22 percentage points. A3–A5 provide higher published recognition accuracy at substantially higher computation. A0 trades recognition accuracy for a smaller input resolution and lower computation. Actual ranking on KTH, UAV, KARD, HON4D and UCF45 requires evaluation or fine-tuning on their data.

## Downloads and actual execution

Downloaded Google's TF Hub classification/3 SavedModels for A0-Stream, A2-Stream and A2-Base. TF Hub resolves to Google's public Kaggle-hosted bundles. Each includes saved_model.pb and its variables checkpoint. Requested handles, resolved unsigned archive locations, sizes and SHA-256 hashes are in downloads.json. Learned variables were loaded by tf.saved_model.load; no model architecture was randomly initialized in place of the public weights. No third-party PyTorch port or MindGraph model implementation was used.

Installed TensorFlow 2.21.0 and dependencies into the local tf_deps folder and executed through the Miniconda xai interpreter. Existing xai site-packages were not modified. Runtime used the Xeon w7-2595X CPU, four intra-op threads and one inter-op thread, with default oneDNN enabled. GPU enumeration returned an empty list. Current official TensorFlow does not provide native Windows CUDA support; GPU testing would require WSL/Linux or a separately validated conversion. See the [official installation documentation](https://www.tensorflow.org/install/pip#windows-native).

| Model | Resolution | Variable file MB | Complete SavedModel MB | Retained-state frame latency (ms) | Eight-frame clip latency (ms) |
|---|---:|---:|---:|---:|---:|
| A0-Stream | 172×172 | 15.32 | 44.17 | 31.69 ± 6.21 | 75.85 ± 8.51 |
| A2-Stream | 224×224 | 29.90 | 80.90 | 100.63 ± 22.89 | 247.51 ± 21.67 |
| A2-Base | 224×224 | 21.46 | 35.36 | N/A | 7021.61 ± 1456.17 |

MB means 1,000,000 bytes. Variable-file size is the actual downloaded serialized checkpoint data, including auxiliary stored variables; it is not a count of trainable parameters. The complete SavedModel additionally contains graph/function metadata. Compressed archive sizes are smaller and must not be confused with loaded model size.

Latency values are mean ± sample standard deviation over 30 calls after five warm-up calls. Frame latency is actual rolling inference with retained state; eight-frame clip latency starts from fresh zero state. Both use the official jumping-jacks input, evaluation FP32, CPU execution, and synchronized NumPy output materialization. State initialization/reset, loading, decoding, preprocessing and top-five generation are outside timing. Other timing modes, p95, extrema and raw observations are in results.json and timings.csv. These are single-workstation fixed-input measurements under normal OS background activity. They are not directly comparable with the earlier FP16 ONNX GPU table or proof of performance on a mini-PC.

For the retained-state benchmark, the 13 reference frames cycle across 30 timed calls after five warm-up calls. This tests rolling state handling and execution cost; it is not a new action-recognition evaluation on a naturally continuous 35-frame clip.

## Recognition results on the supplied demonstrations

| Model | Demo clip | Sampled frames | Predicted Kinetics-600 class | Softmax probability |
|---|---|---:|---|---:|
| A0-Stream | jumping-jacks | 13 | jumping jacks | 99.03% |
| A0-Stream | kth-boxing | 30 | juggling balls | 21.38% |
| A0-Stream | kth-running | 30 | jogging | 73.55% |
| A0-Stream | kard-drink | 28 | front raises | 60.21% |
| A0-Stream | kard-kick | 17 | stretching leg | 59.56% |
| A2-Stream | jumping-jacks | 13 | jumping jacks | 99.94% |
| A2-Stream | kth-boxing | 30 | tai chi | 35.01% |
| A2-Stream | kth-running | 30 | jogging | 93.20% |
| A2-Stream | kard-drink | 28 | front raises | 54.00% |
| A2-Stream | kard-kick | 17 | stretching leg | 55.43% |
| A2-Base | jumping-jacks | 13 | jumping jacks | 83.35% |
| A2-Base | kth-boxing | 30 | tai chi | 56.69% |
| A2-Base | kth-running | 30 | jogging | 57.51% |
| A2-Base | kard-drink | 28 | front raises | 75.63% |
| A2-Base | kard-kick | 17 | stretching leg | 59.84% |

The supplied KTH/KARD clips and tutorial GIF are demonstration inputs with unknown training membership. Kinetics-600 labels do not match the original dataset-specific classifier vocabularies. Softmax values are not calibrated confidence or a reliability guarantee. Incorrect labels or weak probabilities on these videos indicate domain/task mismatch; producing a result does not establish target-dataset accuracy.

## Streaming checks

For each streaming model and each of the five videos, compared one-frame updates, eight-frame chunks and processing the full sampled video with fresh state. Checked finite 600-class logits, finite state tensors, the head's consumed-frame count and repeatability after state reset. State resets also separate different video files. Numerical closeness uses atol=1e-3, rtol=1e-3 and is explicitly reported rather than assumed.

| Model | Video | Framewise vs full max logit error | 8-frame chunks vs full max error | Numerically close | Reset repeat error |
|---|---|---:|---:|---|---:|
| A0-Stream | jumping-jacks | 2.38419e-06 | 2.5034e-06 | True | 0 |
| A0-Stream | kth-boxing | 2.86102e-06 | 1.90735e-06 | True | 0 |
| A0-Stream | kth-running | 2.86102e-06 | 3.8147e-06 | True | 0 |
| A0-Stream | kard-drink | 2.38419e-06 | 4.05312e-06 | True | 0 |
| A0-Stream | kard-kick | 2.86102e-06 | 2.86102e-06 | True | 0 |
| A2-Stream | jumping-jacks | 1.90735e-06 | 1.43051e-06 | True | 0 |
| A2-Stream | kth-boxing | 1.43051e-06 | 1.90735e-06 | True | 0 |
| A2-Stream | kth-running | 3.8147e-06 | 1.90735e-06 | True | 0 |
| A2-Stream | kard-drink | 1.43051e-06 | 1.90735e-06 | True | 0 |
| A2-Stream | kard-kick | 1.19209e-06 | 1.43051e-06 | True | 0 |

Streaming state should be maintained separately for each camera/video and reset when the source changes or the analysis window restarts. Cumulative pooling can retain earlier activity, so an application needs an explicit window/reset policy for changes of action. This model classifies an entire video scene; per-person HAR in multi-person footage additionally needs person tracking/crops or a suitable recognition design.

## Reuse with the HAR dashboard

MoViNet can replace the classification backend behind the existing camera/upload UI after adding its adapter. It does not reuse the existing ViT-BiLSTM weights. The input contract is RGB [B,T,H,W,C], float32 [0,1], resolution 172 for A0 or 224 for A2, with approximately 5 fps as the A2 reference sampling. The existing HAR pipeline's [B,T,C,H,W] layout and [-1,1] normalization must not be passed unchanged.

The downloaded head predicts 600 Kinetics classes. For the specific 6/13/18/12/45 class tasks, use the pretrained backbone, replace/train the classifier head, and generally fine-tune/evaluate on the target dataset. No fine-tuning was performed here. An overlapping label map can support a limited demo but is not a substitute for that study. These downloads have not been integrated into the dashboard and do not establish privacy protection by themselves.

## Reproduce

```powershell
& 'python' -B '<local-workspace>\movinet_evaluation\test_models.py'
& 'python' -B '<local-workspace>\movinet_evaluation\run_video.py' '<local-workspace>\movinet_evaluation\jumpingjack.gif' --model a2_stream --output '<local-workspace>\movinet_evaluation\single_video_result.json'
```

run_video.py reads local video files and produces per-chunk top-five predictions. It performs no uploads. Select --model a0_stream for the smaller variant, a2_base for offline clips, or --chunk-frames 8 to process streaming input in larger chunks. Its processing_ms includes first-run warm-up and prediction formatting; use the benchmark statistics for latency claims.

## Follow-up person-crop experiment

The user proposed person localization before recognition. A subsequent controlled detector/crop test is documented in [CROP_REPORT.md](../../research/person-crop/REPORT.md). It improves the boxing-like prediction in an aspect-preserving condition, but does not consistently resolve the other task/label mismatches.
