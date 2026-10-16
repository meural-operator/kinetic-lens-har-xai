# HAR model size and inference latency

This report reconciles existing results; it does not run a new benchmark.

**Coverage:** four base architectures; nine checkpoint/classifier configurations; three precision levels (FP32, FP16, mixed dynamic INT8); 31 measured model/runtime/precision forms; 26 saved inference artifacts plus five PyTorch FP16 in-memory forms.

## Five ViT–BiLSTM checkpoints: 8 RGB frames at 224×224

| Checkpoint | Runtime | Precision | File MB | CPU ms | GPU ms |
|---|---|---|---:|---:|---:|
| KTH | PyTorch | FP32 | 357.74 | 876.17 ± 41.08 | 33.10 ± 2.52 |
| KTH | ONNX | FP32 | 355.43 | 699.14 ± 35.23 | 32.23 ± 1.76 |
| KTH | PyTorch | FP16 | — (in memory) | — | 23.89 ± 10.91 |
| KTH | ONNX | FP16 | 177.83 | — | 14.08 ± 2.63 |
| KTH | ONNX | Mixed INT8 | 92.18 | 216.66 ± 14.85 | — |
| UAV-Actions | PyTorch | FP32 | 357.74 | 892.58 ± 41.09 | 32.93 ± 1.41 |
| UAV-Actions | ONNX | FP32 | 355.43 | 687.27 ± 37.06 | 33.36 ± 1.73 |
| UAV-Actions | PyTorch | FP16 | — (in memory) | — | 19.08 ± 8.80 |
| UAV-Actions | ONNX | FP16 | 177.83 | — | 13.98 ± 3.56 |
| UAV-Actions | ONNX | Mixed INT8 | 92.18 | 205.86 ± 12.70 | — |
| KARD | PyTorch | FP32 | 357.74 | 858.29 ± 39.39 | 33.50 ± 1.48 |
| KARD | ONNX | FP32 | 355.43 | 669.12 ± 26.14 | 32.61 ± 1.43 |
| KARD | PyTorch | FP16 | — (in memory) | — | 18.18 ± 7.68 |
| KARD | ONNX | FP16 | 177.84 | — | 13.68 ± 3.12 |
| KARD | ONNX | Mixed INT8 | 92.18 | 206.48 ± 11.31 | — |
| HON4D | PyTorch | FP32 | 357.74 | 907.31 ± 34.83 | 33.73 ± 1.62 |
| HON4D | ONNX | FP32 | 355.43 | 711.69 ± 28.76 | 32.81 ± 1.38 |
| HON4D | PyTorch | FP16 | — (in memory) | — | 18.82 ± 7.72 |
| HON4D | ONNX | FP16 | 177.83 | — | 13.78 ± 2.67 |
| HON4D | ONNX | Mixed INT8 | 92.18 | 197.18 ± 10.16 | — |
| UCF45-DS | PyTorch | FP32 | 357.76 | 895.78 ± 30.68 | 33.44 ± 0.89 |
| UCF45-DS | ONNX | FP32 | 355.45 | 707.81 ± 20.49 | 33.74 ± 1.42 |
| UCF45-DS | PyTorch | FP16 | — (in memory) | — | 12.77 ± 2.41 |
| UCF45-DS | ONNX | FP16 | 177.84 | — | 12.47 ± 0.79 |
| UCF45-DS | ONNX | Mixed INT8 | 92.18 | 201.60 ± 16.42 | — |

Mean ± sample SD over 45 calls per backend. CPU and GPU are separate sessions. GPU inputs/outputs were resident; transfer costs excluded. No measured CPU FP16 or GPU INT8 here.

## KARD MoViNet-A0: 20 RGB frames at 172×172

| Runtime | Precision | Device | File MB | Mean ms | Median ms | P95 ms |
|---|---|---|---:|---:|---:|---:|
| TensorFlow | FP32 | CPU | 26.363 | 121.30 | 119.61 | 139.36 |
| ONNX | FP32 | GPU | 8.430 | 34.61 | 31.74 | 67.58 |
| ONNX | FP16 | GPU | 4.997 | 28.73 | 26.69 | 41.71 |

90 calls per backend, 18 validation windows × five repeats. GPU timings include transfers and synchronous CPU output, unlike the ViT block. Only aggregate statistics were saved; SD is unavailable. The TensorFlow size is its full SavedModel directory; ONNX size is the graph file. FP16 keeps FP32 I/O.

## Official MoViNet scales: separate CPU FP32 experiment

| Model | Resolution | Full package MB | Weight-data MB | 8-frame clip ms | Retained-state single-frame ms |
|---|---|---:|---:|---:|---:|
| A0-Stream | 172×172 | 44.17 | 15.32 | 75.85 ± 8.51 | 31.69 ± 6.21 |
| A2-Stream | 224×224 | 80.90 | 29.90 | 247.51 ± 21.67 | 100.63 ± 22.89 |
| A2-Base | 224×224 | 35.36 | 21.46 | 7021.61 ± 1456.17 | — |

30 calls after five warm-ups on the tutorial reference input. Single-frame streaming and fresh-state clip timings are different workloads. These models predict 600 Kinetics classes; KARD predicts 18.

## Counting rules and missing evidence

- Five ViT checkpoints × five tested forms = 25: PyTorch FP32/FP16 and ONNX FP32/FP16/mixed INT8.
- Three stock MoViNet FP32 configurations = 3.
- KARD MoViNet TensorFlow FP32 / ONNX FP32 / ONNX FP16 = 3.
- Total 31 forms; only 26 serialized artifacts because PyTorch FP16 was in memory.
- A1/A3/A4/A5 were not locally executed. No recorded TFLite, INT4, GPU INT8, static INT8/QAT, TensorRT or MoViNet INT8 benchmark.
- B/C/D head ablations and LoRA/guided/random adapters have training results, not complete deployed size–latency results. Seeds, training fractions and sampling rates are not new quantization sizes.

## Conversion agreement

| ViT checkpoint | ONNX FP16 vs original | Original mixed INT8 export vs original |
|---|---:|---:|
| KTH | 9/9 | 9/9 |
| UAV-Actions | 9/9 | 8/9 |
| KARD | 9/9 | 8/9 |
| HON4D | 9/9 | 8/9 |
| UCF45-DS | 9/9 | 9/9 |

Agreement on supplied demonstration windows is not held-out task accuracy. KARD MoViNet ONNX FP32 and FP16 each agree on 18/18 selected validation windows. The historical KARD head used in these latency tests is not the later strong baseline or adapter checkpoint.

## Reuse in an article

- `article_tables.tex`: three self-contained table* environments; requires booktabs. Copy this snippet into the manuscript folder and include it with `\input{article_tables.tex}`.
- `model_size_latency.csv`: exact byte counts, full-precision timing values, compression and speedups.
- `size_latency_audit.json`: source/artifact SHA256, available historical-hash checks and counting rules.
- `size_latency_tradeoff.pdf`, `.svg`, `.png`: vector/raster plots; distinct protocol panels.
- Existing open `mindgraph_audit/latency_benchmark/latency_table.tex` updated in place; the preceding version is preserved as `latency_table_before_size_latency.tex`.

Hardware: Xeon w7-2595X, 4 intra-/1 inter-op threads; RTX 4000 Ada 20 GB; Windows 11; PyTorch 2.13.0+cu130, ONNX Runtime GPU 1.31.0, TensorFlow 2.21.0. Size is decimal MB of serialized files, not runtime memory. Different sessions, frame counts, resolutions, class heads, graph formats and transfer policies prevent a controlled cross-family ranking. All timings exclude capture/display, decoding, loading and XAI.


Temporal sampling follow-up: `table_temporal_sampling.tex`, `temporal_sampling.csv` and `temporal_sampling.md` report verified 5/10-fps recognition results and distinguish them from the missing matched frame-count latency sweep.
