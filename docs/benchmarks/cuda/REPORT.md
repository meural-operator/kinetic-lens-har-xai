# MoViNet KARD: official ONNX Runtime CUDA trial

Date: 2026-10-10. Host: native Windows, Intel Xeon w7-2595X, NVIDIA RTX 4000 Ada Generation. Miniconda environment: xai. TensorFlow 2.21.0, ONNX Runtime GPU 1.31.0, tf2onnx 1.17.0, ONNX 1.23.2.

The original MoViNet-A0 plus trained KARD-18 head was exported without retraining. The selected source remains `kard_a0_head_100pct/saved_model`. This trial uses official ONNX Runtime, not Mindgraph inference code. Earlier isolated dependency folders retain their audit names; this does not change package provenance.

## Measured latency

| Backend | Model mean / median / p95 (ms) | App processing mean / median / p95 (ms) |
|---|---:|---:|
| TensorFlow CPU FP32 | 121.30 / 119.61 / 139.36 | 283.23 / 287.85 / 314.77 |
| ONNX CUDA FP32 | 34.61 / 31.74 / 67.58 | 94.66 / 91.90 / 121.27 |
| ONNX CUDA FP16 | 28.73 / 26.69 / 41.71 | 103.33 / 103.17 / 126.28 |

Model invocation: 18 real validation windows, one per KARD class, five repeats (90 calls per backend), three warm-up calls per backend. Samples and backend order were randomized with seed 42 in each repeat. All backends consume identical preprocessed float32 CPU inputs. ONNX timing includes transfer to the GPU and synchronous CPU output. It excludes decode, resize and display.

App processing: one held-out KARD Walk video (`a14_s09_e03.mp4`, subject 9, test split), seven rolling windows repeated three times (21 calls per backend), measured sequentially. This established timing clip was used only for throughput and implementation checks, not for checkpoint or provider selection by recognition accuracy. Includes BGR-to-RGB, stacking, original TensorFlow bilinear resize-with-padding, normalization, model invocation, softmax and top-three formatting. Excludes video decoding, camera capture, rendering, model loading and warm-up. These are separate workloads: do not subtract their numbers to estimate a precise preprocessing cost.

The input is [1,20,172,172,3], sampled at 5 fps: a four-second context window, with fresh MoViNet states for each invocation. Computation latency does not remove that temporal context requirement. The app continues to update once per second by default. Warm latency varies with host load; these single-host runs are not confidence intervals or independent accuracy trials.

## Numerical agreement and placement

- CUDA FP32: 18/18 top-one predictions agree with TensorFlow; maximum absolute logit error 1.7881393e-05; probability error 8.3446503e-07.
- CUDA FP16: 18/18 agree; maximum absolute logit error 0.057655811; probability error 0.006631434 (0.663 percentage points).
- Independent operator profiling observed 999 CUDA node events for FP32 and 1001 for FP16; zero CPU node events. The CPU provider can appear registered in the provider list, but CPU execution fallback is disabled and actual placement is checked.
- Both GPU app paths and the CPU path pass preprocessing agreement, deterministic state-reset repeatability, finite-output and bounded-mailbox checks.

Agreement measures preservation of the original predictions, not their correctness. The original also misclassified the sampled High throw window. Videos too short for 20 distinct samples were skipped, recorded in `benchmark.json`, and never padded. This coverage check is not full validation-set accuracy, blind test accuracy or webcam accuracy. No camera was opened during this CUDA trial.

## FP16 implementation and tuning

FP32 artifact: 8.430 MB. FP16 artifact: 4.997 MB (40.7% smaller). Inputs and outputs remain float32; learned weights and eligible internal tensors use FP16, with FP32 accumulation in some CUDA kernels. This is not INT8 quantization.

The first converter configuration left Min/Max/CumSum in FP32 and generated 92 executed Cast nodes. CUDA support was verified for those operations in FP16, reducing the profile to two input/output casts. Only provably equivalent duplicate Cast producers and unreferenced captured inputs were removed; ONNX full checking passed. Artifacts retain exact source and output SHA-256 checksums.

The final FP16 CUDA provider requests `use_tf32=1` to allow Tensor Core plans; this does not change FP16 tensor types to TF32. FP32 requests `use_tf32=0`. ONNX Runtime's cuDNN frontend can retry an unsupported convolution plan internally with different CUDA math options, so the FP32 provider request is not a proof that every kernel used strict FP32 arithmetic. Numerical comparison and actual CUDA placement provide the empirical checks. No Tensor Core hardware trace was collected; enabling eligible plans is not proof that each kernel uses Tensor Cores. `cudnn_conv_algo_search=HEURISTIC`, four intra-op threads, one inter-op thread, and CPU execution fallback disabled.

Initial FP16 results and the intermediate configuration are preserved in `benchmark_initial.json`, `benchmark_optimized_no_tensor_cores.json` and `live_pipeline_fp16_no_tensor_cores.json`. They are separate runs and must not be pooled to manufacture an uncertainty estimate. Some cuDNN frontend plan warnings appeared at first execution; the runtime recovered through another CUDA plan and completed profiling successfully.

## Public repository entry points

See [CUDA deployment](../../cuda.md) for installation, portable desktop launchers, browser configurations and reproduction commands. These numbers were measured with the original desktop SavedModel wrapper. The published adapter uses the same checked ONNX artifacts, CUDA provider settings and TensorFlow pixel preprocessing; its migration is checked separately.

Archived JSON paths use workspace placeholders. Data, recordings, runtime wheels and raw subject imagery are not redistributed. The original FP16 conversion attempt and intermediate configuration remain separate historical records. Only the final checked exports are distributed.

Paper assets: latency.csv and latency_tables.tex (booktabs include snippet).

Official references: [FP16 conversion](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html), [CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [cuDNN frontend](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/nn/conv.cc).
