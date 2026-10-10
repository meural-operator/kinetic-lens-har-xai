# MoViNet KARD on CUDA: FP16 and FP32

The checked ONNX exports include the official pretrained MoViNet-A0 backbone and our selected trained KARD-18 head. They are inference conversions; no retraining was performed. Both artifacts and their SHA-256 provenance are versioned in `models/kard/onnx/`.

## Install and launch

Use an environment with a compatible NVIDIA driver and CUDA/cuDNN runtime. Run from the repository root:

```sh
python -m pip install -e ".[cuda]"
```

Install one ONNX Runtime distribution per environment. The CUDA extra installs `onnxruntime-gpu`; do not also install the CPU `onnxruntime` wheel. Compatible PyTorch CUDA installations can supply runtime libraries. Alternatively, ONNX Runtime documents installing its CUDA/cuDNN extras:

```sh
python -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.31,<2"
```

The tested host used official ONNX Runtime GPU 1.31.0 and libraries supplied by its existing PyTorch CUDA installation. New environments must match CUDA/cuDNN major versions; consult the [official compatibility and DLL-loading guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). Dependencies and hardware drivers are not bundled.

**Original desktop camera window**, with predictions overlaid on the same feed:

```sh
python scripts/desktop/realtime_kard.py --config configs/movinet_kard_cuda_fp16.json
python scripts/desktop/realtime_kard.py --config configs/movinet_kard_cuda_fp32.json
```

Windows launchers are in `scripts/desktop/`: `launch_live_har_gpu.bat` (FP16), `launch_live_har_gpu_fp32.bat` (FP32) and `launch_live_har_cpu.bat` (the TensorFlow model). They use the Miniconda environment named `xai`; run from a terminal where `conda` is available after installing the required dependencies there. For the CPU config, first run `python scripts/setup_models.py` to obtain the original backbone. GPU configs already include their complete ONNX graphs and do not require that download. Q, Escape or closing the desktop window releases the camera.

**Browser dashboard**, using the same CUDA model adapter:

```sh
kinetic-lens --config configs/movinet_kard_cuda_fp16.json serve
```

Open http://127.0.0.1:8765 and select Start camera. Use `movinet_kard_cuda_fp32.json` for FP32. The server/model contract is unchanged, but browser camera/XAI performance with CUDA has not been measured. The user elected to perform camera checks.

Both configs require exactly **20 distinct RGB samples at 5 fps**, a four-second window. Short clips raise an error rather than being padded. The browser and desktop default to four-second windows and one-second prediction updates. Pixel preprocessing preserves the original TensorFlow bilinear resize-with-padding to 172x172 and normalization to [0,1]; preprocessing remains on CPU. Each invocation starts from fresh MoViNet states.

## What was measured

RTX 4000 Ada Generation, native Windows, Intel Xeon w7-2595X:

| Runtime | Median model invocation (ms) | Median app processing (ms) |
|---|---:|---:|
| Original TensorFlow CPU FP32 | 119.61 | 287.85 |
| ONNX CUDA FP32 | 31.74 | 91.90 |
| ONNX CUDA FP16 | 26.69 | 103.17 |

Model invocation uses 18 real validation windows covering all 18 classes, five repeats per backend. App processing uses seven rolling windows from one established test-split video, three repeats per backend; it includes preprocessing but excludes decode/camera/display. These are different workloads, and host load affects timings. FP32 was faster for full app processing in this run; FP16 was faster for model invocation and reduces the ONNX file from 8.43 MB to 5.00 MB. The four-second context requirement remains even with faster computation.

Both exports matched the original predicted class on 18/18 sampled windows. This checks conversion fidelity, not recognition correctness or full-dataset accuracy. Independent profiling observed all executed graph operators on CUDA, with no CPU node execution. Computational CPU fallback is disabled. A registered CPU provider in `get_providers()` does not by itself indicate CPU execution.

The original wrapper produced these timing measurements. [Archived report](benchmarks/cuda/REPORT.md), [JSON](benchmarks/cuda/benchmark.json), [CSV](benchmarks/cuda/latency.csv) and [LaTeX table](benchmarks/cuda/latency_tables.tex) preserve the scope and numerical errors. Publication migration checks, when executed, are recorded separately in `benchmarks/cuda/adapter_migration.json`; they are not additional latency or accuracy trials.

## Explanations and precision

The ONNX adapter exposes prediction only. Temporal perturbation explanations can use it. Spatial gradients, exact affine-head evidence, concept hooks and channel interventions remain available through `configs/movinet_kard.json`, whose TensorFlow model has those hooks. The dashboard advertises supported capabilities instead of silently approximating missing hooks.

FP16 learned weights and eligible internal tensors retain float32 input/output; some kernels accumulate in FP32. The final converter allows CUDA-supported Min/Max/CumSum in FP16 and has two executed boundary casts. FP16 requests `use_tf32=1` to allow eligible Tensor Core plans; FP32 requests `use_tf32=0`. These options do not prove individual hardware-kernel math choices. Numerical comparison and operation placement were measured separately. The archive documents cuDNN plan retries and initial conversion attempts.

## Reproduce without opening a camera

```sh
python scripts/desktop/realtime_kard.py --config configs/movinet_kard_cuda_fp16.json --benchmark-video /path/to/KARD/Walk/a14_s09_e03.mp4 --output outputs/desktop_fp16.json
python scripts/verify_cuda_migration.py --dataset /path/to/KARD --experiment-dir /path/to/kard_a0_head_100pct
```

The migration check reads the same frame indices as the archived paired experiment, checks original SavedModel source hashes, and compares both packaged adapters' pixel tensors and logits. The full original paired benchmark also requires a CUDA-enabled PyTorch installation for runtime diagnostics/DLLs and the original experiment's split CSV:

```sh
python scripts/benchmark_kard_cuda.py --dataset /path/to/KARD --experiment-dir /path/to/kard_a0_head_100pct --repeats 5 --samples-per-class 1
```

These scripts leave fresh results in `outputs/`, which is ignored by Git. No dataset or camera frames are uploaded. To export another trained checkpoint or a different frame count, install `.[cuda,onnx-export]` and run:

```sh
python scripts/export_kard_onnx.py --experiment-dir /path/to/experiment --output-dir outputs/onnx_export --frames 20 --cuda-fp16-ops
```

The source experiment must contain `saved_model/`, `protocol.json` and `labels.json`. Re-run numerical agreement and placement checks before selecting new exports. `--reuse-fp32` validates saved source/artifact hashes; it cannot reuse an unrelated checkpoint. Archived reports retain earlier tuning attempts separately, with local paths replaced by placeholders.
