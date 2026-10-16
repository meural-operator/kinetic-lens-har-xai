# Reduced HAR inference models

Converted the bundled MeitY showcase checkpoints without changing or retraining the originals. Export uses ONNX opset 17, followed by mixed-precision dynamic INT8 quantization of constant MatMul and LSTM weights. Convolution, normalization and other operations remain floating point. The INT8 files target ONNX Runtime CPU, including its DynamicQuantizeLSTM extension; they are not universal INT8 graphs for every ONNX or TFLite runtime.

| Model | Original MB | FP32 ONNX MB | INT8 ONNX MB | File reduction | INT8/PyTorch top-1 agreement |
|---|---:|---:|---:|---:|---:|
| kth | 357.74 | 355.43 | 92.18 | 74.23% | 9/9 |
| uav | 357.74 | 355.43 | 92.18 | 74.23% | 8/9 |
| kard | 357.74 | 355.43 | 92.18 | 74.23% | 8/9 |
| hon4d | 357.74 | 355.43 | 92.18 | 74.23% | 8/9 |
| ucf45 | 357.76 | 355.45 | 92.18 | 74.23% | 9/9 |

MB means 1,000,000 bytes. The FP32 graph omits unused model computation such as the ViT pooler. The much larger INT8 reduction comes from quantization, not changing the file extension.

## GPU FP16 exports

Each model folder additionally contains `<model_id>-fp16.onnx` and `fp16_manifest.json`.
These GPU exports use FLOAT16 inputs and outputs with the same fixed shape and
preprocessing as FP32. Each file is approximately 177.8 MB, about 50% smaller
than the FP32 ONNX export. No retraining or calibration was performed.

All five were tested using Microsoft's official `onnxruntime-gpu` 1.31.0 CUDA
execution provider on the RTX 4000 Ada. CPU fallback was prohibited, and
independent profiles verified 386 CUDA nodes per model, including three LSTMs.
Both PyTorch and ONNX FP16 retained the original FP32 class prediction on all
nine supplied windows for each model. This is conversion agreement, not a
held-out accuracy result. The FP16 conversion also repairs 24 attention Cast
target types that the conversion library otherwise leaves inconsistent.

Fresh GPU latency results, protocol, agreement errors, raw timings and profiles
are in `../latency_benchmark_fp16/REPORT.md` and `summary.json`. The existing
article table in `../latency_benchmark/latency_table.tex` includes these results.
Reproduce the GPU comparison with the xai interpreter and
`../benchmark_har_fp16.py`. Existing trained checkpoints are preserved.

The CPU `run_video.py` runner below supports FP32 and INT8; using FP16 requires
a CUDA session and float16 inputs. Do not pass the FP16 export to the CPU runner.

## CPU export validation scope

Each completed model passed the ONNX graph checker, actual ONNX Runtime CPU execution, expected output shape and finite-output checks. FP32 numerical parity was checked against the loaded PyTorch model with a maximum absolute logit tolerance of 0.002. Each model's validation.json contains exact observed errors, label agreement, sampled frame indices and CPU inference timings.

The comparison uses nine full eight-frame windows across four supplied KTH/KARD demonstration videos. Other model vocabularies were also run on these same inputs to test conversion behavior. This is not a held-out accuracy study for any dataset. Matching an original model's label is not proof that the label is correct. The originals' training membership for these clips is unknown. Quantization can change probabilities and may change labels on other samples.

CPU timings are warmed inference-only measurements, four threads and three repetitions per input window on this Windows workstation. They exclude capture, preprocessing and session/model loading and are not mini-PC performance guarantees. Check target-device RAM, latency and held-out recognition quality before replacing the original inference model.

## Input/output contract

- Fixed input: batch 1, eight frames, three channels, 224 x 224; float32 tensor layout [B,T,C,H,W].
- RGB, resize with INTER_AREA, divide by 255, normalize with mean/std 0.5.
- Sample approximately four frames per second; videos are grouped into non-overlapping eight-frame windows in the supplied runner.
- Short final windows are explicitly marked and padded by repeating the last frame. They can behave differently from full windows.
- Output: raw action-class logits, with the matching dataset's class map stored in manifest.json.
- This uses the showcase's sampled preview protocol, not its reported research evaluation protocol. Other sequence lengths require separate exports.
- The exported models produce HAR logits only. The existing PyTorch attention/TAM gradient workflow is not included in these artifacts.

## Run on this computer

```powershell
& 'python' -B '<local-workspace>\mindgraph_audit\edge_exports\run_video.py' --model-dir '<local-workspace>\mindgraph_audit\edge_exports\kth' --video '<original-showcase>\static\media\kth-boxing\original.mp4'
```

Add --variant fp32 to run the unquantized ONNX version, or --output predictions.json to save results. Portable runner execution was also checked on the KTH boxing clip, including final-window padding. It requires NumPy, OpenCV and ONNX Runtime; it does not load PyTorch or the original showcase backend.

## Deploy to another CPU computer

Copy run_video.py and the chosen model's manifest.json and *-int8.onnx file. Install numpy, opencv-python and onnxruntime in the target runtime; the tested ONNX Runtime version is recorded in summary.json. The optional_deps fallback only serves this local audit installation and is not required on the target machine.

## Reproduce export

```powershell
& 'python' -B '<local-workspace>\mindgraph_audit\export_har_onnx.py' --models kth uav kard hon4d ucf45
```

The exporter reads the original checkpoint through the showcase's strict loader. Conversion dependencies were installed only in mindgraph_audit/optional_deps; the xai environment was not modified. Windows sandbox permission failures in ONNX Runtime temporary external-data writing required an approved execution with filesystem access. All converted models and validation outputs are under edge_exports.

## Retraining, calibration and TFLite

FP32 format export and the dynamic INT8 method here do not retrain or fine-tune the model and do not require a calibration dataset. The clips were used to validate results, not to learn new weights. Static INT8 methods generally need representative calibration inputs without updating the model's trained weights. Quantization-aware training is a different procedure that fine-tunes/retrains to accommodate reduced precision.

Direct TFLite conversion was not performed. The official LiteRT Torch converter currently specifies Linux as its conversion host, while this user-requested xai environment is Windows. TFLite remains a separate possible target if required; bidirectional recurrent layers and target runtime compatibility would need actual conversion and testing.

Sources: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [official LiteRT Torch requirements](https://github.com/google-ai-edge/litert-torch#requirements-and-dependencies).
