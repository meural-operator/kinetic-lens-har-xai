# GPU FP16 benchmark

| Model | PyTorch FP32 (ms) | ONNX FP32 (ms) | PyTorch FP16 (ms) | ONNX FP16 (ms) | ONNX FP16 speedup¹ |
|---|---:|---:|---:|---:|---:|
| KTH | 33.10 ± 2.52 | 32.23 ± 1.76 | 23.89 ± 10.91 | 14.08 ± 2.63 | 2.35× |
| UAV Actions | 32.93 ± 1.41 | 33.36 ± 1.73 | 19.08 ± 8.80 | 13.98 ± 3.56 | 2.36× |
| KARD | 33.50 ± 1.48 | 32.61 ± 1.43 | 18.18 ± 7.68 | 13.68 ± 3.12 | 2.45× |
| HON4D | 33.73 ± 1.62 | 32.81 ± 1.38 | 18.82 ± 7.72 | 13.78 ± 2.67 | 2.45× |
| UCF45-DS | 33.44 ± 0.89 | 33.74 ± 1.42 | 12.77 ± 2.41 | 12.47 ± 0.79 | 2.68× |

Values are mean ± sample standard deviation in milliseconds per eight-frame inference window; batch 1, RGB 224×224. ¹Speedup is ONNX FP16 versus the freshly measured original PyTorch FP32 mean. All four GPU backends were remeasured together; these FP32 baselines supersede the earlier GPU measurements for this comparison.

NVIDIA RTX 4000 Ada Generation, 20 GB VRAM; official Microsoft onnxruntime-gpu 1.31.0 with CUDA EP, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0. Each backend had five initial warm-up calls, untimed parity checks, twenty additional warm-up calls immediately before timing, and 45 timed calls (nine fixed windows × five repetitions). Inputs and backend order were randomized within a continuous GPU-only block. TF32 was disabled for FP32. Timing used synchronized wall time, GPU-resident inputs/outputs and ONNX I/O binding, with no profiler active during timing. Loading, preprocessing, frame collection, video decoding, host-device copies, dashboard/network overhead and XAI are excluded.

PyTorch FP16 uses half-precision model weights and inputs in eager evaluation/inference mode. ONNX FP16 was converted from the FP32 graph using onnxconverter-common, with FP16 inputs/outputs and the default unsupported-operator block list. Constants round to half precision (min_positive_val=0, max_finite_val=65504). ONNX may perform internal accumulation in higher precision. No retraining or calibration was used. CPU fallback was prohibited during session creation, and independent post-timing profiles confirmed CUDA execution of every executed ONNX node, including all three LSTMs.

Conversion repair: all 24 attention Cast targets were changed from FLOAT to FLOAT16 to match the converter's declared output types. Full ONNX shape/type checking passed after this repair. Source FP32 exports and trained checkpoints were preserved. FP16 exports and separate manifests are in edge_exports/<model_id>/.

Agreement checks compare FP16 outputs with original PyTorch FP32 on the supplied four KTH/KARD demonstration clips. These nine windows have unknown training membership and do not form a held-out test set for all five datasets. Top-1 agreement is a conversion diagnostic, not task accuracy. Held-out labelled evaluation remains necessary before deployment. Standard deviations describe repeated calls on fixed inputs in one workstation session under normal OS background activity, not independent experimental runs.

| Model | PyTorch FP16 top-1 agreement | ONNX FP16 top-1 agreement | ONNX FP16 max logit error | ONNX FP16 max probability error |
|---|---:|---:|---:|---:|
| KTH | 9/9 | 9/9 | 0.009779 | 0.002280 |
| UAV Actions | 9/9 | 9/9 | 0.005586 | 0.001199 |
| KARD | 9/9 | 9/9 | 0.007193 | 0.000844 |
| HON4D | 9/9 | 9/9 | 0.007009 | 0.000949 |
| UCF45-DS | 9/9 | 9/9 | 0.009599 | 0.001705 |
