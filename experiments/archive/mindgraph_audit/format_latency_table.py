"""Create article-ready tables and reproducibility notes from measured data."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'latency_benchmark'
data = json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
assert len(data['models']) == 5, 'Wait for all five models to finish'
names = {'kth':'KTH','uav':'UAV Actions','kard':'KARD','hon4d':'HON4D','ucf45':'UCF45-DS'}


def value(item, backend):
    s = item['stats'][backend]
    return f"{s['mean_ms']:.2f} ± {s['std_ms']:.2f}"


def latex_value(item, backend):
    s = item['stats'][backend]
    return rf"${s['mean_ms']:.2f} \pm {s['std_ms']:.2f}$"


cpu_rows, gpu_rows, tex_cpu, tex_gpu = [], [], [], []
csv_rows = []
for item in data['models']:
    s = item['speedups']
    cpu_rows.append(f"| {names[item['model_id']]} | {value(item,'pytorch_cpu_fp32')} | "
                    f"{value(item,'onnx_cpu_fp32')} | {s['cpu_fp32']:.2f}× | "
                    f"{value(item,'onnx_cpu_int8')} | {s['cpu_int8']:.2f}× |")
    gpu_rows.append(f"| {names[item['model_id']]} | {value(item,'pytorch_gpu_fp32')} | "
                    f"{value(item,'onnx_gpu_fp32')} | {s['gpu_fp32']:.2f}× |")
    tex_cpu.append(' & '.join([names[item['model_id']],latex_value(item,'pytorch_cpu_fp32'),
                    latex_value(item,'onnx_cpu_fp32'),f"{s['cpu_fp32']:.2f}\\,$\\times$",
                    latex_value(item,'onnx_cpu_int8'),f"{s['cpu_int8']:.2f}\\,$\\times$"])+r' \\')
    tex_gpu.append(' & '.join([names[item['model_id']],latex_value(item,'pytorch_gpu_fp32'),
                    latex_value(item,'onnx_gpu_fp32'),f"{s['gpu_fp32']:.2f}\\,$\\times$"])+r' \\')
    for backend,stats in item['stats'].items():
        csv_rows.append({'model':names[item['model_id']],'backend':backend,**stats})

cpu_table = '''| Dataset-specific model | Original PyTorch FP32 (ms) | ONNX FP32 (ms) | FP32 speedup | ONNX INT8 (ms) | INT8 speedup |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(cpu_rows)
gpu_table = '''| Dataset-specific model | Original PyTorch FP32 (ms) | ONNX FP32 (ms) | Speedup |
|---|---:|---:|---:|
'''+ '\n'.join(gpu_rows)
protocol = data['protocol']
markdown = f'''# HAR inference latency on CPU and GPU

Table 1. CPU inference latency of the original PyTorch checkpoints and exported ONNX models.

{cpu_table}

Table 2. GPU inference latency of the original PyTorch checkpoints and FP32 ONNX models.

{gpu_table}

Values are mean ± sample standard deviation in milliseconds per eight-frame inference window, batch size 1, input shape [1,8,3,224,224]. Speedup is the original PyTorch mean latency divided by the corresponding ONNX mean latency on the same device; values below 1 mean a slowdown.

Hardware: Intel Xeon w7-2595X (26 physical cores/52 logical processors), four intra-op threads and one inter-op thread; NVIDIA RTX 4000 Ada Generation (20 GB VRAM); approximately 128 GiB system memory; Windows 11 Pro for Workstations. Python {data['python']}, PyTorch {data['torch']}, ONNX Runtime {data['onnxruntime']}, CUDA {data['torch_cuda']}, cuDNN {data['cudnn']}. GPU driver 580.97 was recorded before benchmarking.

Protocol: {protocol['warmup_calls_per_backend']} initial untimed warm-up calls per backend, followed by untimed numerical parity checks. CPU and GPU were measured in separate blocks, with {protocol['additional_warmup_calls_per_gpu_backend_before_gpu_block']} additional warm-up calls per GPU backend immediately before its block. Each backend received {protocol['timed_calls_per_backend']} timed calls (nine fixed windows × five repetitions). Input and backend order were randomized within each device block and repetition with seed {protocol['seed']}. Calls were sequential. GPU measurements used synchronized wall time and pre-bound device-resident input/output tensors. FP32 comparisons disabled TF32 in both runtimes. All models used evaluation/inference mode and classification outputs without XAI computation. ONNX Runtime enabled its default graph optimizations; PyTorch used the original eager model. CPU idle thread spinning was disabled for ONNX Runtime, and OpenMP used a passive wait policy. Each backend retained its model/session throughout timed execution.

Timings exclude checkpoint loading, session creation, video decoding, frame collection, preprocessing, host-device input/output copies and dashboard/network overhead. They are inference-window latencies, not per-frame or end-to-end camera latencies. Data preprocessing matches the existing showcase's sampled eight-frame preview protocol, not the reported research evaluation protocol.

All five FP32 ONNX GPU sessions were independently profiled after timing. The per-model JSON files contain executed-node provider counts. Checkpoint and input metadata are preserved in the conversion manifests and benchmark summary. Raw timings, parity checks, median, p95 and extrema are available in summary.json and timings.csv.

The INT8 graph is mixed precision and was benchmarked only with ONNX Runtime CPU. It is not a GPU INT8 measurement. INT8 prediction agreement checks are conversion diagnostics; accuracy should be evaluated on held-out data for each task before replacing the original model. The supplied windows span four KTH/KARD demonstration clips, with unknown training membership, and are not a test set for all five model vocabularies.

This table reports a single workstation measurement session under normal OS background activity. The standard deviations describe repeated calls on these fixed inputs; they are not independent experimental runs or cross-device uncertainty. Before claiming a general hardware performance advantage in a publication, repeat the experiment across fresh process runs and the intended deployment hardware.

Reproduce with the xai interpreter:

```powershell
& 'C:\\Users\\Xerox\\miniconda3\\envs\\xai\\python.exe' -B 'C:\\Users\\DIAT\\ashish\\xai\\mindgraph_audit\\benchmark_har_devices.py'
```
'''
(ROOT/'latency_table.md').write_text(markdown,encoding='utf-8')
with (ROOT/'latency_table.csv').open('w',newline='',encoding='utf-8') as handle:
    writer = csv.DictWriter(handle,fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
with (ROOT/'timings.csv').open('w',newline='',encoding='utf-8') as handle:
    writer = csv.DictWriter(handle,fieldnames=['model_id','backend','repeat','sample_index','video','latency_ms'])
    writer.writeheader()
    for item in data['models']:
        writer.writerows({'model_id':item['model_id'],**obs} for obs in item['timing_observations'])

latex = r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.4cm]{geometry}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\pagestyle{empty}
\begin{document}
\begin{table}[ht]
\centering
\caption{CPU inference latency of dataset-specific ViT-B/16 plus residual BiLSTM checkpoints before and after ONNX export.}
\label{tab:har_cpu_latency}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Model & \shortstack{PyTorch FP32\\(ms)} & \shortstack{ONNX FP32\\(ms)} & \shortstack{FP32\\speedup} & \shortstack{ONNX INT8\\(ms)} & \shortstack{INT8\\speedup} \\
\midrule
'''+ '\n'.join(tex_cpu)+r'''
\bottomrule
\end{tabular*}
\end{table}

\begin{table}[ht]
\centering
\caption{GPU inference latency of the same checkpoints using FP32 PyTorch and FP32 ONNX execution.}
\label{tab:har_gpu_latency}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Model & PyTorch FP32 (ms) & ONNX FP32 (ms) & Speedup \\
\midrule
'''+ '\n'.join(tex_gpu)+r'''
\bottomrule
\end{tabular*}
\end{table}

\noindent\footnotesize
\textit{Measurement notes.} Values are mean $\pm$ sample standard deviation per eight-frame window, batch size 1, RGB resolution $224\times224$.
Each backend received five initial untimed warm-up calls and separate parity checks.
CPU and GPU were measured in separate blocks, with 20 additional warm-up calls per GPU backend immediately before the GPU block.
There were 45 timed calls per backend over nine fixed windows with five repetitions and randomized input/backend order within each device block.
Speedup equals the original PyTorch mean divided by the corresponding ONNX mean on the same device; values below 1 indicate a slowdown.

\medskip\noindent
\textit{Hardware and runtime.} Intel Xeon w7-2595X (26 physical cores), four intra-op threads and one inter-op thread;
NVIDIA RTX 4000 Ada Generation (20 GB VRAM); approximately 128 GiB RAM; Windows 11 Pro for Workstations.
PyTorch 2.13.0+cu130; ONNX Runtime 1.31.0; CUDA 13.0; NVIDIA driver 580.97.
FP32 execution disables TF32 in both runtimes. ONNX uses default graph optimizations; PyTorch uses eager evaluation/inference mode.

\medskip\noindent
\textit{Timing scope.} GPU measurements use synchronized wall time with input/output tensors resident on the GPU through I/O binding.
Model loading, session creation, frame collection, decoding, preprocessing, host-device copies and dashboard/network overhead are excluded.
XAI computation is excluded. GPU node placement was independently profiled after timing.
The INT8 graph is mixed precision and targets CPU; no GPU INT8 latency is reported.

\medskip\noindent
\textit{Interpretation.} Measurements cover a single workstation session under normal OS background activity.
The nine windows are supplied KTH/KARD demonstrations and do not form a held-out accuracy test for all five model vocabularies.
Standard deviations describe repeated calls on fixed inputs, not independent experimental runs.
These are steady-state execution latencies, not per-frame or end-to-end camera latencies.
\end{document}
'''
(ROOT/'latency_table.tex').write_text(latex,encoding='utf-8')
print(cpu_table+'\n\n'+gpu_table)

# When FP16 measurements exist, keep current tables and downloads on the fresh
# paired GPU comparison instead of inadvertently reverting to the old baseline.
fp16_summary = ROOT.parent/'latency_benchmark_fp16/summary.json'
if fp16_summary.exists() and len(json.loads(fp16_summary.read_text())['models'])==5:
    import runpy
    runpy.run_path(str(ROOT.parent/'format_fp16_results.py'))
