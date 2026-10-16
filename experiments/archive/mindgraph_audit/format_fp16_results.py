"""Publish fresh paired GPU results and update the existing LaTeX table in place."""
import csv
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE/'latency_benchmark_fp16'
OLD = HERE/'latency_benchmark'
data = json.loads((ROOT/'summary.json').read_text())
assert len(data['models'])==5
names = {'kth':'KTH','uav':'UAV Actions','kard':'KARD','hon4d':'HON4D','ucf45':'UCF45-DS'}
backends = ['pytorch_gpu_fp32','onnx_gpu_fp32','pytorch_gpu_fp16','onnx_gpu_fp16']
rows,tex_rows,accuracy_rows = [],[],[]

for model in data['models']:
    cells = [f"{model['stats'][name]['mean_ms']:.2f} ± {model['stats'][name]['std_ms']:.2f}"
             for name in backends]
    speed = model['speedups_vs_pytorch_fp32']['onnx_gpu_fp16']
    rows.append('| '+' | '.join([names[model['model_id']],*cells,f'{speed:.2f}×'])+' |')
    tex_cells = [rf"${model['stats'][name]['mean_ms']:.2f} \pm {model['stats'][name]['std_ms']:.2f}$"
                 for name in backends]
    tex_rows.append(' & '.join([names[model['model_id']],*tex_cells,rf'{speed:.2f}\,$\times$'])+r' \\')
    pt,ort = model['agreement']['pytorch_gpu_fp16'],model['agreement']['onnx_gpu_fp16']
    accuracy_rows.append(f"| {names[model['model_id']]} | {pt['top1_matches']}/{pt['total']} | "
                        f"{ort['top1_matches']}/{ort['total']} | {ort['max_abs_logit_error']:.6f} | "
                        f"{ort['max_abs_probability_error']:.6f} |")

table = ('| Model | PyTorch FP32 (ms) | ONNX FP32 (ms) | PyTorch FP16 (ms) | ONNX FP16 (ms) | ONNX FP16 speedup¹ |\n'
         '|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows))
agreement = ('| Model | PyTorch FP16 top-1 agreement | ONNX FP16 top-1 agreement | ONNX FP16 max logit error | ONNX FP16 max probability error |\n'
             '|---|---:|---:|---:|---:|\n'+'\n'.join(accuracy_rows))
notes = '''Values are mean ± sample standard deviation in milliseconds per eight-frame inference window; batch 1, RGB 224×224. ¹Speedup is ONNX FP16 versus the freshly measured original PyTorch FP32 mean. All four GPU backends were remeasured together; these FP32 baselines supersede the earlier GPU measurements for this comparison.

NVIDIA RTX 4000 Ada Generation, 20 GB VRAM; official Microsoft onnxruntime-gpu 1.31.0 with CUDA EP, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0. Each backend had five initial warm-up calls, untimed parity checks, twenty additional warm-up calls immediately before timing, and 45 timed calls (nine fixed windows × five repetitions). Inputs and backend order were randomized within a continuous GPU-only block. TF32 was disabled for FP32. Timing used synchronized wall time, GPU-resident inputs/outputs and ONNX I/O binding, with no profiler active during timing. Loading, preprocessing, frame collection, video decoding, host-device copies, dashboard/network overhead and XAI are excluded.

PyTorch FP16 uses half-precision model weights and inputs in eager evaluation/inference mode. ONNX FP16 was converted from the FP32 graph using onnxconverter-common, with FP16 inputs/outputs and the default unsupported-operator block list. Constants round to half precision (min_positive_val=0, max_finite_val=65504). ONNX may perform internal accumulation in higher precision. No retraining or calibration was used. CPU fallback was prohibited during session creation, and independent post-timing profiles confirmed CUDA execution of every executed ONNX node, including all three LSTMs.

Conversion repair: all 24 attention Cast targets were changed from FLOAT to FLOAT16 to match the converter's declared output types. Full ONNX shape/type checking passed after this repair. Source FP32 exports and trained checkpoints were preserved. FP16 exports and separate manifests are in edge_exports/<model_id>/.

Agreement checks compare FP16 outputs with original PyTorch FP32 on the supplied four KTH/KARD demonstration clips. These nine windows have unknown training membership and do not form a held-out test set for all five datasets. Top-1 agreement is a conversion diagnostic, not task accuracy. Held-out labelled evaluation remains necessary before deployment. Standard deviations describe repeated calls on fixed inputs in one workstation session under normal OS background activity, not independent experimental runs.
'''
(ROOT/'REPORT.md').write_text('# GPU FP16 benchmark\n\n'+table+'\n\n'+notes+'\n'+agreement+'\n',encoding='utf-8')
with (ROOT/'latency_table.csv').open('w',newline='',encoding='utf-8') as handle:
    fields = ['model_id','backend','n','mean_ms','std_ms','median_ms','p95_ms','min_ms','max_ms']
    writer = csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    for model in data['models']:
        writer.writerows({'model_id':model['model_id'],'backend':name,**model['stats'][name]} for name in backends)
with (ROOT/'timings.csv').open('w',newline='',encoding='utf-8') as handle:
    writer = csv.DictWriter(handle,fieldnames=['model_id','backend','repeat','sample_index','video','latency_ms'])
    writer.writeheader()
    for model in data['models']:
        writer.writerows({'model_id':model['model_id'],**item} for item in model['timing_observations'])

md_path = OLD/'latency_table.md'
original_md = md_path.read_text(encoding='utf-8')
cpu_part = original_md.split('Table 2.')[0]
md_path.write_text(cpu_part+'Table 2. Fresh paired GPU FP32/FP16 inference latency.\n\n'+table+'\n\n'+notes+
                  '\n'+agreement+'\n\nCPU measurements retain the earlier session and its four-thread protocol. '
                  'Original CPU protocol and timing records remain in latency_benchmark/summary.json; '
                  'fresh GPU records are in latency_benchmark_fp16/summary.json and timings.csv.\n',encoding='utf-8')

tex_path = OLD/'latency_table.tex'
source = tex_path.read_text(encoding='utf-8')
backup = OLD/'latency_table_before_fp16.tex'
if not backup.exists():
    backup.write_text(source,encoding='utf-8')
gpu_tex = r'''\begin{table}[ht]
\centering
\caption{Fresh paired GPU inference latency using FP32 and FP16 PyTorch and ONNX execution.}
\label{tab:har_gpu_latency}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Model & \shortstack{PyTorch FP32\\(ms)} & \shortstack{ONNX FP32\\(ms)} & \shortstack{PyTorch FP16\\(ms)} & \shortstack{ONNX FP16\\(ms)} & \shortstack{ONNX FP16\\speedup} \\
\midrule
'''+ '\n'.join(tex_rows)+r'''
\bottomrule
\end{tabular*}
\end{table}'''
source,count = re.subn(r'\\begin\{table\}\[ht\]\s*\\centering\s*\\caption\{(?:GPU inference latency|Fresh paired GPU inference latency).*?\\end\{table\}',
                       lambda _:gpu_tex,source,flags=re.S)
assert count==1, 'Existing GPU table not found; preserve source for manual review.'
source = source.replace('Speedup equals the original PyTorch mean divided by the corresponding ONNX mean on the same device; values below 1 indicate a slowdown.',
    'Speedup equals the original PyTorch FP32 mean divided by the corresponding ONNX mean on the same device. '
    'The GPU FP32 baselines were remeasured together with FP16 in a fresh, randomized four-backend GPU-only block; CPU results retain the earlier session.')
source = source.replace('XAI computation is excluded. GPU node placement was independently profiled after timing.',
    'XAI computation is excluded. GPU node placement was independently profiled after timing; all executed ONNX FP32/FP16 nodes used CUDA, with CPU fallback prohibited.')
source = source.replace('Measurements cover a single workstation session under normal OS background activity.',
    'CPU and GPU tables each cover a single workstation measurement session under normal OS background activity.')
fp16_notes = r'''
\medskip\noindent
\textit{FP16 conversion and validation.} PyTorch uses half-precision weights/inputs; ONNX exports have FP16 inputs/outputs and use the official CUDA provider.
No retraining or calibration was used. Internal accumulation may use higher precision.
Attention Cast target types were corrected to match the converted tensor types; full ONNX shape/type checks passed.
ONNX FP16 top-1 agreement with PyTorch FP32 on the nine supplied windows was
'''+', '.join(f"{names[m['model_id']]} {m['agreement']['onnx_gpu_fp16']['top1_matches']}/9" for m in data['models'])+r'''.
This is conversion agreement, not held-out task accuracy. Raw GPU timing and profile records are retained separately from the earlier CPU measurements.
'''
if r'\textit{FP16 conversion and validation.}' not in source:
    source = source.replace(r'\end{document}',fp16_notes+'\n'+r'\end{document}')
tex_path.write_text(source,encoding='utf-8')

# Keep the existing CSV downloads consistent with the newly displayed tables.
previous = json.loads((OLD/'summary.json').read_text())
combined_stats,combined_times = [],[]
for item in previous['models']:
    for backend,values in item['stats'].items():
        if '_cpu_' in backend:
            combined_stats.append({'model':names[item['model_id']],'backend':backend,
                                   'source_session':'original_cpu',**values})
    combined_times.extend({'model_id':item['model_id'],'source_session':'original_cpu',**obs}
                          for obs in item['timing_observations'] if '_cpu_' in obs['backend'])
for item in data['models']:
    combined_stats.extend({'model':names[item['model_id']],'backend':backend,
                           'source_session':'paired_gpu_fp16',**values}
                          for backend,values in item['stats'].items())
    combined_times.extend({'model_id':item['model_id'],'source_session':'paired_gpu_fp16',**obs}
                          for obs in item['timing_observations'])
for filename,records in [('latency_table.csv',combined_stats),('timings.csv',combined_times)]:
    with (OLD/filename).open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
print(table+'\n\n'+agreement)
