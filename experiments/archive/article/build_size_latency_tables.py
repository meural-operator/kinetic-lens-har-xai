"""Reconcile historical timing records with model artifacts; no new benchmarks.

Run in Miniconda xai. All paths are confined to the existing workspace.
Preserves the original open LaTeX file once, then updates it in place.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / 'mindgraph_audit'
MOV = ROOT / 'movinet_evaluation'
OPEN_TEX = AUDIT / 'latency_benchmark/latency_table.tex'
READS = {}
ARTIFACTS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def load(path):
    READS[relative(path)] = sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def artifact(path, expected_bytes=None, expected_sha=None):
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob('*') if p.is_file())
        count = sum(p.stat().st_size for p in files)
        components = [{'path': relative(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
        digest = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    else:
        count = path.stat().st_size
        digest = sha(path)
        components = None
    if expected_bytes is not None:
        assert count == expected_bytes, (path, count, expected_bytes)
    if expected_sha is not None:
        assert digest == expected_sha, (path, digest, expected_sha)
    ARTIFACTS.append({'path': relative(path), 'bytes': count, 'sha256': digest,
                      'components': components, 'historical_hash_checked': expected_sha is not None,
                      'historical_byte_count_checked': expected_bytes is not None})
    return count


def verify_statistics(model):
    for backend, saved in model['stats'].items():
        values = [r['latency_ms'] for r in model['timing_observations'] if r['backend'] == backend]
        assert len(values) == saved['n'] == 45
        assert math.isclose(statistics.mean(values), saved['mean_ms'], abs_tol=1e-9)
        assert math.isclose(statistics.stdev(values), saved['std_ms'], abs_tol=1e-9)


def number(value):
    return f'{value:.2f}'


def timing(value, tex=False):
    if value is None:
        return r'\textemdash' if tex else '—'
    if 'std_ms' in value:
        text = f"{value['mean_ms']:.2f} " + (r'\pm' if tex else '±') + f" {value['std_ms']:.2f}"
        return '$' + text + '$' if tex else text
    return f"{value['mean_ms']:.2f}"


def mb(value):
    return value / 1_000_000


def escaped(text):
    return text.replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


cpu = load(AUDIT / 'latency_benchmark/summary.json')
gpu = load(AUDIT / 'latency_benchmark_fp16/summary.json')
exports = load(AUDIT / 'edge_exports/summary.json')
stock = load(MOV / 'results.json')
downloads = load(MOV / 'downloads.json')
cuda = load(MOV / 'onnx_cuda/benchmark.json')
cuda_manifest = load(MOV / 'onnx_cuda/manifest.json')
rows, vit_rows, agreement_rows, mov_rows, stock_rows = [], [], [], [], []
FILES = {'kth': 'kth_vit_bilstm.pth', 'uav': 'UAV_actionsvit_gru_model.pth',
         'kard': 'kard_vit_gru_model.pth', 'hon4d': 'hon4d_vit_gru_model.pth',
         'ucf45': 'UCF45-DSvit_gru_model.pth'}

for cm in cpu['models']:
    mid = cm['model_id']
    gm = next(m for m in gpu['models'] if m['model_id'] == mid)
    em = next(m for m in exports['models'] if m['model_id'] == mid)
    verify_statistics(cm)
    verify_statistics(gm)
    folder = AUDIT / 'edge_exports' / mid
    manifest = load(folder / 'manifest.json')
    half = load(folder / 'fp16_manifest.json')
    validation = load(folder / 'validation.json')
    assert manifest['architecture'] == 'ViT-B/16 + additive residual BiLSTM'
    assert half['source_checkpoint_sha256'] == manifest['checkpoint_sha256']
    orig_path = ROOT / 'code_repo/weights' / FILES[mid]
    orig_bytes = artifact(orig_path, em['checkpoint_bytes'], manifest['checkpoint_sha256'])
    sizes = {'fp32': artifact(folder / f'{mid}-fp32.onnx', half['fp32_bytes']),
             'fp16': artifact(folder / f'{mid}-fp16.onnx', half['fp16_bytes'], half['fp16_sha256']),
             'int8': artifact(folder / f'{mid}-int8.onnx', validation['stages']['int8_graph']['bytes'])}
    for prec in ('fp32', 'fp16'):
        providers = gm['gpu_placement'][prec]['executed_node_counts']
        assert providers == {'CUDAExecutionProvider': 386}
    specs = [('PyTorch', 'FP32', orig_bytes, 'pytorch_cpu_fp32', 'pytorch_gpu_fp32', orig_path),
             ('ONNX', 'FP32', sizes['fp32'], 'onnx_cpu_fp32', 'onnx_gpu_fp32', folder / f'{mid}-fp32.onnx'),
             ('PyTorch', 'FP16', None, None, 'pytorch_gpu_fp16', None),
             ('ONNX', 'FP16', sizes['fp16'], None, 'onnx_gpu_fp16', folder / f'{mid}-fp16.onnx'),
             ('ONNX', 'Mixed INT8', sizes['int8'], 'onnx_cpu_int8', None, folder / f'{mid}-int8.onnx')]
    for runtime, precision, size, ck, gk, model_path in specs:
        row = {'family': 'ViT-B/16 + residual BiLSTM', 'model': cm['dataset'], 'runtime': runtime,
               'precision': precision, 'artifact_bytes': size, 'artifact_mb': mb(size) if size else None,
               'artifact_path': relative(model_path) if model_path else None,
               'cpu': cm['stats'].get(ck), 'gpu': gm['stats'].get(gk),
               'cpu_source': 'mindgraph_audit/latency_benchmark/summary.json' if ck else None,
               'gpu_source': 'mindgraph_audit/latency_benchmark_fp16/summary.json' if gk else None,
               'frames': 8, 'resolution': 224, 'batch_size': 1,
               'compression_vs_onnx_fp32': sizes['fp32'] / size if runtime == 'ONNX' else None,
               'cpu_speedup_vs_pytorch_fp32': cm['stats']['pytorch_cpu_fp32']['mean_ms'] / cm['stats'][ck]['mean_ms'] if ck else None,
               'gpu_speedup_vs_pytorch_fp32': gm['stats']['pytorch_gpu_fp32']['mean_ms'] / gm['stats'][gk]['mean_ms'] if gk else None,
               'original_parameter_count': em['stages']['pytorch_reference']['parameter_count']}
        vit_rows.append(row)
        rows.append(row)
    agreement_rows.append({
        'model': cm['dataset'],
        'onnx_fp16': f"{gm['agreement']['onnx_gpu_fp16']['top1_matches']}/{gm['agreement']['onnx_gpu_fp16']['total']}",
        'onnx_int8': f"{validation['stages']['int8_runtime']['top1_agreement_count']}/{validation['stages']['int8_runtime']['windows']}"})

# These stock benchmarks used different graph definitions and a fixed reference
# input. Report in their own table, without cross-family speedup claims.
for sm in stock['models']:
    downloaded = next(d for d in downloads if d['name'] == sm['model'])
    directory = MOV / 'models' / sm['model']
    total = artifact(directory, downloaded['unpacked_bytes'])
    weights = 0
    for name, expected in downloaded['variables_sha256'].items():
        vpath = directory / 'variables' / name
        assert sha(vpath) == expected
    # The historical download manifest counts the checkpoint data shards,
    # excluding variables.index. Keep that exact, explicitly named definition.
    weights = sum((directory / 'variables' / name).stat().st_size
                  for name in downloaded['variables_sha256'])
    assert weights == downloaded['variable_bytes']
    for mode, saved in sm['benchmarks'].items():
        assert len(saved['raw_ms']) == saved['n'] == 30
        assert math.isclose(statistics.mean(saved['raw_ms']), saved['mean_ms'], abs_tol=1e-8)
        assert math.isclose(statistics.stdev(saved['raw_ms']), saved['std_ms'], abs_tol=1e-8)
    row = {'family': 'Official MoViNet', 'model': sm['model'], 'runtime': 'TensorFlow',
           'precision': 'FP32', 'artifact_bytes': total, 'artifact_mb': mb(total),
           'artifact_path': relative(directory), 'variable_bytes': weights, 'variable_mb': mb(weights),
           'cpu': sm['benchmarks']['eight_frame_clip'], 'gpu': None,
           'cpu_streaming': sm['benchmarks'].get('continuous_streaming_frame'),
           'cpu_source': 'movinet_evaluation/results.json', 'frames': 8,
           'resolution': sm['resolution'], 'batch_size': 1}
    stock_rows.append(row)
    rows.append(row)

# The source checkpoint in this measurement is the historical KARD full-pool
# head, not the later strong ablation heads or LoRA checkpoints.
source = MOV / 'kard_a0_head_100pct/saved_model'
assert sha(source / 'saved_model.pb') == cuda_manifest['source_saved_model_pb_sha256']
assert sha(source / 'variables/variables.data-00000-of-00001') == cuda_manifest['source_variables_sha256']
assert cuda['artifact_manifest']['source_variables_sha256'] == cuda_manifest['source_variables_sha256']
source_size = artifact(source)
for prec in ('fp32', 'fp16'):
    assert cuda['artifact_manifest'][prec]['sha256'] == cuda_manifest[prec]['sha256']
    artifact(MOV / 'onnx_cuda' / cuda_manifest[prec]['file'],
             cuda_manifest[prec]['bytes'], cuda_manifest[prec]['sha256'])
    assert not any(provider != 'CUDAExecutionProvider' for provider in
                   cuda['placement'][prec]['executed_node_providers'])
for runtime, prec, device, key, size, filename in [
    ('TensorFlow', 'FP32', 'CPU', 'tensorflow_cpu_fp32', source_size, source),
    ('ONNX', 'FP32', 'GPU', 'onnx_cuda_fp32', cuda_manifest['fp32']['bytes'], MOV / 'onnx_cuda' / cuda_manifest['fp32']['file']),
    ('ONNX', 'FP16', 'GPU', 'onnx_cuda_fp16', cuda_manifest['fp16']['bytes'], MOV / 'onnx_cuda' / cuda_manifest['fp16']['file'])]:
    value = cuda['timings'][key]
    assert value['n'] == 90
    row = {'family': 'MoViNet-A0-Stream + trained KARD head', 'model': 'KARD',
           'runtime': runtime, 'precision': prec, 'artifact_bytes': size, 'artifact_mb': mb(size),
           'artifact_path': relative(filename), 'device': device,
           'cpu': value if device == 'CPU' else None, 'gpu': value if device == 'GPU' else None,
           'timing': value, 'cpu_source': 'movinet_evaluation/onnx_cuda/benchmark.json' if device == 'CPU' else None,
           'gpu_source': 'movinet_evaluation/onnx_cuda/benchmark.json' if device == 'GPU' else None,
           'frames': 20, 'resolution': 172, 'batch_size': 1}
    mov_rows.append(row)
    rows.append(row)

assert len(rows) == 31 and len(ARTIFACTS) == 26
assert sum(r['artifact_bytes'] is not None for r in rows) == 26
counts = {'base_architecture_types_with_latency': 4,
          'checkpoint_or_classifier_configurations_with_latency': 9,
          'precision_levels': ['FP32', 'FP16', 'Mixed dynamic INT8'],
          'measured_model_runtime_precision_configurations': 31,
          'distinct_saved_inference_artifacts': 26,
          'pytorch_fp16_in_memory_only_configurations': 5,
          'vit_dataset_checkpoints': 5, 'vit_runtime_precision_forms_per_checkpoint': 5,
          'official_movinet_variants': ['A0-Stream', 'A2-Stream', 'A2-Base'],
          'kard_movinet_runtime_precision_forms': 3,
          'excluded': 'Different devices, reruns, intermediate FP16 exports, training seeds, tuning candidates, training fractions and sampling rates are not additional model-size variants.'}
audit = {'scope': 'Reconciliation of existing measurements; no inference rerun.',
         'size_unit': 'MB = 1,000,000 bytes; serialized files or full SavedModel directory; not RAM/VRAM',
         'counts': counts, 'source_sha256': READS, 'artifacts': ARTIFACTS,
         'verification': {'vit_raw_mean_and_sample_sd_recomputed': True,
                          'stock_raw_mean_and_sample_sd_recomputed': True,
                          'movinet_kard': 'Saved aggregate mean/median/p95 only; per-call samples were not retained, so no reconstructed SD.',
                          'artifact_bytes': 'Current sizes match historical byte counts where recorded.',
                          'hash_limit': 'Current SHA256 inventory; historical hash checks only where manifests retain hashes. Historical FP32/INT8 ViT export hashes are unavailable.',
                          'no_new_latency_or_accuracy_benchmark': True},
         'rows': rows, 'conversion_agreement': agreement_rows}
(OUT / 'size_latency_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

fields = ['family', 'model', 'runtime', 'precision', 'artifact_bytes', 'artifact_mb',
          'frames', 'resolution', 'batch_size', 'cpu_n', 'cpu_mean_ms', 'cpu_sample_sd_ms',
          'cpu_median_ms', 'cpu_p95_ms', 'gpu_n', 'gpu_mean_ms', 'gpu_sample_sd_ms',
          'gpu_median_ms', 'gpu_p95_ms', 'compression_vs_onnx_fp32',
          'cpu_speedup_vs_pytorch_fp32', 'gpu_speedup_vs_pytorch_fp32',
          'artifact_path', 'cpu_source', 'gpu_source']
with (OUT / 'model_size_latency.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        record = {k: row.get(k) for k in fields}
        for device in ('cpu', 'gpu'):
            value = row.get(device) or {}
            for key, saved in [('n', 'n'), ('mean_ms', 'mean_ms'), ('sample_sd_ms', 'std_ms'),
                               ('median_ms', 'median_ms'), ('p95_ms', 'p95_ms')]:
                record[f'{device}_{key}'] = value.get(saved)
        writer.writerow(record)

# Self-contained tables that can be \input into a two-column article.
t1 = [
    r'\begin{table*}[t]', r'\centering', r'\small',
    r'\caption{Serialized model size and warm inference latency for five dataset-specific ViT-B/16 plus residual BiLSTM checkpoints. All inputs contain eight RGB frames at $224\times224$, batch size~1. CPU and GPU results use separate measurement sessions.}',
    r'\label{tab:har-size-latency-vit}', r'\setlength{\tabcolsep}{5pt}',
    r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrr@{}}', r'\toprule',
    r'Checkpoint & Runtime & Precision & Size (MB) & CPU (ms) & GPU (ms) \\', r'\midrule']
for i, row in enumerate(vit_rows):
    if i and i % 5 == 0:
        t1.append(r'\addlinespace[3pt]')
    size = number(row['artifact_mb']) if row['artifact_mb'] else r'\textemdash'
    t1.append(f"{escaped(row['model']) if i % 5 == 0 else ''} & {row['runtime']} & {row['precision']} & {size} & {timing(row['cpu'],True)} & {timing(row['gpu'],True)} " + r'\\')
t1 += [r'\bottomrule', r'\end{tabular*}', r'\par\smallskip',
       r'\begin{minipage}{\textwidth}\footnotesize',
       r'Values are mean $\pm$ sample SD over 45 calls (nine fixed windows, five repeats). Size uses decimal MB. A dash means unmeasured or, for PyTorch FP16 size, not serialized: that model was converted in memory. Mixed INT8 dynamically quantizes constant MatMul/LSTM weights; other operators remain floating point. No GPU INT8 result is available. FP16 is reduced floating-point precision, not integer quantization. CPU/GPU timing excludes loading, decoding, preprocessing, XAI and host--device transfers; GPU tensors were resident through I/O binding. Precision conversion agreement is not held-out recognition accuracy.',
       r'\end{minipage}', r'\end{table*}']

t2 = [
    r'\begin{table*}[t]', r'\centering', r'\small',
    r'\caption{Serialized size and warm model-invocation latency for MoViNet-A0-Stream with the historical trained KARD-18 affine head. Each invocation uses 20 RGB samples at $172\times172$, batch size~1, sampled at 5~fps, with fresh states.}',
    r'\label{tab:har-size-latency-kard}', r'\setlength{\tabcolsep}{5pt}',
    r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}', r'\toprule',
    r'Runtime & Precision & Device & Size (MB) & Mean (ms) & Median (ms) & P95 (ms) \\', r'\midrule']
for row in mov_rows:
    s = row['timing']
    t2.append(f"{row['runtime']} & {row['precision']} & {row['device']} & {row['artifact_mb']:.3f} & {s['mean_ms']:.2f} & {s['median_ms']:.2f} & {s['p95_ms']:.2f} " + r'\\')
t2 += [r'\bottomrule', r'\end{tabular*}', r'\par\smallskip',
       r'\begin{minipage}{\textwidth}\footnotesize',
       r'Ninety calls per backend: 18 validation windows, five repeats, three warm-up calls; randomized sample/backend order. Timing includes host--device transfer and synchronous CPU output for ONNX. It excludes decoding, resize, preprocessing, capture/display and XAI. Only aggregate timing statistics were retained, so SD is not reconstructed. TensorFlow size is the complete SavedModel directory (graph plus variables); ONNX size is one graph file. Differences across formats include graph serialization/optimization. FP16 ONNX retains FP32 input/output and eligible mixed internal precision. The matched ONNX FP16 export is 40.73\% smaller and 1.20$\times$ faster than ONNX FP32 by mean latency in this session. Neither this table nor Table~\ref{tab:har-size-latency-vit} is an end-to-end camera benchmark. Their different window lengths and transfer policies preclude a controlled cross-family speedup.',
       r'\end{minipage}', r'\end{table*}']
t3 = [
    r'\begin{table*}[t]', r'\centering', r'\small',
    r'\caption{Model scale and CPU latency for the three downloaded official Kinetics-600 MoViNet variants, using TensorFlow FP32. This is a separate fixed-input experiment.}',
    r'\label{tab:har-size-latency-stock}', r'\setlength{\tabcolsep}{5pt}',
    r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}', r'\toprule',
    r'Variant & Resolution & Package (MB) & Weights (MB) & 8-frame clip (ms) & Stream step (ms) \\', r'\midrule']
stock_names = {'a0_stream': 'A0-Stream', 'a2_stream': 'A2-Stream', 'a2_base': 'A2-Base'}
for row in stock_rows:
    t3.append(f"{stock_names[row['model']]} & ${row['resolution']}\\times{row['resolution']}$ & {row['artifact_mb']:.2f} & {row['variable_mb']:.2f} & {timing(row['cpu'],True)} & {timing(row['cpu_streaming'],True)} " + r'\\')
t3 += [r'\bottomrule', r'\end{tabular*}', r'\par\smallskip',
       r'\begin{minipage}{\textwidth}\footnotesize',
       r'Mean $\pm$ sample SD, 30 timed calls after five warm-ups. Eight-frame clips start from fresh states; stream-step timings retain states and process one frame. The latter cycle the 13 reference frames and are not latency per independent action decision. A2-Base has no retained-state benchmark. Package size includes graph metadata; weights means checkpoint data shards, excluding their index, including auxiliary stored tensors, and is not a parameter count. Models use 600-class pretrained heads, unlike the KARD-18 head. Loading, state initialization/reset, decoding, preprocessing and top-five formatting are excluded. These native TensorFlow CPU results do not establish mini-PC or GPU performance.',
       r'\end{minipage}', r'\end{table*}']
for name, lines in [('table_vit_size_latency.tex', t1), ('table_movinet_kard_size_latency.tex', t2),
                    ('table_movinet_scale_size_latency.tex', t3)]:
    (OUT / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'article_tables.tex').write_text('\n\n'.join('\n'.join(t) for t in (t1, t2, t3)) + '\n', encoding='utf-8')

hardware = r"""All measurements used an Intel Xeon w7-2595X (26 physical cores), four intra-op and one inter-op threads, NVIDIA RTX 4000 Ada Generation (20~GB), and Windows 11. ViT measurements used PyTorch 2.13.0+cu130 and official ONNX Runtime GPU 1.31.0; MoViNet used TensorFlow 2.21.0 and ONNX Runtime GPU 1.31.0. CPU results are native Windows CPU runs. In the ViT GPU block, FP32 TF32 was disabled, each backend had five initial and 20 additional GPU warm-ups, and all executed ONNX nodes were profiled on CUDA with CPU fallback prohibited. The MoViNet CUDA trial requested TF32 off for FP32 and eligible Tensor Core plans for FP16; no hardware Tensor Core trace was recorded."""
inventory = r"""The latency inventory covers \textbf{four base architectures} (ViT-B/16--BiLSTM, MoViNet A0-Stream, A2-Stream and A2-Base) and \textbf{nine checkpoint/classifier configurations}: five dataset-specific ViT models, three pretrained MoViNet models and one KARD-adapted A0. Across them, \textbf{31 model--runtime--precision configurations} were timed: $5\times5+3+3$. There are \textbf{26 distinct saved inference artifacts}: $5\times4+3+3$; five PyTorch FP16 variants were tested only in memory. Device changes, reruns, intermediate conversion files, training seeds and tuning candidates are not counted as extra size variants. Precision coverage is FP32, FP16 and mixed dynamic INT8. Only A0-Stream, A2-Stream and A2-Base were executed from the published MoViNet scale family; A1 and A3--A5 were described but not locally benchmarked."""
gaps = r"""The compact mean/temporal heads, capacity-matched MLP and three projection-adapter methods have training/accuracy results, but no complete deployed artifact size plus end-to-end inference latency measurements. Their small head files or trainable-parameter budgets cannot substitute for full-model size. Merged projection adapters retain the baseline graph and kernel dimensions; deployment latency has not been established. Different training fractions and 5/10-fps policies are not quantization levels. No recorded benchmark in this inventory covers MoViNet INT8, GPU INT8, INT4, static INT8/QAT, TFLite or TensorRT."""
quality = r"""ONNX FP16 preserved PyTorch FP32 top-one predictions on all nine demonstration windows for each ViT checkpoint. The original INT8 export validation preserved 9/9 (KTH), 8/9 (UAV Actions), 8/9 (KARD), 8/9 (HON4D) and 9/9 (UCF45-DS); some predictions therefore changed. Those nine fixed windows are not dataset-specific held-out accuracy tests. MoViNet ONNX FP32 and FP16 each matched TensorFlow on 18/18 selected validation windows. The KARD MoViNet source here is the historical full-training-pool affine head, not the later controlled-ablation or adapter checkpoints. All SDs describe repeated timing calls in a single session, not uncertainty across independent experiments or subjects."""
standalone = [
    r'\documentclass[10pt]{article}', r'\usepackage[a4paper,margin=1.4cm]{geometry}',
    r'\usepackage{booktabs}', r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}',
    r'\pagestyle{plain}', r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{5pt}',
    r'\begin{document}', *t1, r'\clearpage', *t2, r'\par', *t3, r'\clearpage',
    r'{\large\bfseries Measurement protocol and variant inventory}\par',
    r'\textbf{Hardware and runtime.} ' + hardware,
    r'\textbf{What has actually been tried.} ' + inventory,
    r'\textbf{Unmeasured configurations.} ' + gaps,
    r'\textbf{Conversion and recognition quality.} ' + quality,
    r'\textbf{Size definition.} File size means decimal MB ($10^6$ bytes), measured from the existing artifacts. It is not peak RAM, VRAM, working memory, FLOPs, compressed download size or trainable parameter count. Compare quantization compression within the same runtime format; package-to-ONNX differences also reflect graph and serialization changes.',
    r'\textbf{Provenance.} Values are reconciled from the saved raw ViT CPU/GPU timing observations, official MoViNet timing observations, MoViNet KARD aggregate benchmark, export manifests and current serialized files. Raw means and sample SDs were recomputed where per-call samples exist. Current file hashes and available historical hashes/byte counts were verified; unavailable historical hashes cannot be reconstructed. No new inference timings were measured for this report.',
    r'\end{document}']
backup = OPEN_TEX.with_name('latency_table_before_size_latency.tex')
if not backup.exists():
    shutil.copy2(OPEN_TEX, backup)
OPEN_TEX.write_text('\n'.join(standalone) + '\n', encoding='utf-8')

# Human-readable record with all measured combinations and sources.
md = ['# HAR model size and inference latency', '',
      'This report reconciles existing results; it does not run a new benchmark.', '',
      '**Coverage:** four base architectures; nine checkpoint/classifier configurations; '
      'three precision levels (FP32, FP16, mixed dynamic INT8); 31 measured model/runtime/precision forms; '
      '26 saved inference artifacts plus five PyTorch FP16 in-memory forms.', '',
      '## Five ViT–BiLSTM checkpoints: 8 RGB frames at 224×224', '',
      '| Checkpoint | Runtime | Precision | File MB | CPU ms | GPU ms |',
      '|---|---|---|---:|---:|---:|']
for r in vit_rows:
    md.append(f"| {r['model']} | {r['runtime']} | {r['precision']} | {r['artifact_mb']:.2f} | {timing(r['cpu'])} | {timing(r['gpu'])} |" if r['artifact_mb'] else
              f"| {r['model']} | {r['runtime']} | {r['precision']} | — (in memory) | {timing(r['cpu'])} | {timing(r['gpu'])} |")
md += ['', 'Mean ± sample SD over 45 calls per backend. CPU and GPU are separate sessions. '
       'GPU inputs/outputs were resident; transfer costs excluded. No measured CPU FP16 or GPU INT8 here.',
       '', '## KARD MoViNet-A0: 20 RGB frames at 172×172', '',
       '| Runtime | Precision | Device | File MB | Mean ms | Median ms | P95 ms |',
       '|---|---|---|---:|---:|---:|---:|']
for r in mov_rows:
    s = r['timing']
    md.append(f"| {r['runtime']} | {r['precision']} | {r['device']} | {r['artifact_mb']:.3f} | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} |")
md += ['', '90 calls per backend, 18 validation windows × five repeats. GPU timings include transfers and '
       'synchronous CPU output, unlike the ViT block. Only aggregate statistics were saved; SD is unavailable. '
       'The TensorFlow size is its full SavedModel directory; ONNX size is the graph file. FP16 keeps FP32 I/O.',
       '', '## Official MoViNet scales: separate CPU FP32 experiment', '',
       '| Model | Resolution | Full package MB | Weight-data MB | 8-frame clip ms | Retained-state single-frame ms |',
       '|---|---|---:|---:|---:|---:|']
for r in stock_rows:
    md.append(f"| {stock_names[r['model']]} | {r['resolution']}×{r['resolution']} | {r['artifact_mb']:.2f} | {r['variable_mb']:.2f} | {timing(r['cpu'])} | {timing(r['cpu_streaming'])} |")
md += ['', '30 calls after five warm-ups on the tutorial reference input. Single-frame streaming and fresh-state '
       'clip timings are different workloads. These models predict 600 Kinetics classes; KARD predicts 18.', '',
       '## Counting rules and missing evidence', '',
       '- Five ViT checkpoints × five tested forms = 25: PyTorch FP32/FP16 and ONNX FP32/FP16/mixed INT8.',
       '- Three stock MoViNet FP32 configurations = 3.',
       '- KARD MoViNet TensorFlow FP32 / ONNX FP32 / ONNX FP16 = 3.',
       '- Total 31 forms; only 26 serialized artifacts because PyTorch FP16 was in memory.',
       '- A1/A3/A4/A5 were not locally executed. No recorded TFLite, INT4, GPU INT8, static INT8/QAT, TensorRT or MoViNet INT8 benchmark.',
       '- B/C/D head ablations and LoRA/guided/random adapters have training results, not complete deployed size–latency results. Seeds, training fractions and sampling rates are not new quantization sizes.',
       '', '## Conversion agreement', '',
       '| ViT checkpoint | ONNX FP16 vs original | Original mixed INT8 export vs original |',
       '|---|---:|---:|']
for a in agreement_rows:
    md.append(f"| {a['model']} | {a['onnx_fp16']} | {a['onnx_int8']} |")
md += ['', 'Agreement on supplied demonstration windows is not held-out task accuracy. '
       'KARD MoViNet ONNX FP32 and FP16 each agree on 18/18 selected validation windows. '
       'The historical KARD head used in these latency tests is not the later strong baseline or adapter checkpoint.', '',
       '## Reuse in an article', '',
       '- `article_tables.tex`: three self-contained table* environments; requires booktabs. Copy this '
       'snippet into the manuscript folder and include it with `\\input{article_tables.tex}`.',
       '- `model_size_latency.csv`: exact byte counts, full-precision timing values, compression and speedups.',
       '- `size_latency_audit.json`: source/artifact SHA256, available historical-hash checks and counting rules.',
       '- `size_latency_tradeoff.pdf`, `.svg`, `.png`: vector/raster plots; distinct protocol panels.',
       '- Existing open `mindgraph_audit/latency_benchmark/latency_table.tex` updated in place; '
       'the preceding version is preserved as `latency_table_before_size_latency.tex`.', '',
       'Hardware: Xeon w7-2595X, 4 intra-/1 inter-op threads; RTX 4000 Ada 20 GB; Windows 11; '
       'PyTorch 2.13.0+cu130, ONNX Runtime GPU 1.31.0, TensorFlow 2.21.0. '
       'Size is decimal MB of serialized files, not runtime memory. '
       'Different sessions, frame counts, resolutions, class heads, graph formats and transfer policies '
       'prevent a controlled cross-family ranking. All timings exclude capture/display, decoding, loading and XAI.']
(OUT / 'README.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

# Plot within comparable protocols. Do not join points across unrelated blocks.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
fig.subplots_adjust(left=.065, right=.99, top=.86, bottom=.27, wspace=.38)
colors = {'FP32': '#386CB0', 'FP16': '#DD8452', 'Mixed INT8': '#42947B'}
for ax, device in zip(axes[:2], ('cpu', 'gpu')):
    for precision in ('FP32', 'FP16', 'Mixed INT8'):
        selected = [r for r in vit_rows if r['runtime'] == 'ONNX' and r['precision'] == precision and r[device]]
        if not selected:
            continue
        ax.errorbar([r['artifact_mb'] for r in selected], [r[device]['mean_ms'] for r in selected],
                    yerr=[r[device]['std_ms'] for r in selected], fmt='o', markersize=4,
                    capsize=2, color=colors[precision], alpha=.85, label=precision)
    ax.set_xlabel('Serialized ONNX size (MB)')
    ax.set_ylabel('Inference latency (ms)')
    ax.set_title(f"({'a' if device == 'cpu' else 'b'}) ViT–BiLSTM: {device.upper()}, 8 frames", loc='left', fontsize=9)
    ax.set_xlim(65, 385)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
ax = axes[2]
for r in mov_rows[1:]:
    s = r['timing']
    ax.scatter(r['artifact_mb'], s['mean_ms'], color=colors[r['precision']], s=30)
    ax.plot([r['artifact_mb']] * 2, [s['median_ms'], s['p95_ms']], color=colors[r['precision']], lw=1.3)
    ax.annotate(r['precision'], (r['artifact_mb'], s['mean_ms']), xytext=(5, -13),
                textcoords='offset points', fontsize=8)
ax.set_xlabel('Serialized ONNX size (MB)')
ax.set_ylabel('Inference latency (ms)')
ax.set_title('(c) KARD MoViNet: GPU, 20 frames', loc='left', fontsize=9)
ax.set_xlim(4, 10); ax.set_ylim(0, 80)
fig.text(.065, .075, 'Panels (a,b): five dataset checkpoints, mean ± call-level SD; GPU tensors resident. '
         'Models nearly overlap in file size.\nPanel (c): mean dots and median–P95 segments; includes transfers. '
         'Different protocols: no cross-panel speedup inference.', fontsize=7)
for extension in ('pdf', 'svg', 'png'):
    fig.savefig(OUT / f'size_latency_tradeoff.{extension}', dpi=250)
plt.close(fig)
print(json.dumps({'counts': counts, 'verified_artifacts': len(ARTIFACTS),
                  'latex_updated': str(OPEN_TEX), 'output': str(OUT)}, indent=2))
