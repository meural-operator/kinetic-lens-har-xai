"""Add verified sampling-quality results without inventing a latency sweep."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'movinet_evaluation/kard_sampling_cv'
SUMMARY = json.loads((SOURCE / 'summary.json').read_text())
RUNS = list(csv.DictReader((SOURCE / 'run_results.csv').open(newline='')))
AUDIT_PATH = ROOT / 'movinet_evaluation/kard_a0_head_100pct_10fps/sampling_reader_verification.json'
AUDIT = json.loads(AUDIT_PATH.read_text())
assert len(RUNS) == 30
assert {(int(r['fold']), int(r['seed']), int(r['fps'])) for r in RUNS} == {
    (fold, seed, fps) for fold in range(1, 6) for seed in (42, 123, 2026) for fps in (5, 10)}
assert AUDIT['all_requested_indices_real_and_distinct']
assert AUDIT['no_frame_repetition_or_padding']
assert AUDIT['same_window_start_for_every_clip']
assert AUDIT['all_5fps_indices_are_every_other_10fps_index']
rows = []
for rate in SUMMARY['rates']:
    fps = rate['fps']
    row = {'fps': fps, 'maximum_frames': fps * 10, 'maximum_context_seconds': 10,
           'backbone': 'Frozen official MoViNet A0-Stream', 'resolution': 172,
           'head_trainable_parameters': 8658, 'folds': 5, 'seeds_per_fold': 3,
           'inference_latency_ms': None}
    frames = [c[f'{fps}fps_frames'] for c in AUDIT['cases']]
    assert len(frames) == 540
    row.update(actual_frames_min=min(frames), actual_frames_median=statistics.median(frames),
               actual_frames_max=max(frames), actual_frames_mean=statistics.mean(frames))
    for metric in ('validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1'):
        fold_means = [100 * statistics.mean(float(r[metric]) for r in RUNS
                       if int(r['fps']) == fps and int(r['fold']) == fold) for fold in range(1, 6)]
        mean, sd = statistics.mean(fold_means), statistics.stdev(fold_means)
        assert abs(mean - rate[metric + '_mean_pct']) < 1e-10
        assert abs(sd - rate[metric + '_fold_sd_pp']) < 1e-10
        row[metric + '_mean_pct'] = mean
        row[metric + '_fold_sd_pp'] = sd
    rows.append(row)

def score(r, name):
    return f"${r[name + '_mean_pct']:.2f} \\pm {r[name + '_fold_sd_pp']:.2f}$"

table = [
    r'% Requires \usepackage{booktabs}. Standalone include snippet.',
    r'\begin{table*}[t]', r'\centering', r'\small',
    r'\caption{Temporal sampling density versus recognition quality for frozen MoViNet-A0-Stream with a freshly trained KARD-18 affine head. Both policies retain a centered maximum ten-second context and $172\times172$ RGB inputs. Results are percentages: mean $\pm$ sample SD across five subject-fold means after averaging three training seeds within each fold.}',
    r'\label{tab:kard-temporal-sampling}', r'\setlength{\tabcolsep}{4pt}',
    r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrrr@{}}', r'\toprule',
    r'FPS & Frame cap & Context (s) & Val.\ accuracy & Val.\ macro F1 & Test accuracy & Test macro F1 \\',
    r'\midrule']
for row in rows:
    table.append(f"{row['fps']} & {row['maximum_frames']} & $\\leq10$ & "
                 + ' & '.join(score(row, k) for k in
                   ('validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1'))
                 + r' \\')
table += [
    r'\bottomrule', r'\end{tabular*}', r'\par\smallskip',
    r'\begin{minipage}{\textwidth}\footnotesize',
    r'Five researcher-defined subject-held-out folds; seeds 42, 123 and 2026; 324/108/108 train/validation/test clips per fold. Each fold/rate uses fresh training-only normalization and a fresh 8,658-parameter head. Epoch and regularization choices use validation only. Thirty selected heads and 90 candidate fits were completed, with all selections locked before new-fold test scoring. Frame counts are upper limits, not fixed lengths; shorter videos contribute fewer distinct real frames without repeated-frame padding. Higher fps changes sampling density and the physical temporal receptive field, not parameter count. No inference latency was measured in this sampling study. This exploratory evaluation followed earlier observed KARD results; folds have overlapping training sets. SD is descriptive, not a confidence interval.',
    r'\end{minipage}', r'\end{table*}']
tex = '\n'.join(table) + '\n'
(OUT / 'table_temporal_sampling.tex').write_text(tex, encoding='utf-8')
with (OUT / 'temporal_sampling.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
provenance = {
    'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (SOURCE / 'summary.json', SOURCE / 'run_results.csv', AUDIT_PATH)},
    'fold_statistics_independently_recomputed_from_30_run_rows': True,
    'new_training_or_inference_benchmark_performed': False,
    'latency_for_5_vs_10fps': 'Not measured',
    'no_same_model_precision_checkpoint_fixed_length_latency_sweep_available': True,
    'actual_frame_statistics_scope': 'All 540 clips; same raw-feature inputs reused by the fold study.',
    'rows': rows}
(OUT / 'temporal_sampling_audit.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

md = [
    '# Temporal input sampling: recognition quality and benchmark coverage', '',
    'The 5/10-fps comparison preserves a maximum ten-second context and changes both sampling density '
    'and frame cap. It does not hold the number of frames fixed.', '',
    '| FPS | Frame cap | Maximum context | Validation accuracy (%) | Validation macro F1 (%) | Test accuracy (%) | Test macro F1 (%) |',
    '|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    values = [f"{row[k + '_mean_pct']:.2f} ± {row[k + '_fold_sd_pp']:.2f}"
              for k in ('validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1')]
    md.append(f"| {row['fps']} | {row['maximum_frames']} | ≤10 s | " + ' | '.join(values) + ' |')
md += [
    '', 'Mean ± sample SD across five fold means, with three seeds averaged within each fold. '
    'The 30 rate/fold/seed combinations used 90 L2 candidate fits. All selections preceded new-fold test scoring. '
    'These are exploratory, researcher-defined subject folds, not a new blind external benchmark.', '',
    'Ten fps changes mean test accuracy by −2.35 percentage points and macro F1 by −1.99 points. '
    'It wins macro F1 in 4/15 paired fold/seed runs and 1/5 seed-averaged folds. '
    'This does not prove that 5 fps is universally optimal.', '',
    '## Actual input lengths', '',
    '| FPS | Minimum real frames | Median real frames | Maximum real frames | Mean real frames |',
    '|---:|---:|---:|---:|---:|']
for row in rows:
    md.append(f"| {row['fps']} | {row['actual_frames_min']} | {row['actual_frames_median']:g} | "
              f"{row['actual_frames_max']} | {row['actual_frames_mean']:.2f} |")
md += [
    '', 'Statistics cover all 540 KARD clips. A cap of 50/100 does not mean every clip contained that many frames. '
    'No frame repetition or padding was used; the source timestamp audit verifies real distinct samples.', '',
    '## What the existing timing records can and cannot say', '',
    '- Official pretrained MoViNets have retained-state **one-frame** and fresh-state **eight-frame** CPU timing records. '
    'The state policies differ, so this is not a pure fixed-length scaling experiment.',
    '- ViT–BiLSTM precision benchmarks use **eight-frame** windows.',
    '- Our historical trained KARD MoViNet CPU/CUDA benchmarks use **20-frame** windows at 5 fps (four seconds of context).',
    '- The 5/10-fps recognition study did **not** benchmark latency. Its missing timings must remain unreported.',
    '- We have not recorded a matched sweep of one unchanged checkpoint/runtime/precision at, for example, 4/8/16/20/32 frames. '
    'Existing 1/8/20-frame results cannot be pooled to infer that curve.',
    '- Sampling density and input length affect activation/computation workload; they do not create new weight-precision or parameter-count variants.',
    '', 'For a paper, use the verified sampling-quality table separately from the size/latency tables. '
    'A future frame-count trade-off experiment must specify whether it varies context duration at fixed fps, '
    'or sampling density at fixed context, and measure accuracy and latency under that same protocol.']
(OUT / 'temporal_sampling.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

# Keep additions idempotent and preserve the existing open document.
begin, end = '% BEGIN TEMPORAL SAMPLING ADDITION', '% END TEMPORAL SAMPLING ADDITION'
for target in (OUT / 'article_tables.tex', ROOT / 'mindgraph_audit/latency_benchmark/latency_table.tex'):
    content = target.read_text(encoding='utf-8')
    if begin in content:
        before, rest = content.split(begin, 1)
        _, after = rest.split(end, 1)
        content = before.rstrip() + after
    if target.name == 'article_tables.tex':
        content = content.rstrip() + '\n\n' + begin + '\n' + tex + end + '\n'
    else:
        section = (begin + '\n\\clearpage\n' + tex + '\n'
                   r'\textbf{Input length is not sampling rate.} '
                   r'Changing fps at fixed ten-second context changes both input density and frame count; '
                   r'changing frame count at fixed fps changes context duration. The 50/100 values above '
                   r'are caps rather than fixed tensor lengths. '
                   r'Our earlier one-frame retained-state, eight-frame clip and twenty-frame KARD timings '
                   r'used different state policies, checkpoints or protocols; they do not form a controlled '
                   r'latency-versus-frame-count curve. No 5-versus-10-fps latency was measured.'
                   '\n' + end + '\n')
        assert r'\end{document}' in content
        content = content.replace(r'\end{document}', section + r'\end{document}')
    target.write_text(content, encoding='utf-8')
readme = OUT / 'README.md'
text = readme.read_text(encoding='utf-8')
line = ('\n\nTemporal sampling follow-up: `table_temporal_sampling.tex`, `temporal_sampling.csv` '
        'and `temporal_sampling.md` report verified 5/10-fps recognition results and distinguish them '
        'from the missing matched frame-count latency sweep.\n')
if 'Temporal sampling follow-up:' not in text:
    readme.write_text(text + line, encoding='utf-8')
print(json.dumps({'sampling_rows': rows, 'new_latencies_measured': False,
                  'article_bundle_updated': True}, indent=2))
