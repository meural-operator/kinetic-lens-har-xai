"""Generate evidence-led publication figures and report from locked runs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def report(out, gate, evaluation, labels, parameters, names, seeds):
    out = Path(out)
    figure_dir = out/'figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'semibold', 'figure.dpi': 160,
        'savefig.dpi': 320, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ['#64748b', '#168c89', '#c76530', '#7664a5']
    rows, summary, class_rows, delta_rows = [], {}, [], []
    variants = list(names)
    for variant in variants:
        summary[variant] = {'parameters': parameters[variant], 'validation': {}, 'test': {}}
        for seed in seeds:
            key = f'{variant}_seed_{seed}'
            selected = gate['selections'][key]
            test = evaluation[key]['test']
            val = selected['validation']
            rows.append([variant, names[variant], seed, parameters[variant], selected['l2'], selected['best_epoch'],
                val['accuracy'], val['macro_f1'], val['cross_entropy'], test['accuracy'], test['macro_f1'], test['cross_entropy']])
            for split, metrics in [('validation', val), ('test', test)]:
                for i, label in enumerate(labels):
                    class_rows.append([variant, seed, split, i, label, metrics['recall_per_class'][i],
                                       metrics['f1_per_class'][i], metrics['support_per_class'][i]])
                write_csv(out/'evaluation'/key/f'{split}_confusion.csv', ['true_class']+labels,
                          [[label]+metrics['confusion_matrix'][i] for i, label in enumerate(labels)])
        for split in ('validation', 'test'):
            source = [gate['selections'][f'{variant}_seed_{s}']['validation'] if split == 'validation'
                      else evaluation[f'{variant}_seed_{s}']['test'] for s in seeds]
            for metric in ('accuracy', 'macro_f1', 'cross_entropy'):
                values = [item[metric] for item in source]
                summary[variant][split][metric] = {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)), 'per_seed': values}
            recall = np.asarray([item['recall_per_class'] for item in source])
            mean_cm = np.mean([item['confusion_matrix'] for item in source], axis=0)
            summary[variant][split]['recall_per_class_mean'] = recall.mean(0).tolist()
            summary[variant][split]['recall_per_class_sample_sd'] = recall.std(0, ddof=1).tolist()
            summary[variant][split]['confusion_matrix_mean_over_runs'] = mean_cm.tolist()
            write_csv(out/f'{variant}_{split}_mean_confusion.csv', ['true_class']+labels,
                      [[label]+mean_cm[i].tolist() for i, label in enumerate(labels)])
    for seed in seeds:
        for split in ('validation', 'test'):
            b = gate['selections'][f'B_seed_{seed}']['validation'] if split == 'validation' else evaluation[f'B_seed_{seed}']['test']
            c = gate['selections'][f'C_seed_{seed}']['validation'] if split == 'validation' else evaluation[f'C_seed_{seed}']['test']
            delta_rows.append([seed, split, c['accuracy']-b['accuracy'], c['macro_f1']-b['macro_f1'], c['cross_entropy']-b['cross_entropy']])
    write_csv(out/'per_seed_metrics.csv', ['variant','model','seed','changed_part_parameters','selected_l2','best_epoch',
        'validation_accuracy','validation_macro_f1','validation_cross_entropy','test_accuracy','test_macro_f1','test_cross_entropy'], rows)
    write_csv(out/'per_class_per_seed.csv', ['variant','seed','split','class_id','class','recall','f1','support'], class_rows)
    write_csv(out/'paired_C_minus_B.csv', ['seed','split','accuracy_delta','macro_f1_delta','cross_entropy_delta'], delta_rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    class_summary_rows = []
    for variant in variants:
        for split in ('validation', 'test'):
            for i, label in enumerate(labels):
                class_summary_rows.append([variant, split, i, label, summary[variant][split]['recall_per_class_mean'][i],
                                           summary[variant][split]['recall_per_class_sample_sd'][i]])
    write_csv(out/'per_class_summary.csv', ['variant','split','class_id','class','recall_mean','recall_sample_sd'], class_summary_rows)
    summary_rows = [[v, split, metric, values['mean'], values['sample_sd']] for v in variants
                    for split in ('validation','test') for metric, values in summary[v][split].items()
                    if metric in ('accuracy','macro_f1','cross_entropy')]
    write_csv(out/'summary_metrics.csv', ['variant','split','metric','mean','sample_sd'], summary_rows)

    def save(fig, name):
        for suffix in ('pdf', 'svg', 'png'):
            fig.savefig(figure_dir/f'{name}.{suffix}', bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.4), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        for i, variant in enumerate(variants):
            values = summary[variant][split]['macro_f1']
            ax.bar(i, 100*values['mean'], color=colors[i], alpha=.78, width=.62)
            ax.errorbar(i, 100*values['mean'], yerr=100*values['sample_sd'], color='#263344', capsize=4, linewidth=1.3)
            ax.scatter(i+np.array([-.12,0,.12]), 100*np.asarray(values['per_seed']), s=23, c='#17212c', zorder=3)
        ax.set_xticks(range(4), ['A\n480 affine','B\n104 mean','C\n104 temporal','D\n480 MLP'])
        ax.set_ylim(0,100); ax.set_ylabel('Macro F1 (%)'); ax.set_title(split.capitalize())
        ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
    fig.suptitle('KARD: three training seeds, fixed subject split', fontsize=11)
    save(fig, 'macro_f1_comparison')

    fig, axes = plt.subplots(1,2,figsize=(7.6,3.2),layout='constrained')
    for ax, split in zip(axes, ('validation','test')):
        deltas = [100*row[3] for row in delta_rows if row[1] == split]
        ax.axhline(0,color='#64748b',linewidth=1)
        ax.scatter(range(3), deltas, s=48, color=colors[2], zorder=3)
        ax.plot(range(3), deltas, color=colors[2],alpha=.4)
        ax.set_xticks(range(3), [str(s) for s in seeds]); ax.set_xlabel('Training seed')
        ax.set_ylabel('C − B macro F1 (percentage points)'); ax.set_title(split.capitalize()); ax.grid(axis='y',alpha=.16)
    save(fig,'paired_temporal_delta')

    for split in ('validation','test'):
        fig, ax = plt.subplots(figsize=(9.2,5.4),layout='constrained')
        y = np.arange(18)
        for i, variant in enumerate(variants):
            means = np.asarray(summary[variant][split]['recall_per_class_mean'])*100
            sd = np.asarray(summary[variant][split]['recall_per_class_sample_sd'])*100
            ax.errorbar(means, y+(i-1.5)*.16, xerr=sd, fmt='o', markersize=3.4, linewidth=.85,
                        capsize=2, color=colors[i], label=variant+' '+names[variant])
        ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(-3,103)
        ax.set_xlabel('Recall (%), mean ± sample SD across training seeds')
        ax.set_title(f'{split.capitalize()}: all 18 classes'); ax.grid(axis='x',alpha=.16)
        ax.legend(fontsize=7,loc='lower right')
        save(fig,f'{split}_class_recall')

        fig, axes = plt.subplots(2,2,figsize=(10.4,9.4),layout='constrained')
        for ax, variant in zip(axes.flat,variants):
            cm = np.asarray(summary[variant][split]['confusion_matrix_mean_over_runs'])
            ax.imshow(cm,cmap='Blues',vmin=0,vmax=6)
            for i in range(18):
                for j in range(18):
                    if cm[i,j] > 0:
                        ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',fontsize=5.5,color='white' if cm[i,j]>3.4 else '#18344b')
            ax.set_xticks(range(18),range(1,19),fontsize=6)
            ax.set_yticks(range(18),range(1,19),fontsize=6)
            ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(variant+' '+names[variant],fontsize=9)
        fig.suptitle(f'{split.capitalize()} confusion: elementwise mean over 3 runs; 108 unique clips',fontsize=11)
        save(fig,f'{split}_mean_confusions')
        # Per-seed figures preserve run-specific errors, not only averages.
        for seed in seeds:
            fig, axes = plt.subplots(2,2,figsize=(9.4,8.3),layout='constrained')
            for ax,variant in zip(axes.flat,variants):
                key = f'{variant}_seed_{seed}'
                item = gate['selections'][key]['validation'] if split == 'validation' else evaluation[key]['test']
                cm = np.asarray(item['confusion_matrix'])
                ax.imshow(cm,cmap='Blues',vmin=0,vmax=6)
                for i,j in zip(*np.where(cm>0)):
                    ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if cm[i,j]>3 else '#18344b')
                ax.set_xticks(range(18),range(1,19),fontsize=6); ax.set_yticks(range(18),range(1,19),fontsize=6)
                ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(variant+' '+names[variant],fontsize=9)
            fig.suptitle(f'{split.capitalize()} confusion, training seed {seed}; 108 clips',fontsize=11)
            save(fig,f'{split}_seed_{seed}_confusions')

    base = out.parent/'kard_a0_head_100pct'
    historical = {'selection': json.loads((base/'selection.json').read_text()),
                  'evaluation': json.loads((base/'evaluation.json').read_text()),
                  'role': 'Historical TensorFlow CPU result, contextual only; no new test feature access.'}
    (out/'historical_context.json').write_text(json.dumps(historical,indent=2),encoding='utf-8')
    table = ['| Variant | New trainable parameters | Validation macro F1 | Test macro F1 | Test accuracy |',
             '|---|---:|---:|---:|---:|']
    fmt = lambda value: f"{100*value['mean']:.2f} ± {100*value['sample_sd']:.2f}"
    for variant in variants:
        table.append(f"| {variant}: {names[variant]} | {parameters[variant]:,} | {fmt(summary[variant]['validation']['macro_f1'])} | {fmt(summary[variant]['test']['macro_f1'])} | {fmt(summary[variant]['test']['accuracy'])} |")
    deltas = np.asarray([row[3] for row in delta_rows if row[1]=='test'])*100
    direction = 'positive for all three seeds' if np.all(deltas>0) else 'negative for all three seeds' if np.all(deltas<0) else 'mixed or tied across seeds'
    hv = historical['selection']['winner']['validation']; ht = historical['evaluation']['test']
    perseed = ['| Variant | Seed | L2 | Best epoch | Validation macro F1 | Test macro F1 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        perseed.append(f'| {row[0]} | {row[2]} | {row[4]:g} | {row[5]} | {row[7]:.2%} | {row[10]:.2%} |')
    report_text = f'''# KARD controlled architecture ablations

Four prespecified variants were trained with three prespecified seeds (42, 123,
2026). Every variant/seed received the same three-candidate L2 tuning budget.
All primary runs use PyTorch CUDA on the same frozen cached representations.
The main controlled comparison is B versus C: the causal temporal convolution
is their only architectural difference. Neither model uses attention.

{chr(10).join(table)}

Values are percentages, mean ± sample standard deviation across **training
seeds**, not population confidence intervals. Each seed uses the same 108 test
clips. This split was observed in earlier studies, so test results are
descriptive and are not a newly blind benchmark.

The paired test macro F1 difference C−B is {deltas.mean():+.2f} ± {deltas.std(ddof=1):.2f}
percentage points and is {direction}. Per-seed differences are
{', '.join(f'{s}: {d:+.2f}' for s,d in zip(seeds,deltas))} percentage points.
This small experiment supports a conditional comparison on this fixed split;
three seeds do not establish definitive superiority or subject generalization.
Negative and mixed results are retained without changing the protocol.

{chr(10).join(perseed)}

## Architecture and experimental controls

A is an original-style standardized480D affine PyTorch control with dropout.25.
B uses equal-clip train-only standardized104D per-frame features, masked mean,
Linear104→128, LayerNorm, hard-swish, dropout.25 and Linear128→18.
C inserts a causal depthwise Conv1d104→104,k3 with hard-swish residual before
masked mean. B/C common initial weights and normalization are bitwise identical
per seed, verified in preflight.json. C's temporal weights use an isolated RNG.
D uses the original pooled480D endpoint, Linear480→32, LayerNorm, hard-swish,
dropout.25 and Linear32→18. D approximately matches the changed-part parameter
budget; it retains the original frozen104→480 projection and does not match
B/C full-model compute. Original stem and all15 bottlenecks remain frozen.

Adam uses lr.001, batch16, global gradient norm clip1, maximum200 epochs,
patience25. Loss adds explicit kernel L2 (bias and LayerNorm excluded).
The L2 grid is .0001,.001,.01; checkpoints and L2 are selected independently
per variant/seed by validation macro F1, then lower validation cross entropy.
No best seed is selected. All candidates share deterministic per-seed shuffled
minibatches and reset dropout streams. Atomic epoch checkpoints include model,
optimizer, all RNG, history, best weights and early-stop state. Exact short
continuation equivalence, gradients, updates, reload, padding and causality
were checked on real training caches.

Input sampling is5fps, resize with padding172×172, maximum50 frames, centered
10-second window when needed. Training subjects1,3,4,6,7,8 (324 clips),
validation5,10 (108), test2,9 (108); all18 labels are retained.
480D normalization was independently reconstructed and compared bitwise to
the historical training-only reference.104D normalization uses equal clip
weights and also matches the saved reference.

All12 selections and their checkpoint hashes were persisted before any new
test cache reads. Only selected checkpoints were evaluated. No post-test
tuning occurred. Cache inventories verify video/model/manifest provenance,
stored104D cache/source hashes, frame agreement and finite array shapes.
Historical480D NPZ sidecars do not contain stored cache checksums; independently
created SHA256 inventories verify immutability from this audit onward and
cannot establish unavailable historical checksums retrospectively.

## Historical reference and limits

The historical TensorFlow CPU affine baseline had validation macro F1
{hv['macro_f1']:.2%}, test macro F1 {ht['macro_f1']:.2%}, and test accuracy
{ht['accuracy']:.2%}. It is contextual. Fresh A is not an exact reproduction:
PyTorch initialization, Adam defaults and gradient clipping differ.
See historical_context.json for the original result and selection.

Seed variation measures training randomness, not variability across new
subjects. Two subjects in each held-out partition and a previously observed
split constrain inference. The main B/C pair isolates the added temporal
module within this prespecified training recipe; comparisons to A/D also
change representation and/or head capacity. No whole-model latency,
end-to-end speed, artifact-size, novelty or generic-superiority claim is made.

## Outputs

per_seed_metrics.csv, summary_metrics.csv, paired_C_minus_B.csv,
per_class_per_seed.csv and per_class_summary.csv preserve numerical results.
Per-seed confusion CSVs and plots retain each run. Aggregate confusion
matrices are **elementwise mean counts across runs**, with fractional cells;
they are not pooled324 independent clips. All18 classes map to indices1–18
in plots using the ordering in labels.json. Figures are supplied as PDF, SVG
and320dpi PNG. Histories and candidate/selected checkpoints are retained.
Use figures_for_paper.tex as a snippet; existing paper files are untouched.
'''
    (out/'REPORT.md').write_text(report_text,encoding='utf-8')
    snippet = r'''% Copy figures and this snippet into your paper; requires graphicx.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/macro_f1_comparison.pdf}
  \caption{KARD validation and descriptive test macro F1 for four prespecified
  frozen-feature controls. Points show three training seeds; error bars show
  sample standard deviation across training seeds on the same fixed subject
  split, not population confidence intervals.}
  \label{fig:kard-controlled-ablation}
\end{figure}
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/paired_temporal_delta.pdf}
  \caption{Paired per-seed macro F1 changes when adding the causal depthwise
  temporal convolution to the otherwise identical compact model (C minus B).
  The previously observed test split is descriptive.}
  \label{fig:kard-temporal-paired}
\end{figure}
'''
    (out/'figures_for_paper.tex').write_text(snippet,encoding='utf-8')


if __name__ == '__main__':
    from run_kard_ablation import OUT, LABELS, PARAMETERS, NAMES, SEEDS, validate_selection_gate
    report(OUT, validate_selection_gate(), json.loads((OUT/'evaluation.json').read_text()), LABELS, PARAMETERS, NAMES, SEEDS)
