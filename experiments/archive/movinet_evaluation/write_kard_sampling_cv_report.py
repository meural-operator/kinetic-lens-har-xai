"""Audit paired cross-subject runs and create manuscript-ready descriptive assets."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'kard_sampling_cv'
ASSETS = OUT/'paper_assets'
ASSETS.mkdir(exist_ok=True)
TMP = OUT/'tmp'
TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TMP)
os.environ['MPLCONFIGDIR'] = str(TMP/'matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pymupdf
from pypdf import PdfReader, PdfWriter

def read(path):
    return json.loads(path.read_text())

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def simple_metrics(matrix):
    matrix = np.asarray(matrix)
    tp = np.diag(matrix)
    den = matrix.sum(0)+matrix.sum(1)
    return float(tp.sum()/matrix.sum()), float(np.divide(2*tp,den,out=np.zeros(18,float),where=den>0).mean())

protocol = read(OUT/'protocol.json')
lock = read(OUT/'selection_lock.json')
execution = read(OUT/'execution.json')
assert execution['runs_completed'] == 30 and execution['candidate_fits'] == 90
assert lock['all_selections_completed_before_new_test_scoring'] and not lock['test_used_for_selection']
assert lock['protocol_sha256'] == digest(OUT/'protocol.json')
runs = {}
provenance = []
feature_provenance = read(OUT/'feature_provenance.json')
feature_arrays = {}
for fps in [5,10]:
    data = []
    for item in feature_provenance['records']:
        if item['fps'] != fps:
            continue
        path = ROOT/item['cache']
        assert digest(path) == item['cache_sha256']
        with np.load(path) as values:
            data.append(values['features'].copy())
    feature_arrays[fps] = np.stack(data)
    assert feature_arrays[fps].shape == (540,480)
for item in lock['heads']:
    folder = OUT/item['folder']
    assert digest(folder/'selection.json') == item['selection_sha256']
    assert digest(folder/'kard_head.keras') == item['head_sha256']
    s = read(folder/'selection.json'); e = read(folder/'evaluation.json')
    assert not s['test_used_for_training_or_selection'] and not e['test_used_for_selection']
    assert e['selection_lock_sha256'] == digest(OUT/'selection_lock.json')
    assert s['head_reload_max_logit_error'] < 1e-6
    assert len(s['candidates']) == 3 and s['normalization_fit_clips'] == 324
    assert all(c['maximum_trainable_weight_change'] > 1e-5 for c in s['candidates'])
    key = tuple(s['context'][k] for k in ['fold','seed','fps'])
    c = s['context']; subjects = c['subjects']
    train_ids = protocol['folds'][c['fold']-1]['record_indices']['train']
    train_features = feature_arrays[c['fps']][train_ids]
    with np.load(folder/'train_normalization.npz') as norm:
        assert np.array_equal(norm['mean'],train_features.mean(axis=0))
        assert np.array_equal(norm['scale'],np.maximum(train_features.std(axis=0),1e-3))
    assert len(set(subjects['train']) | set(subjects['validation']) | set(subjects['test'])) == 10
    assert set(subjects['train']).isdisjoint(subjects['test']) and set(subjects['train']).isdisjoint(subjects['validation'])
    assert set(subjects['validation']).isdisjoint(subjects['test'])
    winner = max(s['candidates'], key=lambda v:(v['validation']['macro_f1'],-v['validation']['cross_entropy']))
    assert winner == s['winner']
    for candidate in s['candidates']:
        history = read(folder/f"history_l2_{candidate['l2']:g}.json")
        best = max(history,key=lambda v:(v['validation_macro_f1'],-v['validation_cross_entropy']))
        assert best['epoch'] == candidate['best_epoch']
        assert len(history) == candidate['epochs_executed']
        assert candidate['optimizer_steps'] == candidate['epochs_executed']*21
        assert np.isclose(best['validation_macro_f1'],candidate['validation']['macro_f1'])
        assert np.isclose(best['validation_cross_entropy'],candidate['validation']['cross_entropy'])
    matrix = np.zeros((18,18), int)
    for p in e['predictions']:
        assert p['subject'] in subjects['test']
        matrix[p['true_class'],p['predicted_class']] += 1
    assert np.array_equal(matrix, e['test']['confusion_matrix']) and np.all(matrix.sum(1)==6)
    a, f = simple_metrics(matrix)
    assert np.isclose(a,e['test']['accuracy']) and np.isclose(f,e['test']['macro_f1'])
    runs[key] = {'selection':s,'evaluation':e}
    for name in ['selection.json','evaluation.json','kard_head.keras','train_normalization.npz','test_logits.npz']:
        path = folder/name
        provenance.append({'path':str(path.relative_to(ROOT)), 'sha256':digest(path)})
assert len(runs)==30

fold_rows = []
subject_rows = []
seed_rows = []
paired = []
action_values = {(fps,label):[] for fps in [5,10] for label in range(18)}
for fold in range(1,6):
    for fps in [5,10]:
        matches = [runs[(fold,seed,fps)] for seed in protocol['training_seeds']]
        row = {'fold':fold,'fps':fps}
        for split, prefix in [('validation','validation'),('test','test')]:
            for name in ['accuracy','macro_f1']:
                values = [100*(r['selection']['winner'][split][name] if split=='validation' else r['evaluation']['test'][name]) for r in matches]
                row[prefix+'_'+name+'_mean_pct'] = float(np.mean(values))
                row[prefix+'_'+name+'_seed_sd_pp'] = float(np.std(values,ddof=1))
        fold_rows.append(row)
    for seed in protocol['training_seeds']:
        r5, r10 = runs[(fold,seed,5)], runs[(fold,seed,10)]
        assert r5['selection']['winner']['initial_trainable_sha256'] == r10['selection']['winner']['initial_trainable_sha256']
        assert [p['video'] for p in r5['evaluation']['predictions']] == [p['video'] for p in r10['evaluation']['predictions']]
        paired.append({'fold':fold, 'seed':seed,
            'validation_macro_f1_change_pp':100*(r10['selection']['winner']['validation']['macro_f1']-r5['selection']['winner']['validation']['macro_f1']),
            'test_accuracy_change_pp':100*(r10['evaluation']['test']['accuracy']-r5['evaluation']['test']['accuracy']),
            'test_macro_f1_change_pp':100*(r10['evaluation']['test']['macro_f1']-r5['evaluation']['test']['macro_f1'])})
for subject in range(1,11):
    fold = next(f['fold'] for f in protocol['folds'] if subject in f['subjects']['test'])
    for fps in [5,10]:
        values = []
        for seed in protocol['training_seeds']:
            pred = [p for p in runs[(fold,seed,fps)]['evaluation']['predictions'] if p['subject']==subject]
            assert len(pred)==54
            values.append(100*np.mean([p['true_class']==p['predicted_class'] for p in pred]))
        subject_rows.append({'subject':subject,'fold':fold,'fps':fps,'clips_per_seed':54,
            'test_accuracy_mean_pct':float(np.mean(values)), 'test_accuracy_seed_sd_pp':float(np.std(values,ddof=1))})
for seed in protocol['training_seeds']:
    for fps in [5,10]:
        all_predictions = [p for fold in range(1,6) for p in runs[(fold,seed,fps)]['evaluation']['predictions']]
        assert len(all_predictions)==540 and len({p['video'] for p in all_predictions})==540
        matrix = np.zeros((18,18), int)
        for p in all_predictions:
            matrix[p['true_class'],p['predicted_class']] += 1
        a,f = simple_metrics(matrix)
        seed_rows.append({'seed':seed,'fps':fps,'out_of_fold_clips':540,'out_of_fold_accuracy_pct':100*a,'out_of_fold_macro_f1_pct':100*f})
        assert np.all(matrix.sum(1)==30)
        for label in range(18):
            action_values[(fps,label)].append({'recall':100*matrix[label,label]/matrix[label].sum(),
                'f1':100*2*matrix[label,label]/(matrix[label].sum()+matrix[:,label].sum())})

action_rows = []
for label,name in enumerate(protocol['labels']):
    for fps in [5,10]:
        v = action_values[(fps,label)]
        action_rows.append({'class_id':label,'action':name,'fps':fps,'out_of_fold_clips_per_seed':30,
            'recall_mean_pct':float(np.mean([a['recall'] for a in v])),
            'recall_seed_sd_pp':float(np.std([a['recall'] for a in v],ddof=1)),
            'f1_mean_pct':float(np.mean([a['f1'] for a in v])),
            'f1_seed_sd_pp':float(np.std([a['f1'] for a in v],ddof=1))})

summary_rows = []
for fps in [5,10]:
    rows = [r for r in fold_rows if r['fps']==fps]
    result = {'fps':fps,'folds':5,'seeds_per_fold':3}
    for name in ['validation_accuracy','validation_macro_f1','test_accuracy','test_macro_f1']:
        values = [r[name+'_mean_pct'] for r in rows]
        result[name+'_mean_pct'] = float(np.mean(values))
        result[name+'_fold_sd_pp'] = float(np.std(values,ddof=1))
    summary_rows.append(result)
choices = lock['sampling_choices']
selected_fold_scores = []
for fold in range(1,6):
    chosen = [runs[(fold,c['seed'],c['validation_recommended_fps'])]['evaluation']['test'] for c in choices if c['fold']==fold]
    selected_fold_scores.append({'fold':fold, 'test_accuracy_pct':100*float(np.mean([c['accuracy'] for c in chosen])),
                                'test_macro_f1_pct':100*float(np.mean([c['macro_f1'] for c in chosen]))})
summary = {'rates':summary_rows, 'mean_definition':'Mean over five folds after averaging three seeds within each fold.',
    'sd_definition':'Sample SD across the five fold means; descriptive, not confidence intervals.',
    'test_accuracy_change_10_minus_5_pp':summary_rows[1]['test_accuracy_mean_pct']-summary_rows[0]['test_accuracy_mean_pct'],
    'test_macro_f1_change_10_minus_5_pp':summary_rows[1]['test_macro_f1_mean_pct']-summary_rows[0]['test_macro_f1_mean_pct'],
    'paired_run_macro_f1_wins_10fps':sum(p['test_macro_f1_change_pp']>1e-10 for p in paired),
    'paired_run_macro_f1_ties':sum(abs(p['test_macro_f1_change_pp'])<=1e-10 for p in paired),
    'fold_mean_macro_f1_wins_10fps':sum(fold_rows[2*j+1]['test_macro_f1_mean_pct']>fold_rows[2*j]['test_macro_f1_mean_pct']+1e-10 for j in range(5)),
    'validation_choices_10fps':sum(c['validation_recommended_fps']==10 for c in choices),
    'validation_choices_5fps':sum(c['validation_recommended_fps']==5 for c in choices),
    'validation_selected_procedure':{name+'_mean_pct':float(np.mean([r[name+'_pct'] for r in selected_fold_scores])) for name in ['test_accuracy','test_macro_f1']},
    'prior_results_observed':True,'new_test_used_for_checkpoint_or_sampling_selection':False,
    'no_post_test_tuning':True,'no_inferential_significance_claim':True}
write(OUT/'summary.json',summary)
for name,rows in [('fold_results.csv',fold_rows),('subject_results.csv',subject_rows),('seed_results.csv',seed_rows),
                  ('paired_changes.csv',paired),('summary_results.csv',summary_rows)]:
    save_csv(OUT/name,rows)
save_csv(OUT/'action_results.csv',action_rows)

# Editable vector figures at standard single/double column dimensions.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':0.8,'savefig.facecolor':'white'})
colors = {5:'#0072B2',10:'#D55E00'}
figures = []
def save_figure(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(ASSETS/(name+'.'+ext),dpi=600)
    figures.append(name)
    doc = pymupdf.open(ASSETS/(name+'.pdf'))
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(TMP/(name+'_pdf_render.png'))
    doc.close(); plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(6.75,2.65))
for ax,name,title in zip(axes,['test_accuracy','test_macro_f1'],['(a) Test accuracy','(b) Test macro F1']):
    for fps,offset,marker in [(5,-.075,'o'),(10,.075,'s')]:
        rows = [r for r in fold_rows if r['fps']==fps]
        ax.errorbar(np.arange(1,6)+offset,[r[name+'_mean_pct'] for r in rows],
            yerr=[r[name+'_seed_sd_pp'] for r in rows],fmt=marker,color=colors[fps],capsize=3,
            markersize=4,label=f'{fps} fps',elinewidth=1)
    for j in range(5):
        ax.plot([j+1-.075,j+1+.075],[fold_rows[2*j][name+'_mean_pct'],fold_rows[2*j+1][name+'_mean_pct']],
                color='#AAB7C4',lw=.8,zorder=0)
    low = min(r[name+'_mean_pct']-r[name+'_seed_sd_pp'] for r in fold_rows)
    high = max(r[name+'_mean_pct']+r[name+'_seed_sd_pp'] for r in fold_rows)
    ax.set(xticks=range(1,6),xlabel='Subject fold',ylabel='Score (%)',title=title,
           ylim=(max(0,5*np.floor((low-5)/5)),max(102,high+3)))
    ax.grid(axis='y',color='#DCE3EA',lw=.6); ax.set_axisbelow(True)
axes[0].legend(loc='lower left',frameon=False)
fig.tight_layout(pad=1)
save_figure(fig,'cv01_fold_comparison')

fig,ax = plt.subplots(figsize=(3.25,3.45))
for subject in range(1,11):
    rows = [r for r in subject_rows if r['subject']==subject]
    ax.plot([r['test_accuracy_mean_pct'] for r in rows],[subject]*2,color='#BCC9D4',lw=.9)
for fps,marker,offset in [(5,'o',-.06),(10,'s',.06)]:
    rows = [r for r in subject_rows if r['fps']==fps]
    ax.errorbar([r['test_accuracy_mean_pct'] for r in rows],np.arange(1,11)+offset,
        xerr=[r['test_accuracy_seed_sd_pp'] for r in rows],fmt=marker,color=colors[fps],capsize=2,
        markersize=4,label=f'{fps} fps',elinewidth=.8)
ax.set(yticks=range(1,11),yticklabels=[f'S{i:02}' for i in range(1,11)],
       xlabel='Test accuracy (%)',ylabel='Held-out subject',
       xlim=(max(0,5*np.floor(min(r['test_accuracy_mean_pct']-r['test_accuracy_seed_sd_pp'] for r in subject_rows)/5)-5),
             max(102,max(r['test_accuracy_mean_pct']+r['test_accuracy_seed_sd_pp'] for r in subject_rows)+3)))
ax.invert_yaxis(); ax.grid(axis='x',color='#DCE3EA',lw=.6); ax.set_axisbelow(True)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,ncol=2)
fig.tight_layout(pad=.9)
save_figure(fig,'cv02_subject_comparison')

fig,ax = plt.subplots(figsize=(6.75,2.4))
assignment = np.zeros((5,10),int)
for fold in protocol['folds']:
    for split,value in [('validation',1),('test',2)]:
        for subject in fold['subjects'][split]: assignment[fold['fold']-1,subject-1]=value
ax.imshow(assignment,cmap=ListedColormap(['#E4EBF0','#E69F00','#0072B2']),vmin=0,vmax=2,aspect='auto')
ax.set(xticks=np.arange(10),xticklabels=[f'S{i:02}' for i in range(1,11)],yticks=np.arange(5),
       yticklabels=[f'Fold {i}' for i in range(1,6)],xlabel='Subject')
ax.set_xticks(np.arange(-.5,10,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color='white',lw=2); ax.tick_params(which='minor',length=0)
for i in range(5):
    for j in range(10):
        ax.text(j,i,['Tr','V','Te'][assignment[i,j]],ha='center',va='center',fontsize=8,
                color='white' if assignment[i,j]==2 else '#24313A')
ax.legend(handles=[Patch(color=c,label=l) for c,l in [('#E4EBF0','Train: 324 clips'),('#E69F00','Validation: 108 clips'),('#0072B2','Test: 108 clips')]],
          loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
fig.tight_layout(pad=.9)
save_figure(fig,'cv03_subject_protocol')

writer=PdfWriter()
for name in figures:
    writer.append(PdfReader(ASSETS/(name+'.pdf')))
with (ASSETS/'figure_catalog.pdf').open('wb') as f:writer.write(f)
write(ASSETS/'plot_data.json',{'fold_means_and_seed_sd':fold_rows,'subject_means_and_seed_sd':subject_rows,
                            'subject_assignment_matrix':assignment.tolist(),'paired_changes':paired})

table = [r'\begin{table*}[t]',r'\centering',r'\caption{Paired frozen MoViNet A0-Stream adaptation on five subject-held-out KARD folds and three seeds per fold. Values are percentages: mean $\pm$ sample SD across five fold means, after averaging the three seeds within each fold. These SDs are descriptive, not confidence intervals.}',
         r'\label{tab:kard-sampling-cv}',r'\begin{tabular}{ccccc}',r'\toprule',
         r'FPS & Validation accuracy & Validation macro F1 & Test accuracy & Test macro F1 \\',r'\midrule']
for r in summary_rows:
    cells = [str(r['fps'])]+[f"${r[name+'_mean_pct']:.2f} \\pm {r[name+'_fold_sd_pp']:.2f}$" for name in ['validation_accuracy','validation_macro_f1','test_accuracy','test_macro_f1']]
    table.append(' & '.join(cells)+r' \\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
table_text='\n'.join(table)
(ASSETS/'table_results.tex').write_text(table_text,encoding='utf-8')
captions = [
    'Test accuracy and macro F1 for paired 5- and 10-fps frozen-backbone heads. Points average three training seeds within each subject fold; error bars show sample SD across those seeds, not confidence intervals. All checkpoint, L2 and sampling-policy choices were locked using validation before new-fold test scoring. Training subjects overlap across folds.',
    'Held-out subject test accuracy at 5 versus 10 fps. Each subject contributes 54 videos and is tested once per seed. Points average three seeds; horizontal bars show seed SD, not uncertainty from independent subject samples. Each subject is predicted by the model trained without that subject.',
    'Prespecified five-fold subject assignment. Every subject is held out for test once and validation once. The remaining six subjects train. Each assignment is run at 5 and 10 fps with seeds 42, 123 and 2026; frame caps of 50 and 100 preserve a centered maximum 10-second window. This is a researcher-defined protocol, not an official KARD benchmark.'
]
snippets = ['% Requires graphicx. Copy these PDFs into figures/kard_sampling_cv/.',
            '% ICML: figure* for wide assets, figure for cv02. ICLR: use figure for all.']
for name,caption in zip(figures,captions):
    env = 'figure' if name.startswith('cv02') else 'figure*'
    width = r'\columnwidth' if env=='figure' else r'\textwidth'
    snippets += [f'\\begin{{{env}}}[t]',r'\centering',f'\\includegraphics[width={width}]{{figures/kard_sampling_cv/{name}.pdf}}',
                 '\\caption{'+caption+'}',f'\\label{{fig:{name}}}',f'\\end{{{env}}}','']
(ASSETS/'include_figures.tex').write_text('\n'.join(snippets),encoding='utf-8')

lines = ['# KARD: paired sampling comparison across subject folds and seeds','',
    'Completed five subject folds, three seeds per fold, and two sampling rates: **30 fresh heads and 90 L2 candidate fits**. The official A0-Stream backbone remains frozen. All 30 checkpoint/L2 selections and 15 validation-based sampling decisions were locked before new-fold test scoring.','',
    '| FPS | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |',
    '|---|---:|---:|---:|---:|']
for r in summary_rows:
    lines.append('| '+str(r['fps'])+' | '+' | '.join(f"{r[name+'_mean_pct']:.2f} ± {r[name+'_fold_sd_pp']:.2f}%" for name in ['validation_accuracy','validation_macro_f1','test_accuracy','test_macro_f1'])+' |')
lines += ['', 'Values are means ± sample SD across five fold means; three seeds are averaged within each fold. Seeds are not counted as independent subject groups. SDs are descriptive, not confidence intervals.','',
    f"10 minus 5 fps: test accuracy **{summary['test_accuracy_change_10_minus_5_pp']:+.2f} percentage points**, test macro F1 **{summary['test_macro_f1_change_10_minus_5_pp']:+.2f} points**. 10 fps has higher test macro F1 in {summary['paired_run_macro_f1_wins_10fps']}/15 paired fold/seed runs and {summary['fold_mean_macro_f1_wins_10fps']}/5 seed-averaged folds; {summary['paired_run_macro_f1_ties']} paired ties.",
    f"Validation selects 10 fps in {summary['validation_choices_10fps']}/15 runs and 5 fps in {summary['validation_choices_5fps']}/15. The procedure that uses each run's validation-selected rate has descriptive test accuracy {summary['validation_selected_procedure']['test_accuracy_mean_pct']:.2f}% and macro F1 {summary['validation_selected_procedure']['test_macro_f1_mean_pct']:.2f}%. Both fixed-rate arms are still reported; no post-test tuning occurs.",'',
    '## Protocol and leakage controls','',
    '| Fold | Training subjects | Validation subjects | Test subjects |','|---|---|---|---|']
for f in protocol['folds']:
    lines.append('| '+str(f['fold'])+' | '+' | '.join(', '.join(map(str,f['subjects'][s])) for s in ['train','validation','test'])+' |')
lines += ['', '## Per-seed out-of-fold scores','',
    'Each row pools the five held-out folds, covering all 540 unique clips once for that seed. Pooled macro F1 differs from the mean of fold macro F1 values above.',
    '', '| Seed | FPS | Out-of-fold test accuracy | Out-of-fold test macro F1 |','|---|---:|---:|---:|']
for r in seed_rows:
    lines.append(f"| {r['seed']} | {r['fps']} | {r['out_of_fold_accuracy_pct']:.2f}% | {r['out_of_fold_macro_f1_pct']:.2f}% |")
lines += ['', '## Controls and verification']
lines += ['',
    '- Each fold has 324/108/108 clips with disjoint subjects; every action has 18/6/6 clips. Every subject is tested once per seed. Sorted consecutive pairs determine test folds; the next pair cyclically validates. No prediction-based partitioning.',
    '- Seeds 42, 123, 2026; both rates use matched initialization and shuffle seed. RGB [0,1], aspect-preserving padding to 172 square, centered maximum 10-second window, fresh streaming state per clip; caps 50/100. No cropping or augmentation.',
    '- All 1,080 cached feature vectors and 540 source-file hashes are verified. Only raw features from the fixed pretrained backbone are reused. Their former split labels do not restrict their use in a new fold. No old normalization, trained classifier or previous prediction enters training.',
    '- Fresh train-only 480-feature normalization for each fold/rate; floor std 0.001. Dropout 0.25 and Dense(18), 8,658 trainable parameters. Adam 0.001, batch 16, max 200 epochs, patience 25, L2 candidates 0.0001/0.001/0.01. Checkpoint/candidate choice: highest validation macro F1, then lower cross entropy.',
    '- All selected .keras heads reload with maximum logit error below 1e-6. Independent checks confirm all 30 normalizations equal statistics of exactly their training clips, all 90 candidate checkpoints match their best validation history entry, optimizer steps match executed epochs, and confusion-matrix recomputation agrees with all 30 test reports.',
    '- Runs use Miniconda xai and native Windows TensorFlow CPU. This experiment does not benchmark inference latency or GPU execution.','',
    '## Interpretation and limits','',
    'This study measures stability of two sampling policies for a frozen pretrained representation and a lightweight KARD head. It is an exploratory follow-up after earlier KARD results were observed. Test subjects are excluded from their own model training and normalization; this is not a newly blind external evaluation. Training sets overlap across folds and seeds use the same clips. No p-values, confidence intervals or population-level significance claims are made. Higher fps changes both input density and the physical receptive field of temporal convolutions. Performance alone does not establish why a particular action succeeds or fails.',
    'These are researcher-defined subject folds, not a published official KARD benchmark. No new privacy method, architectural novelty or upper-backbone fine-tuning is claimed. Do not choose the best test-scoring fold checkpoint as a deployment model: the fold checkpoints serve evaluation and do not train on all ten subjects.','',
    '## Reproduction and artifacts','',
    'Run `run_kard_sampling_cv.py` followed by `write_kard_sampling_cv_report.py` with the xai interpreter. The runner preserves matching completed selections and verifies immutable protocol/input/selection hashes on resume.',
    '`protocol.json`, `feature_provenance.json`, `selection_lock.json`, and per-run candidate histories, normalization, selected heads, predictions, test logits and metrics retain the evidence. Numeric CSVs provide run, fold, subject, seed, action and paired summaries. Action metrics pool the five held-out folds for each seed, then average seeds. `paper_assets/` contains PDF/SVG/600-dpi PNG figures, figure/table LaTeX snippets and exact plot data. Existing earlier checkpoints are preserved.']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

doc = r'''\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,amsmath}
\title{KARD: Temporal Sampling Across Subject Folds and Seeds}
\author{}\date{}
\begin{document}\maketitle
\section{Protocol}
We compare 5 and 10 fps for a frozen official MoViNet A0-Stream backbone and a freshly initialized 18-class head. Five folds use consecutive test-subject pairs (1,2), (3,4), (5,6), (7,8), (9,10). The next pair cyclically supplies validation subjects; six remaining subjects train. Each fold has 324 training, 108 validation and 108 test clips. Every subject is tested once per seed. The three training seeds are 42, 123 and 2026, paired between rates.

RGB values are scaled to $[0,1]$ and resized with aspect-preserving padding to $172\times172$. A centered maximum 10-second window uses caps of 50 or 100 frames, with streaming state reset for each clip. Only raw features from the fixed pretrained backbone are reused; all source and feature hashes are verified. Normalization is fitted separately on each fold/rate's training clips, with standard deviation floored at 0.001. The head uses dropout 0.25 and a dense 18-class output (8,658 trainable parameters). Adam uses learning rate 0.001, batch size 16, maximum 200 epochs and patience 25. Three L2 candidates ($10^{-4},10^{-3},10^{-2}$) are selected by validation macro F1, then lower validation cross entropy. All 30 head selections and 15 sampling-policy recommendations are recorded and hashed before new-fold test scoring. Native Windows TensorFlow CPU runs in Miniconda xai.
\section{Results}
'''+table_text.replace('table*','table')+'\n'
doc += f"The descriptive test change at 10 versus 5 fps is {summary['test_accuracy_change_10_minus_5_pp']:+.2f} percentage points in accuracy and {summary['test_macro_f1_change_10_minus_5_pp']:+.2f} points in macro F1. Ten fps has higher macro F1 in {summary['paired_run_macro_f1_wins_10fps']} of 15 paired fold/seed runs and {summary['fold_mean_macro_f1_wins_10fps']} of five seed-averaged folds. Validation recommends 10 fps in {summary['validation_choices_10fps']} of 15 cases. The validation-selected sampling procedure has descriptive test accuracy {summary['validation_selected_procedure']['test_accuracy_mean_pct']:.2f}\\% and macro F1 {summary['validation_selected_procedure']['test_macro_f1_mean_pct']:.2f}\\%. No post-test tuning is performed.\n"
doc += r'''\section{Interpretation and limitations}
This is an exploratory comparison following previously observed KARD results, using a researcher-defined subject protocol rather than an official benchmark. Test subjects are excluded from their own head training and normalization; the dataset is not a fresh external evaluation. Folds overlap in training and repeated seeds share clips. Means and SDs are descriptive; seeds are not independent subject groups and no significance or confidence-interval claims are made. Dense sampling changes both available frames and physical temporal receptive fields. No architectural novelty, privacy guarantee or backbone adaptation is claimed. Fold checkpoints serve evaluation and must not be selected by test score for deployment.

The accompanying figure PDFs, editable SVGs, PNGs, captions, table snippet, CSVs and exact plot data support direct manuscript inclusion. All selected heads pass serialization reload checks, and test metrics are independently recomputed from predictions.
\end{document}
'''
(OUT/'sampling_cv_results.tex').write_text(doc,encoding='utf-8')
for path in [OUT/'protocol.json',OUT/'selection_lock.json',OUT/'feature_provenance.json',OUT/'summary.json',
             ROOT/'run_kard_sampling_cv.py',Path(__file__)]:
    provenance.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
write(ASSETS/'provenance.json',{'inputs_and_results':provenance,'figures':figures,
      'audit':'All 30 train-only normalizations and selections, all 90 candidate history/step counts, paired seeds/subjects, 108-clip confusion matrices and 540-clip out-of-fold seed predictions verified.'})
print('CV_REPORT_COMPLETE',json.dumps(summary),flush=True)
