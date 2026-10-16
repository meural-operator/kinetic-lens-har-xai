"""Document partial fine-tuning with exact data, vector figures and LaTeX."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'kard_a0_upper_finetune';BASE=ROOT/'kard_a0_head_100pct'
(OUT/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str(OUT/'tmp');os.environ['MPLCONFIGDIR']=str(OUT/'tmp'/'matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader,PdfWriter
import pymupdf

read=lambda folder,name:json.loads((folder/name).read_text())
s=read(OUT,'selection.json');e=read(OUT,'evaluation.json');p=read(OUT,'protocol.json');h=read(OUT,'history.json')
updated=read(OUT,'updated_backbone_verification.json')
assert updated['frozen_lower_and_bn_unchanged'] and not updated['test_used']
bs=read(BASE,'selection.json');be=read(BASE,'evaluation.json');labels=read(OUT,'labels.json')
assert not s['test_used_for_training_or_selection'] and not e['test_used_for_training_or_selection']
assert s['frozen_and_bn_variables_unchanged'] and e['export_verification']['saved_model_reload_passed']
assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==s['protocol_sha256']==e['protocol_sha256']
old={r['video']:r for r in read(BASE,'test_predictions.json')};new={r['video']:r for r in read(OUT,'test_predictions.json')}
assert old.keys()==new.keys() and len(new)==108
for key in new:assert (old[key]['true_class_id'],old[key]['subject_id'])==(new[key]['true_class_id'],new[key]['subject_id'])
recovered=[v for v in new if new[v]['correct'] and not old[v]['correct']]
regressed=[v for v in new if old[v]['correct'] and not new[v]['correct']]
comparison={'baseline_test_accuracy':be['test']['accuracy'],'finetune_test_accuracy':e['test']['accuracy'],
    'baseline_test_macro_f1':be['test']['macro_f1'],'finetune_test_macro_f1':e['test']['macro_f1'],
    'test_accuracy_change_percentage_points':100*(e['test']['accuracy']-be['test']['accuracy']),
    'selected_epoch':s['selected_epoch'],'corrected_baseline_errors':recovered,'introduced_errors':regressed,
    'selected_by_validation_only':True,'test_previously_observed':True,'post_test_tuning':False}
(OUT/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
rows=[]
selected_name='Epoch-0 fallback' if s['selected_starting_baseline'] else 'Upper fine-tuning'
best_trial=max(h,key=lambda entry:(entry['validation']['macro_f1'],-entry['validation']['cross_entropy']))
for name,train,val,test in [('Frozen head',bs['winner']['train'],bs['winner']['validation'],be['test']),
                            (selected_name,s['train'],s['validation'],e['test'])]:
    rows.append({'model':name,'train_clips':324,'validation_clips':108,'test_clips':108,
        'train_accuracy_pct':100*train['accuracy'],'validation_accuracy_pct':100*val['accuracy'],
        'validation_macro_f1_pct':100*val['macro_f1'],'test_correct':int(np.trace(np.asarray(test['confusion_matrix']))),
        'test_accuracy_pct':100*test['accuracy'],'test_macro_f1_pct':100*test['macro_f1']})
with (OUT/'results_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'action_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['action','test_support','frozen_test_correct','selected_test_correct','frozen_test_recall','selected_test_recall','initial_validation_f1','best_trained_epoch_validation_f1'])
    for i,label in enumerate(labels):w.writerow([label,6,be['test']['confusion_matrix'][i][i],e['test']['confusion_matrix'][i][i],be['test']['recall_per_class'][i],e['test']['recall_per_class'][i],s['initial_validation']['f1_per_class'][i],best_trial['validation']['f1_per_class'][i]])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65,'lines.linewidth':1.3,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2';ORANGE='#D55E00';GRAY='#68737D'
assets=OUT/'paper_assets';assets.mkdir(exist_ok=True);figures=[]
def save(fig,name,caption):
    for suffix in ['pdf','svg','png']:fig.savefig(assets/(name+'.'+suffix),dpi=600)
    with pymupdf.open(assets/(name+'.pdf')) as pdf:
        assert pdf.page_count==1 and len(pdf[0].get_text())>25
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(OUT/'tmp'/(name+'_pdf_render.png'))
    figures.append({'name':name,'caption':caption});plt.close(fig)
def clean(ax):ax.set_axisbelow(True);ax.grid(axis='y',color='#DCE3E9',linewidth=.5)

fig,ax=plt.subplots(figsize=(3.25,2.5));fig.subplots_adjust(left=.18,right=.97,bottom=.25,top=.95)
x=np.arange(3)
for shift,row,color,label,hatch in [(-.18,rows[0],BLUE,'Frozen head',''),(.18,rows[1],ORANGE,selected_name,'//')]:
    vals=[row[k] for k in ['validation_macro_f1_pct','test_accuracy_pct','test_macro_f1_pct']]
    bars=ax.bar(x+shift,vals,width=.33,color=color,edgecolor='white',linewidth=.3,hatch=hatch,label=label)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=6.5)
ax.set_xticks(x,['Validation\nmacro F1','Test\naccuracy','Test\nmacro F1']);ax.set(ylabel='Performance (%)',ylim=(0,109),yticks=[0,25,50,75,100])
ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=6.4,columnspacing=.8,handlelength=1.2);clean(ax)
fallback_caption=' Validation selected the epoch-zero fallback, so the second set of scores repeats the starting model; these are not test scores of a fine-tuned checkpoint.' if s['selected_starting_baseline'] else ''
save(fig,'finetune01_comparison','Frozen-head model versus the checkpoint retained after the partial-fine-tuning experiment, both using 324 training clips and the same 108 validation and 108 test clips. The final two residual layers and projection were optimized at 1e-5 with the initialized head. All normalization and lower layers stayed fixed. One seed and one previously evaluated split; no uncertainty intervals or significance claim.'+fallback_caption)

fig,axs=plt.subplots(1,2,figsize=(6.75,2.6));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.90,wspace=.30)
epochs=[0]+[r['epoch'] for r in h]
train_f1=[100*bs['winner']['train']['macro_f1']]+[100*r['train']['macro_f1'] for r in h]
val_f1=[100*s['initial_validation']['macro_f1']]+[100*r['validation']['macro_f1'] for r in h]
val_ce=[s['initial_validation']['cross_entropy']]+[r['validation']['cross_entropy'] for r in h]
axs[0].plot(epochs,train_f1,color=BLUE,label='Training');axs[0].plot(epochs,val_f1,color=ORANGE,linestyle='--',label='Validation')
axs[0].set(xlabel='Fine-tuning epoch',ylabel='Macro F1 (%)',ylim=(80,102),yticks=[80,85,90,95,100],title='(a) Training and validation');axs[0].legend(frameon=False,loc='center right')
axs[1].plot(epochs,val_ce,color=ORANGE,marker='o',markersize=3);axs[1].set(xlabel='Fine-tuning epoch',ylabel='Validation cross entropy',title='(b) Validation loss')
for ax in axs:ax.axvline(s['selected_epoch'],color=GRAY,linestyle=':',linewidth=1);clean(ax)
save(fig,'finetune02_training','Unsmoothed partial-fine-tuning trajectories, including the initialized frozen-head model at epoch zero. The macro F1 axis starts at 80% to show the small validation changes. The dotted line marks the checkpoint selected by validation macro F1, with lower validation cross entropy as the tie breaker. Test scores are not used for stopping or checkpoint selection. The starting model remains an eligible fallback.')

fig,ax=plt.subplots(figsize=(6.75,4.2));fig.subplots_adjust(left=.25,right=.96,bottom=.14,top=.95)
yy=np.arange(18)
for val,offset,color,marker,label in [(s['initial_validation']['f1_per_class'],-.13,BLUE,'o','Initialization'),(best_trial['validation']['f1_per_class'],.13,ORANGE,'s',f"Best trained epoch ({best_trial['epoch']})")]:
    ax.plot(np.asarray(val)*100,yy+offset,linestyle='none',marker=marker,markersize=4,color=color,label=label)
for i,(a,b) in enumerate(zip(s['initial_validation']['f1_per_class'],best_trial['validation']['f1_per_class'])):ax.plot([a*100,b*100],[i-.13,i+.13],color='#CCD6DE',linewidth=.7,zorder=0)
ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set(xlabel='Per-action validation F1 (%)',xlim=(-4,104),xticks=[0,25,50,75,100])
ax.grid(axis='x',color='#DCE3E9',linewidth=.5);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1),ncol=2)
save(fig,'finetune03_validation_action_f1','Per-action validation F1 for the initialized model and the best actually trained epoch, chosen among epochs one onward by validation macro F1 and cross entropy. The overall selection also permits epoch zero, which may outperform this best trained epoch. Each action has six validation clips from two subjects; the comparison is descriptive and does not establish statistical significance. This figure uses validation results only.')
writer=PdfWriter()
for f in figures:writer.append(str(assets/(f['name']+'.pdf')))
with (assets/'figure_catalog.pdf').open('wb') as f:writer.write(f)
assert len(PdfReader(assets/'figure_catalog.pdf').pages)==3
table=[r'\begin{table*}[t]',r'\centering',r'\caption{KARD frozen-head versus partial-backbone fine-tuning. Both use 324 training clips and the same validation/test split (108 clips each). Values are percentages; mF1 denotes macro F1. One seed and one previously evaluated split.}',r'\label{tab:kard-upper-finetune}',r'\small',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Model & Train acc. & Val. acc. & Val. mF1 & Test acc. & Test mF1 \\',r'\midrule']
for r in rows:table.append(r['model']+' & '+' & '.join(f'{r[k]:.2f}' for k in ['train_accuracy_pct','validation_accuracy_pct','validation_macro_f1_pct','test_accuracy_pct','test_macro_f1_pct'])+' '+r'\\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(assets/'table_results.tex').write_text('\n'.join(table),encoding='utf-8')
snippets=['% Requires graphicx. Copy PDFs into figures/kard_finetune/.', '% Use figure rather than figure* for wide assets in ICLR; width=3.25in for the small asset.']
for i,f in enumerate(figures):
    env='figure' if i==0 else 'figure*';width=r'\columnwidth' if i==0 else r'\textwidth'
    caption_tex=f['caption'].replace('%',r'\%')
    snippets += [f'\\begin{{{env}}}[t]',r'\centering',f"\\includegraphics[width={width}]{{figures/kard_finetune/{f['name']}.pdf}}",f"\\caption{{{caption_tex}}}",f'\\label{{fig:kard-finetune-{i+1}}}',f'\\end{{{env}}}','']
(assets/'include_figures.tex').write_text('\n'.join(snippets),encoding='utf-8')
(assets/'plot_data.json').write_text(json.dumps({'rows':rows,'history':h,'initial_validation':s['initial_validation'],
    'selected_epoch':s['selected_epoch'],'labels':labels,'baseline_test':be['test'],'selected_checkpoint_test':e['test']},indent=2),encoding='utf-8')
epoch_table=[r'\begin{center}',r'\begin{tabular}{rrrr}',r'\toprule',r'Epoch & Val. accuracy (\%) & Val. macro F1 (\%) & Val. cross entropy \\',r'\midrule',
    f"0 & {100*s['initial_validation']['accuracy']:.2f} & {100*s['initial_validation']['macro_f1']:.2f} & {s['initial_validation']['cross_entropy']:.4f} "+r'\\']
for entry in h:
    v=entry['validation'];epoch_table.append(f"{entry['epoch']} & {100*v['accuracy']:.2f} & {100*v['macro_f1']:.2f} & {v['cross_entropy']:.4f} "+r'\\')
epoch_table += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
note=r'''\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,amsmath}
\title{KARD Partial-Backbone Fine-Tuning: Protocol and Negative Result}
\author{}\date{}
\begin{document}\maketitle
\section{Protocol}
We initialize the official MoViNet A0-Stream backbone and the validation-selected KARD head trained on all 324 training clips. The same subject-disjoint split is retained: six training subjects (1,3,4,6,7,8), two validation subjects (5,10; 108 clips), and two test subjects (2,9; 108 clips). RGB input, 5 fps sampling, aspect-preserving padding to $172\times172$ and the centered 10-second/50-frame cap are unchanged.

The final two residual layers (b4/l2, b4/l3) and pooled projection layer are updated together with the head: 498,850 backbone plus 8,658 classifier parameters. All earlier weights and all batch-normalization variables remain fixed; the upper backbone calls use inference mode while gradients update only explicitly selected parameters. Training-only head standardization remains fixed, with dropout 0.25 and L2 regularization 0.01. Adam uses learning rate $10^{-5}$ and global-norm gradient clipping at 1.0. Exact-length buckets contain at most eight clips; no new frame padding, augmentation or cropping is introduced. Every training clip is used once per epoch.

The declared maximum is 20 epochs with early-stopping patience five. Checkpoints are selected by validation macro F1, then lower validation cross entropy. The initialized epoch-zero model is an eligible fallback. No learning-rate or layer-scope search is performed in this run. Frozen lower-layer activations are cached at b4/l1; upper layers execute real forward and backward passes. Training uses native Windows TensorFlow CPU in the Miniconda xai environment.
\section{Validation and checkpoint selection}
'''+ '\n'.join(epoch_table)+ '\n'
if s['selected_starting_baseline']:
    note+=f"The five actually trained epochs did not exceed the initialized validation macro F1 of {100*s['initial_validation']['macro_f1']:.2f}\\%. The best trained epoch was {best_trial['epoch']} ({100*best_trial['validation']['macro_f1']:.2f}\\%). Early stopping retained epoch zero, so the exported model is the original frozen-head baseline rather than an improved fine-tuned model. The trained final-epoch weights are preserved separately for inspection and were not tested.\n"
else:note+=f"Validation selected epoch {s['selected_epoch']}. Only this selected checkpoint was subsequently tested.\n"
note+=r'\section{Selected-model evaluation}'+'\n'+'\n'.join(table).replace('table*','table')+r'''
The starting model and retained checkpoint use the same test clips; these are not independent repetitions. When epoch zero is retained, matching test scores describe the same inference model. The complete export and reload were checked on three test clips; cached-upper and full-model logits agreed within numerical tolerance.
\section{Verification and limitations}
All 17 selected backbone variables had finite, nonzero gradients in a training-clip check. Actual final trained weights changed all three selected layer groups, changed full-model logits on a training clip, and reproduced the cached-upper path exactly on that check. Frozen lower-layer and batch-normalization hashes remained unchanged. Original downloaded weights, head-only experiment outputs and earlier paper assets were preserved.

This is one seed on one researcher-defined split with only two test subjects, whose earlier results were already observed. Checkpoint selection is validation-only and no post-test tuning was performed. The negative result does not establish that all backbone fine-tuning strategies fail; it concerns the specific declared low-learning-rate, upper-layer scope and early-stopping procedure. No statistical significance, privacy guarantee, population-level generalization or inference-latency improvement is established. Training/selection timing excludes activation extraction, model loading, final testing and export.

Use the accompanying vector figures and table snippets when inserting these results into a paper. Actual trained weights and the validation-retained checkpoint must be distinguished in the manuscript.
\end{document}
'''
(OUT/'finetune_results.tex').write_text(note,encoding='utf-8')
lines=['# KARD: careful upper-backbone fine-tuning','',
    'Initialized from the official pretrained A0-Stream backbone and the validation-selected head trained on the full 324-clip training pool. The final two residual layers (`b4/l2`, `b4/l3`) and backbone projection (`head`) are fine-tuned together with the KARD head. This is partial backbone adaptation, not another frozen-feature head-only run.','',
    '## Declared protocol','',
    '- 498,850 trainable upper-backbone parameters plus 8,658 classifier parameters: 507,508 total.',
    '- Adam learning rate 1e-5; gradient global-norm clipping at 1.0; maximum 20 epochs, patience five.',
    '- Seeded exact-length video buckets with batches of at most eight; no frame padding, repetition or new temporal truncation. Every training clip contributes once per epoch.',
    '- Keep all earlier backbone weights and every batch-normalization variable fixed, including affine gamma/beta. Run backbone layers in inference mode; gradients still update the explicitly selected upper weights.',
    '- Retain the original train-only standardization statistics, dropout 0.25 and head L2 0.01. No new backbone weight decay, augmentation, person cropping, learning-rate search or layer-scope search.',
    '- Keep the same RGB [0,1], 5 fps, aspect-preserving 172x172 padding and maximum 50 frames/centered 10-second window. Reset states per clip.',
    '- Cache only the frozen b4/l1 endpoint in float32. Recompute and differentiate the trainable upper layers each step; verified against the full model.',
    '- Checkpoint selection uses highest validation macro F1, then lower validation cross entropy. Include epoch zero as a fallback. New test activations are created only after selection.json is written.',
    '- Native Windows TensorFlow CPU in the xai environment. Training subjects 1,3,4,6,7,8; validation 5,10; test 2,9. Source hashes unchanged.','',
    '| Model | Train accuracy | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['model']+' | '+' | '.join(f"{r[k]:.2f}%" for k in ['train_accuracy_pct','validation_accuracy_pct','validation_macro_f1_pct','test_accuracy_pct','test_macro_f1_pct'])+' |')
if s['selected_starting_baseline']:
    lines += ['',f"**Negative result:** none of the five trained epochs surpassed the starting validation macro F1. The best trained epoch was {best_trial['epoch']}, with validation macro F1 {100*best_trial['validation']['macro_f1']:.2f}% versus {100*s['initial_validation']['macro_f1']:.2f}% initially. The selection rule retained epoch zero. Consequently, the exported model and its repeated test scores are the original frozen-head baseline, not a newly improved fine-tuned model. Actual final trained weights remain in `last_weights.npz` for inspection, but were not selected or evaluated on test."]
lines += ['', '| Epoch | Validation accuracy | Validation macro F1 | Validation cross entropy |','|---:|---:|---:|---:|',
    f"| 0 (initialization) | {100*s['initial_validation']['accuracy']:.2f}% | {100*s['initial_validation']['macro_f1']:.2f}% | {s['initial_validation']['cross_entropy']:.4f} |"]
for entry in h:
    v=entry['validation'];lines.append(f"| {entry['epoch']} | {100*v['accuracy']:.2f}% | {100*v['macro_f1']:.2f}% | {v['cross_entropy']:.4f} |")
lines += ['',f"Selected epoch {s['selected_epoch']} out of {s['epochs_executed']} executed epochs; {s['optimizer_steps']} optimizer steps. Test accuracy changed by {comparison['test_accuracy_change_percentage_points']:+.2f} percentage points, correcting {len(recovered)} earlier errors and introducing {len(regressed)} errors on previously correct clips.",'',
    f"Maximum selected backbone weight change: {s['maximum_backbone_weight_change_selected']:.6g}. Selected changes by layer group: `{json.dumps(s['selected_weight_changes_by_layer_group'])}`. Frozen and batch-normalization digests remained identical. Starting checkpoint selected as fallback: {s['selected_starting_baseline']}.",'',
    f"Training, evaluation on train/validation and checkpoint selection took {s['training_selection_seconds']:.1f} seconds, excluding activation extraction, model loading, final test and export. This is not a total training-cost or inference-latency benchmark.",'',
    '## Verification and limitations','',
    '- Real finite, nonzero gradients were verified for all 17 selected backbone variables on a training clip. Equal-length batch equivalence passed.',
    f"- Actual last-epoch weights change all three selected layer groups and change full-model logits on a training clip. The cached-upper versus full-model maximum logit error after those updates is {updated['actual_trained_upper_vs_full_max_logit_error']:.3g}. This proves the backbone training path works even when the validation rule rejects the trained checkpoint.",
    '- Three test clips were checked through the complete model and reloaded SavedModel; cached-upper and complete-model predictions agree within numerical tolerance.',
    '- All original head-only models, downloaded weights and paper assets are preserved. No post-test tuning was performed.',
    '- One seed and one researcher-defined subject split; only two test subjects. Earlier test results were already observed, so this is not a newly blind evaluation. No statistical significance, cross-validation, dataset-wide benchmark, privacy guarantee or methodological novelty is established.','',
    '## Artifacts and reuse','',
    '`protocol.json`, `source_verification.json`, `gradient_verification.json`, activation provenance, `history.json` and `selection.json` document the training procedure. `selected_weights.npz` stores the selected upper weights and head arrays; `last_weights.npz` stores the final actually trained weights. A head file alone cannot reproduce an updated backbone. Use the complete `saved_model/` for the selected inference model; when the epoch-zero fallback wins, this model is equivalent to the original frozen-head baseline.','',
    '`paper_assets/` contains PDF/SVG/600-dpi PNG plots, the numerical LaTeX table, figure insertion snippets and exact plot data. Existing head-only paper assets are unchanged.','',
    'Reproduce or resume completed stages with `movinet_evaluation/run_kard_finetune_pipeline.ps1`. An interrupted training history requires inspection rather than silently starting over.','',
    '```powershell',
    "& 'C:\\Users\\Xerox\\miniconda3\\envs\\xai\\python.exe' -B movinet_evaluation\\predict_kard_video.py --experiment-dir movinet_evaluation\\kard_a0_upper_finetune --video 'C:\\path\\to\\clip.mp4'",'```','',
    'The low learning rate and inference-mode normalization follow the [TensorFlow transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning). The original backbone is the [official MoViNet model](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet).']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('FINETUNE_REPORT_COMPLETE',json.dumps(rows),flush=True)
