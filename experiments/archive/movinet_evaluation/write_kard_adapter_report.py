"""Descriptive, paired-seed reporting; no best-seed or novelty claims."""
import csv
import json
from pathlib import Path
import numpy as np

SEEDS=(42,123,2026)
VARIANTS=('lora','guided','random')
NAMES={'baseline':'Frozen strong baseline','lora':'Ordinary rank-4 LoRA',
       'guided':'Classifier-guided rank-4','random':'Random-subspace rank-4'}
PARAMETERS={'baseline':0,'lora':2336,'guided':484,'random':484}

def stats(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),'per_seed':a.tolist()}

def report(out,gate,results,baselines):
    out=Path(out);summary={};rows=[];pairs=[]
    for variant in ('baseline',*VARIANTS):
        validation=[];test=[]
        for seed in SEEDS:
            selection=gate['selections'][f'guided_seed_{seed}' if variant=='baseline' else f'{variant}_seed_{seed}']
            vm=selection['baseline_validation'] if variant=='baseline' else selection['validation']
            tm=baselines[str(seed)] if variant=='baseline' else results[f'{variant}_seed_{seed}']['test']
            validation.append(vm);test.append(tm)
            rows.append({'variant':variant,'seed':seed,'adapter_parameters':PARAMETERS[variant],
                'best_epoch':0 if variant=='baseline' else selection['best_epoch'],
                'l2':0 if variant=='baseline' else selection['l2'],
                'validation_macro_f1':vm['macro_f1'],'test_macro_f1':tm['macro_f1'],
                'test_accuracy':tm['accuracy'],'test_cross_entropy':tm['cross_entropy']})
        summary[variant]={'trainable_adapter_parameters':PARAMETERS[variant],
            'validation':{k:stats([m[k] for m in validation]) for k in ('macro_f1','accuracy','cross_entropy')},
            'test':{k:stats([m[k] for m in test]) for k in ('macro_f1','accuracy','cross_entropy')}}
    paired={}
    for a,b in [('lora','baseline'),('guided','baseline'),('random','baseline'),('guided','random'),('guided','lora')]:
        differences=[100*(summary[a]['test']['macro_f1']['per_seed'][i]-summary[b]['test']['macro_f1']['per_seed'][i]) for i in range(3)]
        paired[f'{a}_minus_{b}']=stats(differences)
        pairs.extend({'comparison':f'{a}_minus_{b}','seed':s,'test_macro_f1_difference_pp':d} for s,d in zip(SEEDS,differences))
    for name,value in [('summary.json',summary),('paired_differences.json',paired)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    for name,data in [('per_seed_metrics.csv',rows),('paired_differences.csv',pairs)]:
        with (out/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def fmt(value):return f"{100*value['mean']:.2f} ± {100*value['sample_sd']:.2f}"
    table=['| Variant | Adapter parameters trained | Validation macro F1 | Test macro F1 | Test accuracy |',
        '|---|---:|---:|---:|---:|']
    for v in ('baseline',*VARIANTS):
        m=summary[v];table.append(f"| {NAMES[v]} | {PARAMETERS[v]:,} | {fmt(m['validation']['macro_f1'])} | {fmt(m['test']['macro_f1'])} | {fmt(m['test']['accuracy'])} |")
    pairtext='\n'.join(f"- {name}: {m['mean']:+.2f} ± {m['sample_sd']:.2f} percentage points; per seed {[round(x,2) for x in m['per_seed']]}" for name,m in paired.items())
    epochtext='\n'.join(f"- {name}: selected epoch {item['best_epoch']}, λ={item['l2']:g}." for name,item in gate['selections'].items())
    text=f'''# KARD mergeable projection-adapter controlled comparison

Each adapter started from the corresponding seed's strong, validation-selected
480D affine baseline from the earlier controlled ablation. Original projection,
BN, backbone and classifier weights were frozen; only adapter factors trained.
All nine model selections were locked before test cache preparation or reads.

{chr(10).join(table)}

Values are percentages, mean ± sample SD across three **training seeds**.
All seeds use the same 108 test clips on a previously observed subject split.
These are descriptive results, not independent-subject confidence intervals.

## Paired test macro F1 differences

{pairtext}

## Selected checkpoints

{epochtext}

Epoch zero means the unchanged baseline was selected. Do not interpret such
a selection as a learned architectural improvement. All models kept the
baseline as a validation fallback. This ensures the selected validation
macro F1 does not decrease on this validation set, not that test accuracy
or unseen-subject generalization improves.

## Controls and limitations

Three seeds (42,123,2026), identical seeded minibatches and dropout starts,
three identical L2 candidates (.0001,.001,.01), Adam lr1e-4, batch16,
gradient clipping1, maximum100 epochs, patience15. The update is constrained
to 2% of the original kernel's Frobenius norm. BN and classifier stay frozen.
The penalty is λ times summed squared entries of the actual kernel update.
Guided and random adapters each train484 parameters; ordinary LoRA trains2336.
The random-subspace variant controls for reduced capacity. No best seed is chosen.

Preflight records original baseline prediction agreement on all432 train/val
clips per variant/seed, correct padding, real-data gradients and updates,
immutable frozen buffers, merged-kernel equivalence, and exact short resume.
The test gate is verified to reject before any test read. Cache sidecars and
inventories verify source videos and original backbone activations. Model
and code hashes are stored in protocol/preflight files.

The adapter merges into the same104→480 projection shape with unchanged
deployed parameter count and operation graph. No new end-to-end GPU/ONNX
latency or accuracy robustness claim follows from this experiment. Low-rank
adaptation is established; novelty of classifier-guided HAR adaptation needs
a separate literature review and independent evaluation. The fixed guidance
subspace need not contain the optimal update because activation derivatives
depend on the input. Negative results are retained.

Figures are exported as PDF/SVG/PNG; CSV and JSON preserve per-seed values.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
    order=('baseline',*VARIANTS);colors=['#386CB0','#E58B45','#4B9B7B','#9A81BC']
    for ax,split in zip(axes,('validation','test')):
        means=[100*summary[v][split]['macro_f1']['mean'] for v in order]
        errors=[100*summary[v][split]['macro_f1']['sample_sd'] for v in order]
        ax.bar(range(4),means,yerr=errors,capsize=3,color=colors,width=.65)
        ax.set_xticks(range(4),['Frozen\nbaseline','Ordinary\nLoRA','Classifier\nguided','Random\nsubspace'])
        ax.set_ylabel('Macro F1 (%)');ax.set_ylim(0,105);ax.set_title(split.capitalize(),loc='left')
        for i,value in enumerate(means):ax.text(i,value+errors[i]+2,f'{value:.2f}',ha='center',fontsize=8)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for extension in ('pdf','svg','png'):fig.savefig(figures/f'adapter_comparison.{extension}',dpi=250)
    plt.close(fig)
