"""Publication-vector figures; descriptive evidence, not causal attribution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
out=root/'architecture_challenge_20261010'
summary=json.loads((root/'kard_a0_controlled_ablation_v1'/'summary.json').read_text())
diagnosis=json.loads((out/'diagnosis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
fig.subplots_adjust(left=.08,right=.99,top=.9,bottom=.30,wspace=.32)
ax=axes[0]
means=[100*summary[k]['test']['macro_f1']['mean'] for k in 'ABCD']
sd=[100*summary[k]['test']['macro_f1']['sample_sd'] for k in 'ABCD']
ax.bar(np.arange(4),means,yerr=sd,capsize=3,color=['#3469A0','#A8BCD0','#DD8957','#BEC6CD'],width=.65)
ax.set_xticks(range(4),['Original\naffine','Compact\nmean','Compact\ntemporal','Original\nMLP'])
ax.set_ylim(0,105);ax.set_ylabel('Test macro F1 (%)')
ax.set_title('(a) Completed controlled ablation',loc='left',fontsize=10)
for i,v in enumerate(means):ax.text(i,v+sd[i]+2,f'{v:.2f}',ha='center',fontsize=8)
ax=axes[1]
values=[100*diagnosis['splits']['validation'][k]['accuracy'] for k in ('original','spatial_pool_before_projection')]
ax.bar([0,1],values,color=['#3469A0','#DD8957'],width=.55)
ax.set_xticks([0,1],['Spatial mean\nafter activation','Spatial mean\nbefore projection'])
ax.set_ylim(0,105);ax.set_ylabel('Validation accuracy (%)')
ax.set_title('(b) New representation intervention',loc='left',fontsize=10)
for i,v in enumerate(values):ax.text(i,v+2,f'{v:.2f}',ha='center',fontsize=8)
fig.text(.02,.01,'(a) Error bars: SD across 3 training seeds; fixed, previously observed test split.\n'
    '(b) Same pretrained projection and unchanged original classifier; no retraining.\n'
    'The representation intervention is not an ablation of the separately retrained compact heads.',fontsize=7)
for suffix in ('pdf','svg','png'):
    fig.savefig(out/f'architecture_diagnosis.{suffix}',dpi=250)
plt.close(fig)
