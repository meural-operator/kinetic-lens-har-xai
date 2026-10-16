"""Extend the fixed 25% subset with balanced integer-flow sample selection."""
from collections import Counter,deque
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from kard_config import ROOT,OUT,PROTOCOL,LABELS,SEED,FRACTION

assert FRACTION==0.5
if (OUT/'manifest.json').exists():raise RuntimeError('Manifest already frozen; do not overwrite')
BASE=ROOT/'kard_a0_head_25pct'
parent=json.loads((BASE/'manifest.json').read_text())
records=parent['records']
for r in records:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
pool=[r for r in records if r['split']=='train']
old=[r for r in pool if r['selected']]
assert len(pool)==324 and len(old)==81
subjects=parent['split_subjects']['train']
rng=np.random.default_rng(SEED)

class Flow:
    def __init__(self,n):self.g=[[] for _ in range(n)]
    def edge(self,a,b,capacity,cost):
        forward=[b,len(self.g[b]),capacity,cost,capacity]
        reverse=[a,len(self.g[a]),0,-cost,0]
        self.g[a].append(forward);self.g[b].append(reverse)
        return forward
    def send(self,source,sink,target):
        total=0;cost=0
        while total<target:
            dist=[float('inf')]*len(self.g);dist[source]=0
            prev=[None]*len(self.g);queue=deque([source]);queued={source}
            while queue:
                a=queue.popleft();queued.remove(a)
                for i,e in enumerate(self.g[a]):
                    b,_,cap,weight,_=e
                    if cap and dist[b]>dist[a]+weight:
                        dist[b]=dist[a]+weight;prev[b]=(a,i)
                        if b not in queued:queue.append(b);queued.add(b)
            if prev[sink] is None:break
            amount=target-total;b=sink
            while b!=source:
                a,i=prev[b];amount=min(amount,self.g[a][i][2]);b=a
            b=sink
            while b!=source:
                a,i=prev[b];e=self.g[a][i];e[2]-=amount;self.g[b][e[1]][2]+=amount;b=a
            total+=amount;cost+=amount*dist[sink]
        return total,cost

# Every action uses all six subjects once, plus three subject/action pairs twice.
flow=Flow(26);links={}
for c in range(18):
    flow.edge(0,1+c,3,0)
    for j,s in enumerate(subjects):
        links[c,s]=flow.edge(1+c,19+j,1,int(rng.integers(1,10000)))
for j,s in enumerate(subjects):flow.edge(19+j,25,9,0)
assert flow.send(0,25,54)[0]==54
pair_targets={(c,s):1+int(e[4]-e[2]) for (c,s),e in links.items()}
parent_pairs=Counter((r['class_id'],r['subject_id']) for r in old)
assert max(parent_pairs.values())==1
parent_reps=Counter(r['repetition'] for r in old)
parent_class_reps=Counter((r['class_id'],r['repetition']) for r in old)

# Assign the additional 81 clips, retaining all originals and exactly balancing
# repetition totals. Prefer balanced repetitions within each class as well.
assignment=None
for max_per_class_rep in [4,5,6]:
    graph=Flow(167);clip_edges={}
    for c in range(18):
        for j,s in enumerate(subjects):
            cell=1+c*6+j
            residual=pair_targets[c,s]-parent_pairs[c,s]
            graph.edge(0,cell,residual,0)
            for r in pool:
                if r['class_id']==c and r['subject_id']==s and not r['selected']:
                    node=109+c*3+r['repetition']-1
                    clip_edges[r['relative_path']]=graph.edge(cell,node,1,int(rng.integers(0,50)))
    for c in range(18):
        for repetition in range(1,4):
            present=parent_class_reps[c,repetition]
            for count in range(present,max_per_class_rep):
                graph.edge(109+c*3+repetition-1,163+repetition-1,1,(count+1)*100)
    for repetition in range(1,4):graph.edge(163+repetition-1,166,54-parent_reps[repetition],0)
    sent,cost=graph.send(0,166,81)
    if sent==81:
        assignment=[key for key,e in clip_edges.items() if e[4]-e[2]];break
assert assignment is not None,'Balanced nested sample allocation is infeasible'
original=set(r['relative_path'] for r in old)
extended=original|set(assignment)
for r in records:
    if r['split']=='train':r['selected']=r['relative_path'] in extended
new=[r for r in records if r['split']=='train' and r['selected']]
assert len(new)==162 and original<=extended
class_counts=Counter(r['class_id'] for r in new)
subject_counts=Counter(r['subject_id'] for r in new)
repetition_counts=Counter(r['repetition'] for r in new)
pair_counts=Counter((r['class_id'],r['subject_id']) for r in new)
assert set(class_counts.values())=={9} and set(subject_counts.values())=={27}
assert set(repetition_counts.values())=={54} and len(pair_counts)==108
assert set(pair_counts.values())=={1,2}
parent_hash=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()
manifest={**parent,'protocol':PROTOCOL,'selected_counts':{'train':162,'validation':108,'test':108},
    'parent_manifest_sha256':parent_hash,'nested_original_clips_retained':81,'added_training_clips':81,
    'training_subject_counts':dict(sorted(subject_counts.items())),
    'training_repetition_counts':dict(sorted(repetition_counts.items())),
    'training_class_counts':{LABELS[c]:class_counts[c] for c in range(18)},
    'training_class_repetition_counts':{LABELS[c]:{str(e):sum(r['class_id']==c and r['repetition']==e for r in new) for e in range(1,4)} for c in range(18)},
    'curation_max_per_class_repetition':max_per_class_rep,'records':records}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
for name in ['dataset_audit.json','dataset_full_decode_audit.json','class_contact_sheet.jpg']:
    shutil.copy2(BASE/name,OUT/name)
(OUT/'nested_selection_verification.json').write_text(json.dumps({
    'source_video_hashes_verified':540,'original_training_clips_retained':81,'new_training_clips':81,
    'class_counts':dict(class_counts),'subject_counts':dict(subject_counts),'repetition_counts':dict(repetition_counts),
    'all_training_subjects_present_per_class':True,'no_subject_overlap_between_splits':True,
    'selection_uses_predictions_or_features':False,'parent_manifest_sha256':parent_hash},indent=2),encoding='utf-8')
print('NESTED_PREPARATION_COMPLETE',json.dumps({'clips':162,'per_class':9,'per_subject':27,
    'per_repetition':54,'retained_original':81,'max_class_repetition':max_per_class_rep}),flush=True)
