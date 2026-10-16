"""Audit source files, select a balanced subset, and freeze split manifest."""
from collections import Counter
import argparse
import hashlib
import json
import re
import cv2
import numpy as np
from kard_config import DATASET,OUT,LABELS,PROTOCOL,SEED,FRACTION

assert FRACTION==0.25,'Use prepare_kard_50.py for the nested 50% experiment'
if (OUT/'selection.json').exists() or (OUT/'evaluation.json').exists():
    raise RuntimeError('Preserve the existing trained experiment; do not replace its manifest.')

parser=argparse.ArgumentParser()
parser.add_argument('--reuse-audit',action='store_true',help='Reuse a completed full-decoding audit only if no source file was modified afterward; still recheck metadata and hashes.')
args=parser.parse_args()
if args.reuse_audit:
    prior=json.loads((OUT/'dataset_audit.json').read_text())
    assert prior['files']==540 and not prior['errors']
    audit_time=(OUT/'dataset_audit.json').stat().st_mtime
    assert all(p.stat().st_mtime<audit_time for p in DATASET.rglob('*.mp4'))
rng=np.random.default_rng(SEED)
subjects=rng.permutation(np.arange(1,11)).tolist()
split_subjects={'train':sorted(subjects[:6]),'validation':sorted(subjects[6:8]),'test':sorted(subjects[8:])}
records=[]
errors=[]
panels={}
for path in sorted(DATASET.rglob('*.mp4')):
    match=re.fullmatch(r'a(\d{2})_s(\d{2})_e(\d{2})',path.stem)
    if not match:
        errors.append({'path':str(path),'error':'Unexpected filename'});continue
    action,subject,repetition=map(int,match.groups())
    if not (1<=action<=18 and 1<=subject<=10 and 1<=repetition<=3) or path.parent.name.casefold()!=LABELS[action-1].casefold():
        errors.append({'path':str(path),'error':'Folder/action ID or range mismatch'});continue
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS); declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count=0; shapes=set(); sum_std=0.; middle=None
    if args.reuse_audit:
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,declared//2-1))
        ok,frame=cap.read()
        if ok:
            count=declared;shapes.add(frame.shape);sum_std=float(frame.std())*count;middle=frame.copy()
    else:
        while True:
            ok,frame=cap.read()
            if not ok:break
            shapes.add(frame.shape);sum_std+=float(frame.std());count+=1
            if count==max(1,declared//2):middle=frame.copy()
    cap.release()
    issues=[]
    if not np.isfinite(fps) or fps<=0:issues.append('Invalid FPS')
    if not count or count!=declared:issues.append('Missing/undecodable frames')
    if len(shapes)!=1:issues.append('Inconsistent frame shape')
    if count and sum_std/count<2:issues.append('Nearly uniform image sequence')
    if issues:errors.append({'path':str(path),'error':issues})
    split=next(k for k,v in split_subjects.items() if subject in v)
    record={'path':str(path.resolve()),'relative_path':str(path.relative_to(DATASET)),
        'class_id':action-1,'class_name':LABELS[action-1],'subject_id':subject,
        'repetition':repetition,'split':split,'selected':split!='train',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fps':fps,
        'declared_frames':declared,'decoded_frames':count,
        'duration_seconds':count/fps if fps>0 else None,
        'frame_shape':list(next(iter(shapes))) if shapes else None,'integrity_issues':issues}
    records.append(record)
    if middle is not None and action not in panels:
        h,w=middle.shape[:2];scale=min(256/w,192/h)
        image=cv2.resize(middle,(round(w*scale),round(h*scale)))
        canvas=np.zeros((220,256,3),np.uint8)
        rh,rw=image.shape[:2];canvas[28:28+rh,:rw]=image
        cv2.putText(canvas,f'{action:02d} {LABELS[action-1]}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
        panels[action]=canvas
    if len(records)%54==0:print(f'AUDITED {len(records)}/540',flush=True)
duplicates={h:[r['relative_path'] for r in records if r['sha256']==h]
            for h,n in Counter(r['sha256'] for r in records).items() if n>1}
if duplicates:errors.append({'error':'Duplicate file hashes','groups':duplicates})
expected={(c,s,e) for c in range(18) for s in range(1,11) for e in range(1,4)}
actual=[(r['class_id'],r['subject_id'],r['repetition']) for r in records]
if set(actual)!=expected or len(actual)!=540:errors.append({'error':'Incomplete or duplicated action/subject/repetition grid'})
audit={'files':len(records),'split_subjects':split_subjects,'errors':errors,
       'reused_prior_full_decode_audit':args.reuse_audit,
       'duration_seconds':{'min':min(r['duration_seconds'] for r in records),
           'median':float(np.median([r['duration_seconds'] for r in records])),
           'max':max(r['duration_seconds'] for r in records)},'duplicate_hash_groups':duplicates}
(OUT/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
if not args.reuse_audit:
    (OUT/'dataset_full_decode_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
if errors:raise RuntimeError('Dataset audit failed; inspect dataset_audit.json before training')

extra=set(rng.permutation(18)[:9].tolist())
subject_counts=Counter();repeat_counts=Counter();selected_pairs=[]
for c in rng.permutation(18):
    candidates=[r for r in records if r['split']=='train' and r['class_id']==c]
    tie={r['relative_path']:float(rng.random()) for r in candidates}
    selected_subjects=set()
    for _ in range(5 if c in extra else 4):
        options=[r for r in candidates if r['subject_id'] not in selected_subjects]
        chosen=min(options,key=lambda r:(subject_counts[r['subject_id']],tie[r['relative_path']]))
        selected_pairs.append((int(c),chosen['subject_id']));selected_subjects.add(chosen['subject_id'])
        subject_counts[chosen['subject_id']]+=1
for position in rng.permutation(len(selected_pairs)):
    c,subject=selected_pairs[position]
    repetition=1+sum(repeat_counts.values())%3
    chosen=next(r for r in records if r['class_id']==c and r['subject_id']==subject and r['repetition']==repetition)
    chosen['selected']=True;repeat_counts[repetition]+=1
counts={k:sum(r['selected'] and r['split']==k for r in records) for k in split_subjects}
assert counts=={'train':81,'validation':108,'test':108}
assert max(subject_counts.values())-min(subject_counts.values())<=1
assert max(repeat_counts.values())-min(repeat_counts.values())<=1
manifest={'protocol':PROTOCOL,'labels':LABELS,'split_subjects':split_subjects,
          'selected_counts':counts,'training_pool_count':324,
          'training_subject_counts':dict(sorted(subject_counts.items())),
          'training_repetition_counts':dict(sorted(repeat_counts.items())),
          'training_class_counts':{LABELS[c]:sum(r['selected'] and r['split']=='train' and r['class_id']==c for r in records) for c in range(18)},
          'records':records}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
grid=np.concatenate([np.concatenate([panels[j] for j in range(i,i+3)],axis=1) for i in range(1,19,3)],axis=0)
cv2.imwrite(str(OUT/'class_contact_sheet.jpg'),grid)
print(json.dumps({**audit,'selected_counts':counts,'training_subject_counts':dict(subject_counts),
    'training_repetition_counts':dict(repeat_counts)},indent=2),flush=True)
print('PREPARATION_COMPLETE',flush=True)
