"""Freeze the full training-pool experiment without changing held-out subjects."""
from collections import Counter
import hashlib
import json
import shutil
from pathlib import Path
from kard_config import ROOT,OUT,PROTOCOL,LABELS,FRACTION

assert FRACTION==1.0
if (OUT/'manifest.json').exists():raise RuntimeError('Manifest already frozen; do not overwrite')
BASE=ROOT/'kard_a0_head_50pct'
parent=json.loads((BASE/'manifest.json').read_text())
records=parent['records']
for r in records:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
pool=[r for r in records if r['split']=='train']
old={r['relative_path'] for r in pool if r['selected']}
assert len(pool)==324 and len(old)==162
small=json.loads((ROOT/'kard_a0_head_25pct'/'manifest.json').read_text())
original={r['relative_path'] for r in small['records'] if r['split']=='train' and r['selected']}
assert len(original)==81 and original<=old
subjects=[set(parent['split_subjects'][s]) for s in ['train','validation','test']]
assert all(not subjects[i]&subjects[j] for i in range(3) for j in range(i+1,3))
for r in records:r['selected']=True
class_counts=Counter(r['class_id'] for r in pool)
subject_counts=Counter(r['subject_id'] for r in pool)
repetition_counts=Counter(r['repetition'] for r in pool)
assert set(class_counts.values())=={18} and set(subject_counts.values())=={54}
assert set(repetition_counts.values())=={108}
parent_hash=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()
manifest={**parent,'protocol':PROTOCOL,'selected_counts':{'train':324,'validation':108,'test':108},
    'parent_manifest_sha256':parent_hash,'nested_original_clips_retained':162,'added_training_clips':162,
    'training_subject_counts':dict(sorted(subject_counts.items())),
    'training_repetition_counts':dict(sorted(repetition_counts.items())),
    'training_class_counts':{LABELS[c]:class_counts[c] for c in range(18)},
    'training_class_repetition_counts':{LABELS[c]:{str(e):6 for e in range(1,4)} for c in range(18)},
    'curation_max_per_class_repetition':6,'records':records}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
for name in ['dataset_audit.json','dataset_full_decode_audit.json','class_contact_sheet.jpg']:
    shutil.copy2(BASE/name,OUT/name)
(OUT/'nested_selection_verification.json').write_text(json.dumps({
    'source_video_hashes_verified':540,'previous_training_clips_retained':162,'first_baseline_clips_retained':81,
    'new_training_clips':162,'class_counts':dict(class_counts),'subject_counts':dict(subject_counts),
    'repetition_counts':dict(repetition_counts),'all_training_subjects_present_per_class':True,
    'no_subject_overlap_between_splits':True,'selection_uses_predictions_or_features':False,
    'parent_manifest_sha256':parent_hash},indent=2),encoding='utf-8')
print('FULL_TRAINING_PREPARATION_COMPLETE',json.dumps({'clips':324,'per_class':18,'per_subject':54,
    'per_repetition':108,'retained_previous':162}),flush=True)
