"""Freeze the 10-fps protocol and retain every original clip and subject split."""
import hashlib
import json
import shutil
from pathlib import Path
from kard_config import ROOT,OUT,PROTOCOL,FRACTION,TARGET_FPS

assert FRACTION==1.0 and TARGET_FPS==10.0
if (OUT/'manifest.json').exists():raise RuntimeError('Existing manifest is preserved')
BASE=ROOT/'kard_a0_head_100pct'
parent=json.loads((BASE/'manifest.json').read_text())
assert parent['selected_counts']=={'train':324,'validation':108,'test':108}
assert parent['protocol']['max_frames']/parent['protocol']['target_fps']==PROTOCOL['max_frames']/PROTOCOL['target_fps']==10
for key in ['model','resolution','stream_chunk_frames','head','training','seed','selection']:
    assert parent['protocol'][key]==PROTOCOL[key],key
for r in parent['records']:
    assert r['selected'] and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
parent_hash=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()
manifest={**parent,'protocol':PROTOCOL,'parent_manifest_sha256':parent_hash,
    'sampling_comparison_parent':'kard_a0_head_100pct','records':parent['records']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
for name in ['dataset_audit.json','dataset_full_decode_audit.json']:
    shutil.copy2(BASE/name,OUT/name)
(OUT/'sampling_protocol_verification.json').write_text(json.dumps({'source_hashes_verified':540,
    'same_training_validation_test_clips':True,'same_subject_split':True,'same_ten_second_window':True,
    'same_head_architecture_training_and_seed':True,'new_head_from_initialization':True,
    '5fps_features_reused_for_10fps_training':False,'baseline_manifest_sha256':parent_hash,
    'previous_test_results_already_observed':True},indent=2),encoding='utf-8')
print('SAMPLING_PROTOCOL_FROZEN',json.dumps({'fps':10,'max_frames':100,'train':324,'validation':108,'test':108}),flush=True)
