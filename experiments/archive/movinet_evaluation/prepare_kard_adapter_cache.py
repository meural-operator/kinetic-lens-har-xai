"""Replay original final blocks to cache unpooled b4/l3 for projection adaptation.

Test preparation is forbidden until this experiment's selections are locked.
"""
import os
os.environ['KARD_TRAIN_FRACTION']='1.0'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'kard_a0_projection_adapter_v1'
BASE=ROOT/'kard_a0_head_100pct'
UPPER=ROOT/'kard_a0_upper_finetune'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def test_gate():
    gate=json.loads((OUT/'selection_manifest.json').read_text())
    assert gate['all_selections_locked'] and not gate['test_used_for_selection']
    assert gate['protocol_sha256']==sha(OUT/'protocol.json')
    assert set(gate['selections'])=={f'{v}_seed_{s}' for v in ('lora','guided','random') for s in (42,123,2026)}
    for item in gate['selections'].values():assert sha(OUT/item['checkpoint'])==item['checkpoint_sha256']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train_validation','test'],default='train_validation')
    args=parser.parse_args()
    # Gate before selecting records, inspecting test files or importing TF.
    if args.split=='test':test_gate()
    from kard_model import FrozenBackbone, variable_digest, tf
    OUT.mkdir(exist_ok=True);cache=OUT/'activation_cache';cache.mkdir(exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text())
    splits=('train','validation') if args.split=='train_validation' else ('test',)
    records=[r for r in manifest['records'] if r['split'] in splits and r['selected']]
    f=FrozenBackbone();states=f.model.init_states(tf.constant([1,1,172,172,3]))
    layers=[getattr(f.backbone,f'layer_with_weights-{i}') for i in (14,15)]
    project=getattr(f.backbone,'layer_with_weights-16')
    @tf.function(input_signature=[tf.TensorSpec([1,None,6,6,104],tf.float32)])
    def extract(a):
        next_states=states
        for layer in layers:a,next_states=layer(a,states=next_states,training=False)
        pooled,_=project(a,states=next_states,training=False)
        return a,tf.reshape(pooled,[480])
    projection_path=OUT/'original_projection.npz'
    values={v.name:v.numpy() for v in project.variables}
    projection={'kernel':next(v for k,v in values.items() if 'kernel' in k).reshape(104,480)}
    for k in ('gamma','beta','moving_mean','moving_variance'):
        projection[k]=next(v for name,v in values.items() if name.endswith('/'+k+':0'))
    if projection_path.exists():
        with np.load(projection_path) as previous:
            assert all(np.array_equal(previous[k],v) for k,v in projection.items())
    else:np.savez(projection_path,**projection)
    files={};start=time.perf_counter()
    for index,r in enumerate(records,1):
        name=r['split']+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')
        source=UPPER/'activation_cache'/(name+'.npz')
        info=json.loads(source.with_suffix('.json').read_text())
        assert info['manifest_sha256']==sha(BASE/'manifest.json')
        assert info['original_backbone_file_sha256']==sha(ROOT/'models/a0_stream/variables/variables.data-00000-of-00001')
        assert info['protocol_sha256']==sha(UPPER/'protocol.json')
        assert sha(source)==info['cache_sha256'] and info['source_sha256']==r['sha256']==sha(r['path'])
        target=cache/(name+'.npz');sidecar=target.with_suffix('.json')
        expected={'source_activation_sha256':sha(source),'video_sha256':r['sha256'],
            'manifest_sha256':sha(BASE/'manifest.json'),'projection_sha256':sha(projection_path),
            'model_sha256':info['original_backbone_file_sha256'],'endpoint':'b4/l3_unpooled',
            'class_id':r['class_id'],'subject_id':r['subject_id'],
            'frame_indices':info['metadata']['frame_indices'] if 'metadata' in info else info['frame_indices']}
        if target.exists() and sidecar.exists():
            saved=json.loads(sidecar.read_text())
            assert all(saved[k]==v for k,v in expected.items())
            assert saved['cache_sha256']==sha(target)
        else:
            with np.load(source) as data:activation=data['activation']
            late,pooled=extract(tf.constant(activation[None]))
            late=late.numpy()[0];pooled=pooled.numpy()
            with np.load(BASE/'feature_cache'/(name+'.npz')) as data:wide=data['features'].copy()
            error=float(np.abs(pooled-wide).max());assert error<1e-4
            assert late.shape[1:]==(6,6,104) and np.isfinite(late).all()
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,activation=late,wide=wide,label=r['class_id'],subject=r['subject_id'])
            temporary.replace(target)
            saved={**expected,'cache_sha256':sha(target),'shape':list(late.shape),
                   'original_pooled_max_error':error}
            write_json(sidecar,saved)
        files[name]=saved
        if index%50==0:print('CACHE',index,'/',len(records),flush=True)
    assert variable_digest(f.backbone.variables)==f.initial_hash
    for split in splits:
        inventory={k:v for k,v in files.items() if k.startswith(split+'_')}
        assert len(inventory)=={'train':324,'validation':108,'test':108}[split]
        write_json(OUT/f'cache_inventory_{split}.json',{'files':inventory,'backbone_unchanged':True})
    write_json(OUT/f'cache_preparation_{args.split}.json',{'clips':len(records),
        'test_gate_checked':args.split=='test','backbone_unchanged':True,'seconds':time.perf_counter()-start,
        'projection_sha256':sha(projection_path),'preparer_sha256':sha(__file__)})
    print('CACHE_READY',args.split,len(records),flush=True)

if __name__=='__main__':main()
