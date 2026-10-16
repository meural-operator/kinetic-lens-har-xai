"""Cache immutable frozen features only for the selected split videos."""
import argparse
import hashlib
import json
import time
from kard_config import ROOT,OUT,PROTOCOL
from kard_model import FrozenBackbone,read_clip,variable_digest,np

parser=argparse.ArgumentParser()
parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation'])
args=parser.parse_args()
manifest=json.loads((OUT/'manifest.json').read_text())
manifest_hash=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
model_hash=hashlib.sha256((ROOT/'models'/'a0_stream'/'variables'/'variables.data-00000-of-00001').read_bytes()).hexdigest()
selected=[r for r in manifest['records'] if r['selected'] and r['split'] in args.splits]
if 'test' in args.splits and not (OUT/'selection.json').exists():
    raise RuntimeError('Select head with validation before extracting held-out test features')
if 'test' in args.splits and PROTOCOL.get('controlled_sampling_comparison') and not (OUT/'sampling_selection.json').exists():
    raise RuntimeError('Select the sampling policy using validation before extracting 10-fps test features')
features=FrozenBackbone()
cache=OUT/'feature_cache';cache.mkdir(exist_ok=True)
start=time.perf_counter()
partition_verified=False
for number,record in enumerate(selected,1):
    stem=record['split']+'_'+record['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')
    npz=cache/(stem+'.npz');info_path=cache/(stem+'.json')
    expected={'manifest_sha256':manifest_hash,'video_sha256':record['sha256'],'model_sha256':model_hash}
    if npz.exists() and info_path.exists():
        info=json.loads(info_path.read_text())
        if all(info.get(k)==v for k,v in expected.items()):
            assert np.load(npz)['features'].shape==(480,)
            continue
    assert hashlib.sha256(__import__('pathlib').Path(record['path']).read_bytes()).hexdigest()==record['sha256']
    images,metadata=read_clip(record['path'])
    if PROTOCOL.get('controlled_sampling_comparison'):
        baseline_info=json.loads((ROOT/PROTOCOL['baseline_experiment']/'feature_cache'/(stem+'.json')).read_text())
        assert baseline_info['video_sha256']==record['sha256'] and baseline_info['model_sha256']==model_hash
        assert abs(metadata['window_start_seconds']-baseline_info['window_start_seconds'])<1e-10
        assert metadata['frame_indices'][::2]==baseline_info['frame_indices'],'5-fps timestamps must be a subset of the 10-fps timestamps'
        metadata['same_window_as_5fps_verified']=True
        metadata['5fps_indices_match_every_other_10fps_frame']=True
    vector,states=features.extract(images)
    if not partition_verified:
        whole,_=features.extract(images,chunk=int(images.shape[1]))
        error=float(np.max(np.abs(vector-whole)))
        assert error<1e-4
        partition_verified=True
        print('FEATURE_PARTITION_MAX_ERROR',error,flush=True)
        (OUT/'feature_partition_check.json').write_text(json.dumps({'max_error':error,
            'video':record['relative_path'],'sampled_frames':metadata['sampled_frames']}),encoding='utf-8')
    np.savez_compressed(npz,features=vector,label=record['class_id'],subject=record['subject_id'])
    info={**expected,**metadata,'video':record['relative_path'],'split':record['split'],
          'feature_dimension':480,'state_head_frame_count':int(states['state/head/pool_frame_count'].numpy()[0])}
    info_path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    if number%10==0 or number==len(selected):
        print(f'FEATURES {number}/{len(selected)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
after_hash=variable_digest(features.backbone.variables)
assert after_hash==features.initial_hash
(OUT/('feature_verification_'+'_'.join(args.splits)+'.json')).write_text(json.dumps({
    'backbone_variables_before':features.initial_hash,'backbone_variables_after':after_hash,
    'backbone_unchanged':True,'manifest_sha256':manifest_hash,'model_sha256':model_hash,
    'selected_videos':len(selected),'splits':args.splits,'seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
print('FEATURE_EXTRACTION_COMPLETE',flush=True)
