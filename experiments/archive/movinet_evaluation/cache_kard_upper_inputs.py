"""Cache only the immutable lower-backbone activations, with source provenance."""
import argparse
import time
from kard_finetune import *

parser=argparse.ArgumentParser();parser.add_argument('--splits',nargs='+',default=['train','validation']);args=parser.parse_args()
if 'test' in args.splits:assert (OUT/'selection.json').exists(),'Select a checkpoint before test caching'
protocol=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
mh=sha(OUT/'manifest.json');ph=sha(OUT/'protocol.json')
assert sha(BASE/'manifest.json')==mh==protocol['parent_manifest_sha256']
assert sha(ROOT/'models'/'a0_stream'/'variables'/'variables.data-00000-of-00001')==protocol['original_backbone_file_sha256']
tuner=UpperTuner();cache=OUT/'activation_cache';cache.mkdir(exist_ok=True)
@tf.function(input_signature=[tf.TensorSpec([1,None,172,172,3],tf.float32)])
def extract(images):
    ep=tuner.full_endpoints(images)
    return ep['b4/l1'],tf.reshape(ep['head'],[1,480])
start=time.perf_counter();processed=0;reused=0;errors=[];shape_counts={}
records=[r for r in manifest['records'] if r['split'] in args.splits]
for i,r in enumerate(records):
    path=cache/(cache_stem(r)+'.npz');info_path=path.with_suffix('.json')
    if path.exists() and info_path.exists():
        info=json.loads(info_path.read_text())
        assert info['source_sha256']==r['sha256'] and info['manifest_sha256']==mh and info['protocol_sha256']==ph
        assert info['cache_sha256']==sha(path)
        reused+=1;continue
    assert sha(r['path'])==r['sha256']
    images,metadata=read_clip(r['path']);activation,original_pooled=extract(images)
    a=activation.numpy()[0];assert a.shape[1:]==(6,6,104) and np.isfinite(a).all()
    if processed<3:
        upper=tuner.pooled(activation).numpy()
        error=float(np.max(np.abs(upper-original_pooled.numpy())));assert error<1e-4
        errors.append(error)
        baseline_cache=BASE/'feature_cache'/(cache_stem(r)+'.npz')
        with np.load(baseline_cache) as data:
            assert np.max(np.abs(original_pooled.numpy()[0]-data['features']))<1e-4
    np.savez_compressed(path,activation=a,label=np.int32(r['class_id']),subject=np.int32(r['subject_id']))
    info={'video':r['relative_path'],'source_sha256':r['sha256'],'manifest_sha256':mh,'protocol_sha256':ph,
          'original_backbone_file_sha256':protocol['original_backbone_file_sha256'],'cache_sha256':sha(path),
          'frozen_variable_digest':tuner.frozen_hash,'activation_shape':list(a.shape),'metadata':metadata}
    info_path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    processed+=1;shape_counts[str(a.shape[0])]=shape_counts.get(str(a.shape[0]),0)+1
    if (i+1)%25==0:print('ACTIVATIONS',i+1,'/',len(records),'seconds',round(time.perf_counter()-start,1),flush=True)
tuner.verify_frozen()
verification={'manifest_sha256':mh,'protocol_sha256':ph,'splits':args.splits,'clips':len(records),
    'created':processed,'reused':reused,'lower_and_bn_unchanged':True,'cached_full_equivalence_max_errors':errors,
    'seconds':time.perf_counter()-start,'frame_count_distribution_new':shape_counts}
(OUT/('activation_verification_'+'_'.join(args.splits)+'.json')).write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('ACTIVATION_CACHE_COMPLETE',json.dumps(verification),flush=True)
