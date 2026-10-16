"""Train/validation-only diagnosis and untrained architecture verification."""
import os
os.environ['KARD_TRAIN_FRACTION']='1.0'
import json
from pathlib import Path
import time
import numpy as np
from kard_model import FrozenBackbone,variable_digest,tf,metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'architecture_challenge_20261010'
OUT.mkdir(exist_ok=True)
BASE=ROOT/'kard_a0_head_100pct'
UPPER=ROOT/'kard_a0_upper_finetune'
manifest=json.loads((BASE/'manifest.json').read_text())
records=[r for r in manifest['records'] if r['split'] in ('train','validation')]
f=FrozenBackbone()
layers=[getattr(f.backbone,f'layer_with_weights-{i}') for i in (14,15)]
project=getattr(f.backbone,'layer_with_weights-16')
states=f.model.init_states(tf.constant([1,1,172,172,3]))

@tf.function(input_signature=[tf.TensorSpec([1,None,6,6,104],tf.float32)])
def extract(a):
    next_states=states
    for layer in layers:
        a,next_states=layer(a,states=next_states,training=False)
    # The archived inner _project has no callable restored signature. Use the
    # original complete head with one frame per independent batch element.
    per_frame=tf.transpose(a,[1,0,2,3,4])
    frame_states=f.model.init_states(tf.stack([tf.shape(a)[1],1,172,172,3]))
    projected,_=project(per_frame,states=frame_states,training=False)
    frame_features=tf.reshape(projected,[1,tf.shape(a)[1],480])
    # Diagnostic intervention ONLY: same pretrained projection and nonlinearity
    # but average spatially before them. This is not the compact trained model.
    spatial_mean=tf.reduce_mean(a,axis=[2,3],keepdims=True)
    spatial_mean=tf.broadcast_to(spatial_mean,tf.shape(a))
    pool_first,_=project(spatial_mean,states=states,training=False)
    return frame_features,tf.reshape(pool_first,[1,480])

def stem(r):
    return r['split']+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')

start=time.perf_counter()
frames=[];wide=[];reordered=[];labels=[];splits=[];errors=[]
for i,r in enumerate(records,1):
    source=UPPER/'activation_cache'/(stem(r)+'.npz')
    import hashlib
    info=json.loads(source.with_suffix('.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==info['cache_sha256']
    assert info['source_sha256']==r['sha256']
    with np.load(source) as data:
        activation=data['activation']
    ff,pf=extract(tf.convert_to_tensor(activation[None]))
    ff=ff.numpy()[0]
    with np.load(BASE/'feature_cache'/(stem(r)+'.npz')) as data:
        original=data['features']
    error=float(np.max(np.abs(ff.mean(0)-original)))
    assert error<1e-4
    errors.append(error)
    frames.append(ff);wide.append(original);reordered.append(pf.numpy()[0])
    labels.append(r['class_id']);splits.append(r['split'])
    if i%50==0:
        print('DIAGNOSIS',i,'/',len(records),flush=True)
assert variable_digest(f.backbone.variables)==f.initial_hash
wide=np.stack(wide);reordered=np.stack(reordered);labels=np.asarray(labels);splits=np.asarray(splits)
np.savez_compressed(OUT/'train_validation_features.npz',wide=wide,reordered=reordered,labels=labels,splits=splits,
    lengths=np.asarray([len(x) for x in frames]),frames=np.concatenate(frames))
head=tf.keras.models.load_model(BASE/'kard_head.keras')
result={'clips':len(records),'test_accessed':False,'backbone_unchanged':True,
    'per_frame_to_original_pooled_max_error':max(errors),
    'diagnostic':'Same pretrained projection and old classifier, changing spatial-pooling order only; not the trained B/C models.',
    'hypothesis_limit':'A change in representation/performance cannot by itself prove why a separately trained compact model failed.',
    'splits':{},'seconds':time.perf_counter()-start}
for split in ('train','validation'):
    ids=splits==split
    a=head(wide[ids],training=False).numpy();b=head(reordered[ids],training=False).numpy()
    result['splits'][split]={'original':metrics(a,labels[ids]),'spatial_pool_before_projection':metrics(b,labels[ids]),
        'prediction_agreement':float((a.argmax(1)==b.argmax(1)).mean()),
        'mean_absolute_standardized_feature_difference':float(np.abs((wide[ids]-reordered[ids])/np.load(BASE/'train_normalization.npz')['scale']).mean())}
(OUT/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='splits'},indent=2),flush=True)
print(json.dumps({s:{'original_accuracy':v['original']['accuracy'], 'reordered_accuracy':v['spatial_pool_before_projection']['accuracy'],
    'prediction_agreement':v['prediction_agreement'],'feature_difference':v['mean_absolute_standardized_feature_difference']} for s,v in result['splits'].items()},indent=2),flush=True)
