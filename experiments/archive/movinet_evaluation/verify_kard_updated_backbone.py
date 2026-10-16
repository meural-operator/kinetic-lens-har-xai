"""Verify actual rejected training weights, using only a training clip."""
from kard_finetune import *

manifest=json.loads((OUT/'manifest.json').read_text());r=next(r for r in manifest['records'] if r['split']=='train')
tuner=UpperTuner();images,_=read_clip(r['path'])
ep=tuner.full_endpoints(images);initial=tuner.head(tf.reshape(ep['head'],[1,480]),training=False).numpy()
with np.load(OUT/'activation_cache'/(cache_stem(r)+'.npz')) as data:activation=tf.constant(data['activation'][None])
tuner.restore_weights(OUT/'last_weights.npz')
full_ep=tuner.full_endpoints(images);full=tuner.head(tf.reshape(full_ep['head'],[1,480]),training=False).numpy()
partial=tuner.predict(activation).numpy()
error=float(np.max(np.abs(full-partial)));change=float(np.max(np.abs(full-initial)))
assert error<1e-4 and change>1e-5
tuner.verify_frozen()
groups={prefix:max(float(np.max(np.abs(v.numpy()-old))) for v,old in zip(tuner.upper,tuner.initial_upper) if v.name.startswith(prefix)) for prefix in PREFIXES}
assert all(v>1e-7 for v in groups.values())
check={'video':r['relative_path'],'split':'train','checkpoint':'last_weights.npz',
    'actual_trained_upper_vs_full_max_logit_error':error,'changed_full_model_logits_vs_initial_max_difference':change,
    'actual_trained_weight_changes_by_group':groups,'frozen_lower_and_bn_unchanged':True,
    'test_used':False,'purpose':'Verify that training changes the complete backbone inference, even when validation rejects that checkpoint.'}
(OUT/'updated_backbone_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
print('UPDATED_BACKBONE_VERIFIED',json.dumps(check),flush=True)
