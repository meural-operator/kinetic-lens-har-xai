"""Prove activation-cache equivalence and gradients before fine-tuning."""
import json
import time
from kard_finetune import *

tf.keras.utils.set_random_seed(42)
tuner=UpperTuner()
manifest=json.loads((BASE/'manifest.json').read_text())
r=next(r for r in manifest['records'] if r['split']=='train')
images,_=read_clip(r['path'])
ep=tuner.full_endpoints(images);x=ep['b4/l1']
start=time.perf_counter()
pooled=tuner.pooled(x);error=float(np.max(np.abs(pooled.numpy()-tf.reshape(ep['head'],[1,480]).numpy())))
assert error<1e-4
head_vars=[v.value for v in tuner.head.trainable_variables]
variables=tuner.upper+head_vars
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(variables)
    logits=tuner.head(tuner.pooled(x),training=False)
    loss=tf.keras.losses.sparse_categorical_crossentropy(tf.constant([r['class_id']]),logits,from_logits=True)[0]
grads=tape.gradient(loss,variables)
assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
assert all(float(tf.linalg.global_norm([g]).numpy())>0 for g in grads[:len(tuner.upper)])
tuner.verify_frozen()
batch=tf.concat([x,x],axis=0)
assert np.max(np.abs(tuner.predict(batch).numpy()-tuner.predict(x).numpy()))<1e-4
check={'video':r['relative_path'],'activation_shape':x.shape.as_list(),
    'cached_upper_vs_full_max_feature_error':error,'all_upper_gradients_finite_and_nonzero':True,
    'equal_length_batch_equivalence_passed':True,'frozen_and_bn_hash_checks_passed':True,
    'upper_trainable_variables':len(tuner.upper),'upper_trainable_parameters':sum(int(np.prod(v.shape)) for v in tuner.upper),
    'gradient_check_seconds':time.perf_counter()-start,'upper_variables':[v.name for v in tuner.upper],
    'bn_policy':'All batch-normalization variables frozen; backbone layer calls in inference mode.'}
(OUT/'gradient_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
print('UPPER_GRADIENT_CHECK_PASSED',json.dumps(check),flush=True)
