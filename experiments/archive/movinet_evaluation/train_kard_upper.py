"""Train upper backbone kernels and the initialized head; select on validation."""
from collections import defaultdict
import time
from kard_finetune import *

if (OUT/'selection.json').exists():raise RuntimeError('Preserve the existing selected experiment')
if (OUT/'history.json').exists():raise RuntimeError('Interrupted history exists; inspect before restarting')
protocol=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
assert protocol['learning_rate']==1e-5 and protocol['maximum_epochs']==20
tf.keras.utils.set_random_seed(protocol['seed']);tf.config.experimental.enable_op_determinism()
tuner=UpperTuner()
mh=sha(OUT/'manifest.json');ph=sha(OUT/'protocol.json')
assert sha(BASE/'kard_head.keras')==protocol['initial_head_sha256']
records={s:[r for r in manifest['records'] if r['split']==s] for s in ['train','validation']}
arrays={};labels={};buckets={}
for split,rs in records.items():
    aa=[];yy=[];bb=defaultdict(list)
    for i,r in enumerate(rs):
        path=OUT/'activation_cache'/(cache_stem(r)+'.npz');info=json.loads(path.with_suffix('.json').read_text())
        assert info['source_sha256']==r['sha256'] and info['manifest_sha256']==mh and info['protocol_sha256']==ph
        assert info['cache_sha256']==sha(path)
        with np.load(path) as data:
            assert int(data['label'])==r['class_id'] and int(data['subject'])==r['subject_id']
            a=data['activation'].copy();assert np.isfinite(a).all();aa.append(a);yy.append(r['class_id']);bb[len(a)].append(i)
    arrays[split]=aa;labels[split]=np.asarray(yy,np.int32);buckets[split]=bb
assert len(labels['train'])==324 and len(labels['validation'])==108

def batches(split,rng=None):
    result=[]
    for ids in buckets[split].values():
        ids=np.asarray(ids)
        if rng is not None:ids=rng.permutation(ids)
        result.extend(ids[o:o+8] for o in range(0,len(ids),8))
    if rng is not None:rng.shuffle(result)
    return result

def evaluate(split):
    logits=np.zeros((len(labels[split]),18),np.float32)
    for ids in batches(split):
        x=np.stack([arrays[split][int(i)] for i in ids])
        logits[ids]=tuner.predict(x).numpy()
    return metrics(logits,labels[split])

head_resources=[v.value for v in tuner.head.trainable_variables]
resources=tuner.upper+head_resources
opt_variables=tuner.upper+list(tuner.head.trainable_variables)
optimizer=tf.keras.optimizers.Adam(learning_rate=protocol['learning_rate'])
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
@tf.function(input_signature=[tuner.input_spec,tf.TensorSpec([None],tf.int32)])
def step(x,y):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(resources)
        logits=tuner.head(tuner.pooled(x),training=True)
        loss=loss_fn(y,logits)+tf.add_n(tuner.head.losses)
    grads=tape.gradient(loss,resources)
    for g in grads:
        if g is None:raise ValueError('An explicitly selected variable has no gradient')
        tf.debugging.assert_all_finite(g,'Nonfinite fine-tuning gradient')
    clipped,norm=tf.clip_by_global_norm(grads,protocol['gradient_global_norm_clip'])
    optimizer.apply_gradients(zip(clipped,opt_variables))
    return loss,norm

start=time.perf_counter();train=evaluate('train');validation=evaluate('validation')
baseline=json.loads((BASE/'selection.json').read_text())['winner']['validation']
assert abs(validation['accuracy']-baseline['accuracy'])<1e-12 and abs(validation['macro_f1']-baseline['macro_f1'])<1e-12
best_key=(validation['macro_f1'],-validation['cross_entropy']);best_epoch=0;stale=0
best={'epoch':0,'train':train,'validation':validation};history=[];gradient_norms=[]
tuner.save_weights(OUT/'selected_weights.npz')
print('INITIAL_VALIDATION',json.dumps(validation),flush=True)
for epoch in range(1,protocol['maximum_epochs']+1):
    epoch_start=time.perf_counter();rng=np.random.default_rng(protocol['seed']+epoch)
    sample_count=0;loss_total=0.;max_norm=0.;seen=[]
    for ids in batches('train',rng):
        x=np.stack([arrays['train'][int(i)] for i in ids])
        loss,norm=step(x,labels['train'][ids]);seen.extend(int(i) for i in ids)
        loss_total+=float(loss.numpy())*len(ids);sample_count+=len(ids);max_norm=max(max_norm,float(norm.numpy()))
    assert sample_count==324 and sorted(seen)==list(range(324))
    train=evaluate('train');validation=evaluate('validation');tuner.verify_frozen()
    key=(validation['macro_f1'],-validation['cross_entropy'])
    if key>best_key:
        best_key=key;best_epoch=epoch;stale=0;best={'epoch':epoch,'train':train,'validation':validation}
        tuner.save_weights(OUT/'selected_weights.npz')
    else:stale+=1
    entry={'epoch':epoch,'train':train,'validation':validation,'mean_regularized_loss':loss_total/sample_count,
           'max_gradient_global_norm_before_clipping':max_norm,'optimizer_steps_total':int(optimizer.iterations.numpy()),
           'clips_seen':sample_count,'selected_epoch_so_far':best_epoch,'seconds':time.perf_counter()-epoch_start}
    history.append(entry)
    (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    tuner.save_weights(OUT/'last_weights.npz')
    print('FINETUNE_EPOCH',json.dumps({k:v for k,v in entry.items() if k not in ['train','validation']}|{
          'train_accuracy':train['accuracy'],'validation_accuracy':validation['accuracy'],'validation_macro_f1':validation['macro_f1'],'validation_ce':validation['cross_entropy']}),flush=True)
    if stale>=protocol['early_stopping_patience']:break
last_delta=max(float(np.max(np.abs(v.numpy()-old))) for v,old in zip(tuner.upper,tuner.initial_upper))
assert last_delta>1e-7,'No backbone learning occurred'
tuner.restore_weights(OUT/'selected_weights.npz');tuner.verify_frozen()
selected_delta=max(float(np.max(np.abs(v.numpy()-old))) for v,old in zip(tuner.upper,tuner.initial_upper))
group_deltas={prefix:max(float(np.max(np.abs(v.numpy()-old))) for v,old in zip(tuner.upper,tuner.initial_upper) if v.name.startswith(prefix)) for prefix in PREFIXES}
if best_epoch>0:assert all(delta>1e-7 for delta in group_deltas.values())
selected_validation=evaluate('validation');assert abs(selected_validation['macro_f1']-best['validation']['macro_f1'])<1e-12
tuner.head.save(OUT/'kard_head.keras')
selection={'protocol_sha256':ph,'manifest_sha256':mh,'initial_head_sha256':protocol['initial_head_sha256'],
    'selected_epoch':best_epoch,'selected_checkpoint':'selected_weights.npz','selected_weights_sha256':sha(OUT/'selected_weights.npz'),
    'train':best['train'],'validation':selected_validation,'initial_validation':baseline,
    'epochs_executed':len(history),'optimizer_steps':int(optimizer.iterations.numpy()),
    'train_clips':324,'validation_clips':108,'test_used_for_training_or_selection':False,
    'learning_rate':1e-5,'maximum_backbone_weight_change_last':last_delta,'maximum_backbone_weight_change_selected':selected_delta,
    'upper_backbone_trainable_parameters':protocol['upper_backbone_trainable_parameters'],'head_trainable_parameters':8658,
    'frozen_and_bn_variables_unchanged':True,'frozen_variable_digest':tuner.frozen_hash,'batch_norm_variable_digest':tuner.bn_hash,
    'upper_variable_names':[v.name for v in tuner.upper],'training_selection_seconds':time.perf_counter()-start,
    'selected_weight_changes_by_layer_group':group_deltas,
    'selection_rule':protocol['checkpoint_selection'],'device':'CPU','tensorflow_version':tf.__version__,
    'selected_starting_baseline':best_epoch==0}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
print('FINETUNE_SELECTION_COMPLETE',json.dumps({k:v for k,v in selection.items() if k not in ['train','validation','initial_validation','upper_variable_names']}),flush=True)
