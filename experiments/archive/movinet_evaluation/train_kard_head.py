"""Validation-only model selection for a frozen-backbone KARD linear head."""
import hashlib
import json
import time
from kard_model import tf,np,build_head,metrics
from kard_config import OUT,SEED,LABELS

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
manifest=json.loads((OUT/'manifest.json').read_text())
manifest_hash=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()

def load_split(split):
    selected=[r for r in manifest['records'] if r['selected'] and r['split']==split]
    arrays=[];labels=[]
    for r in selected:
        stem=split+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')
        info=json.loads((OUT/'feature_cache'/(stem+'.json')).read_text())
        assert info['manifest_sha256']==manifest_hash and info['video_sha256']==r['sha256']
        data=np.load(OUT/'feature_cache'/(stem+'.npz'))
        assert int(data['label'])==r['class_id'] and int(data['subject'])==r['subject_id']
        arrays.append(data['features']);labels.append(r['class_id'])
    return np.stack(arrays),np.asarray(labels,dtype=np.int32)

if __name__=='__main__':
    if (OUT/'selection.json').exists() or (OUT/'evaluation.json').exists():
        raise RuntimeError('An existing trained experiment is preserved; choose a new experiment directory.')
    x,y=load_split('train');vx,vy=load_split('validation')
    assert x.shape==(manifest['selected_counts']['train'],480) and vx.shape==(108,480)
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-3)
    np.savez(OUT/'train_normalization.npz',mean=mean,scale=scale)
    experiments=[]
    start=time.perf_counter()
    for number,l2 in enumerate([0.0001,0.001,0.01]):
        tf.keras.utils.set_random_seed(SEED)
        head=build_head(mean,scale,l2)
        initial=[v.numpy().copy() for v in head.trainable_variables]
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        candidate=OUT/f'candidate_l2_{l2:g}.keras'
        rng=np.random.default_rng(SEED)
        history=[];best_key=(-1.0,float('-inf'));best_epoch=0;stale=0

        @tf.function(reduce_retracing=True)
        def step(bx,by):
            with tf.GradientTape() as tape:
                logits=head(bx,training=True)
                loss=loss_fn(by,logits)+tf.add_n(head.losses)
            gradients=tape.gradient(loss,head.trainable_variables)
            for g in gradients:tf.debugging.assert_all_finite(g,'Nonfinite head gradient')
            optimizer.apply_gradients(zip(gradients,head.trainable_variables))
            return loss

        for epoch in range(1,201):
            permutation=rng.permutation(len(y));losses=[]
            for offset in range(0,len(y),16):
                ids=permutation[offset:offset+16]
                losses.append(float(step(x[ids],y[ids]).numpy()))
            train=metrics(head(x,training=False).numpy(),y)
            validation=metrics(head(vx,training=False).numpy(),vy)
            key=(validation['macro_f1'],-validation['cross_entropy'])
            entry={'epoch':epoch,'batch_loss_mean':float(np.mean(losses)),
                'train_accuracy':train['accuracy'],'train_macro_f1':train['macro_f1'],
                'validation_accuracy':validation['accuracy'],'validation_macro_f1':validation['macro_f1'],
                'validation_cross_entropy':validation['cross_entropy']}
            history.append(entry)
            if key>best_key:
                best_key=key;best_epoch=epoch;stale=0;head.save(candidate)
            else:stale+=1
            if epoch==1 or epoch%10==0:
                print(json.dumps({'l2':l2,**entry,'best_epoch':best_epoch}),flush=True)
            (OUT/f'history_l2_{l2:g}.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            if stale>=25:break
        delta=max(float(np.max(np.abs(v.numpy()-old))) for v,old in zip(head.trainable_variables,initial))
        assert delta>1e-5
        best=tf.keras.models.load_model(candidate)
        experiments.append({'l2':l2,'checkpoint':str(candidate),'best_epoch':best_epoch,
            'epochs_executed':len(history),'optimizer_steps':int(optimizer.iterations.numpy()),
            'maximum_trainable_weight_change':delta,
            'train':metrics(best(x,training=False).numpy(),y),
            'validation':metrics(best(vx,training=False).numpy(),vy)})
        print('CANDIDATE_COMPLETE',json.dumps({k:v for k,v in experiments[-1].items() if k not in ['train','validation']}),flush=True)
    winner=max(experiments,key=lambda e:(e['validation']['macro_f1'],-e['validation']['cross_entropy']))
    selected=tf.keras.models.load_model(winner['checkpoint'])
    selected.save(OUT/'kard_head.keras')
    selection={'manifest_sha256':manifest_hash,'labels':LABELS,
        'trainable_parameters':8658,'feature_dimension':480,'train_clips':len(y),'validation_clips':108,
        'selection_rule':'Highest validation macro F1; tie broken by lower validation cross entropy.',
        'test_used_for_training_or_selection':False,'head_checkpoint':'kard_head.keras',
        'winner':winner,'candidates':experiments,'training_seconds':time.perf_counter()-start,
        'tensorflow_version':tf.__version__,'device':'CPU'}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',json.dumps({'l2':winner['l2'],'best_epoch':winner['best_epoch'],
        'train_accuracy':winner['train']['accuracy'],'validation_accuracy':winner['validation']['accuracy'],
        'validation_macro_f1':winner['validation']['macro_f1']}),flush=True)
