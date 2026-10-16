"""Evaluate the validation-selected partial-backbone model and verify export."""
import csv
from kard_finetune import *
from kard_model import KardInference

if (OUT/'evaluation.json').exists():raise RuntimeError('Preserve the first test evaluation')
protocol=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
manifest=json.loads((OUT/'manifest.json').read_text());mh=sha(OUT/'manifest.json');ph=sha(OUT/'protocol.json')
assert selection['manifest_sha256']==mh and selection['protocol_sha256']==ph
assert not selection['test_used_for_training_or_selection']
assert selection['selected_weights_sha256']==sha(OUT/'selected_weights.npz')
tuner=UpperTuner();tuner.restore_weights(OUT/'selected_weights.npz');tuner.verify_frozen()
records=[r for r in manifest['records'] if r['split']=='test'];assert len(records)==108
for r in records:
    path=OUT/'activation_cache'/(cache_stem(r)+'.npz');info=json.loads(path.with_suffix('.json').read_text())
    assert info['source_sha256']==r['sha256'] and info['protocol_sha256']==ph and info['manifest_sha256']==mh
    assert info['cache_sha256']==sha(path)
    with np.load(path) as data:assert int(data['label'])==r['class_id'] and int(data['subject'])==r['subject_id']
logits=predict_records(tuner,records);y=np.asarray([r['class_id'] for r in records]);result=metrics(logits,y)
per_subject={}
for s in manifest['split_subjects']['test']:
    ids=[i for i,r in enumerate(records) if r['subject_id']==s]
    per_subject[str(s)]=metrics(logits[ids],y[ids])
module=KardInference(tuner.features,tuner.head)
probes=[];full_outputs=[]
for i in [0,36,107]:
    images,metadata=read_clip(records[i]['path'])
    direct=module.classify_clip(images)['logits'].numpy()
    difference=float(np.max(np.abs(direct-logits[i:i+1])))
    assert difference<1e-3 and int(direct.argmax())==int(logits[i].argmax())
    full_outputs.append((images,direct))
    probes.append({'video':records[i]['relative_path'],'frames':metadata['sampled_frames'],
                   'cached_upper_vs_full_max_logit_error':difference})
tf.saved_model.save(module,str(OUT/'saved_model'),signatures={'serving_default':module.classify_clip})
restored=tf.saved_model.load(str(OUT/'saved_model'))
for probe,(images,direct) in zip(probes,full_outputs):
    again=restored.signatures['serving_default'](image=images)['logits'].numpy()
    error=float(np.max(np.abs(again-direct)));assert error<1e-5
    probe['export_reload_max_logit_error']=error
tuner.verify_frozen()
backbone_digest=variable_digest(tuner.backbone.variables)
assert sha(BASE/'kard_head.keras')==protocol['initial_head_sha256']
assert sha(ROOT/'models'/'a0_stream'/'variables'/'variables.data-00000-of-00001')==protocol['original_backbone_file_sha256']
prob=tf.nn.softmax(logits).numpy();predictions=[]
for i,r in enumerate(records):
    pred=int(logits[i].argmax())
    predictions.append({'video':r['relative_path'],'subject_id':r['subject_id'],'true_class_id':r['class_id'],
        'true_label':r['class_name'],'predicted_class_id':pred,'predicted_label':LABELS[pred],
        'correct':pred==r['class_id'],'softmax_probability':float(prob[i,pred]),
        'top5':[{'class_id':int(j),'label':LABELS[j],'probability':float(prob[i,j])} for j in np.argsort(prob[i])[-5:][::-1]]})
with (OUT/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
    fields=[k for k in predictions[0] if k!='top5'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    w.writerows({k:p[k] for k in fields} for p in predictions)
(OUT/'test_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'test_logits.npz',logits=logits,labels=y)
evaluation={'test':result,'test_per_subject':per_subject,'selected_epoch':selection['selected_epoch'],
    'validation':selection['validation'],'protocol':protocol,'protocol_sha256':ph,'manifest_sha256':mh,
    'test_used_for_training_or_selection':False,'previous_test_results_already_observed':True,
    'export_verification':{'probes':probes,'saved_model_reload_passed':True,'frozen_lower_and_bn_unchanged':True,
        'upper_backbone_updated':selection['maximum_backbone_weight_change_selected']>0,
        'selected_backbone_variable_digest':backbone_digest,'original_downloaded_weights_preserved':True},
    'limitation':protocol['limitations']}
(OUT/'evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
(OUT/'labels.json').write_text(json.dumps(LABELS,indent=2),encoding='utf-8')
print('FINETUNE_EVALUATION_COMPLETE',json.dumps({'test_accuracy':result['accuracy'],'test_macro_f1':result['macro_f1'],
    'correct':int(np.trace(np.asarray(result['confusion_matrix']))),'n':108,'selected_epoch':selection['selected_epoch'],'export_probes':probes}),flush=True)
