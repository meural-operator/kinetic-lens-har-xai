"""One held-out test evaluation and a verified end-to-end SavedModel export."""
import csv
import hashlib
import json
from pathlib import Path
from kard_config import OUT,LABELS,PROTOCOL
from kard_model import FrozenBackbone,KardInference,tf,np,metrics,read_clip,variable_digest
from train_kard_head import load_split,manifest_hash,manifest

if (OUT/'evaluation.json').exists():
    raise RuntimeError('Test evaluation already exists. Preserve the first result; use a new experiment directory for a new protocol.')
selection=json.loads((OUT/'selection.json').read_text())
assert selection['manifest_sha256']==manifest_hash
head=tf.keras.models.load_model(OUT/'kard_head.keras')
x,y=load_split('test')
assert x.shape==(108,480)
logits=head(x,training=False).numpy()
result=metrics(logits,y)
per_subject={}
test_records=[r for r in manifest['records'] if r['selected'] and r['split']=='test']
for subject in manifest['split_subjects']['test']:
    ids=[i for i,r in enumerate(test_records) if r['subject_id']==subject]
    per_subject[str(subject)]=metrics(logits[ids],y[ids])

features=FrozenBackbone()
module=KardInference(features,head)
sample=test_records[0]
images,metadata=read_clip(sample['path'])
output=module.classify_clip(images)
direct=output['logits'].numpy()
error=float(np.max(np.abs(direct-logits[0:1])))
assert error<1e-3
tf.saved_model.save(module,str(OUT/'saved_model'),signatures={'serving_default':module.classify_clip})
restored=tf.saved_model.load(str(OUT/'saved_model'))
again=restored.signatures['serving_default'](image=images)['logits'].numpy()
export_error=float(np.max(np.abs(again-direct)))
assert export_error<1e-5 and int(again.argmax())==int(logits[0].argmax())
assert variable_digest(features.backbone.variables)==features.initial_hash
predictions=[]
prob=tf.nn.softmax(logits).numpy()
for i,r in enumerate(test_records):
    prediction=int(logits[i].argmax())
    predictions.append({'video':r['relative_path'],'subject_id':r['subject_id'],
        'true_class_id':r['class_id'],'true_label':r['class_name'],
        'predicted_class_id':prediction,'predicted_label':LABELS[prediction],
        'correct':prediction==r['class_id'],'softmax_probability':float(prob[i,prediction]),
        'top5':[{'class_id':int(j),'label':LABELS[j],'probability':float(prob[i,j])}
                for j in np.argsort(prob[i])[-5:][::-1]]})
with (OUT/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
    fields=[k for k in predictions[0] if k!='top5']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in predictions)
(OUT/'test_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'test_logits.npz',logits=logits,labels=y)
evaluation={'test':result,'test_per_subject':per_subject,
    'selection':{'l2':selection['winner']['l2'],'epoch':selection['winner']['best_epoch'],
        'validation_accuracy':selection['winner']['validation']['accuracy'],
        'validation_macro_f1':selection['winner']['validation']['macro_f1']},
    'export_verification':{'video':sample['relative_path'],'cached_vs_end_to_end_max_logit_error':error,
        'export_reload_max_logit_error':export_error,'saved_model_reload_passed':True,
        'backbone_weights_unchanged':True},'test_used_for_model_selection':False,
    'head_sha256':hashlib.sha256((OUT/'kard_head.keras').read_bytes()).hexdigest(),
    'manifest_sha256':manifest_hash,'protocol':PROTOCOL,
    'previous_test_results_already_observed':PROTOCOL.get('previous_test_results_already_observed',False),
    'limitation':'Single subject-disjoint split; not cross-validation or a published benchmark result.'}
(OUT/'evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
(OUT/'labels.json').write_text(json.dumps(LABELS,indent=2),encoding='utf-8')
print('EVALUATION_COMPLETE',json.dumps({'test_accuracy':result['accuracy'],'test_macro_f1':result['macro_f1'],
    'n':result['n'],'export_verification':evaluation['export_verification']}),flush=True)
