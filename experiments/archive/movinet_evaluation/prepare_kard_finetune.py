"""Freeze the low-learning-rate protocol before accessing new test activations."""
import shutil
from kard_finetune import OUT,BASE,ROOT,sha,json

if (OUT/'protocol.json').exists():raise RuntimeError('Fine-tuning protocol already frozen')
assert (OUT/'gradient_verification.json').exists(),'Verify upper-layer gradients first'
manifest=json.loads((BASE/'manifest.json').read_text())
assert manifest['selected_counts']=={'train':324,'validation':108,'test':108}
for r in manifest['records']:assert sha(r['path'])==r['sha256'],r['path']
check=json.loads((OUT/'gradient_verification.json').read_text())
protocol={'version':1,'seed':42,'model':'official MoViNet A0-Stream',
    'initialization':'Original pretrained backbone and the validation-selected full-training-pool KARD head (L2 0.01, epoch 19).',
    'parent_manifest_sha256':sha(BASE/'manifest.json'),'initial_head_sha256':sha(BASE/'kard_head.keras'),
    'original_backbone_file_sha256':sha(ROOT/'models'/'a0_stream'/'variables'/'variables.data-00000-of-00001'),
    'train_clips':324,'validation_clips':108,'test_clips':108,'split_subjects':manifest['split_subjects'],
    'input_policy':manifest['protocol']['preprocessing'],'target_fps':5,'resolution':172,'max_frames':50,
    'trainable_backbone_prefixes':['b4/l2/','b4/l3/','head/'],
    'excluded_from_training':'All batch-normalization variables, including gamma/beta and moving statistics; all earlier backbone layers; fixed head normalization statistics.',
    'backbone_call_mode':'Inference mode for normalization and stochastic depth; GradientTape updates only the explicitly selected upper variables. Head dropout remains active during optimization.',
    'upper_backbone_trainable_parameters':check['upper_trainable_parameters'],'head_trainable_parameters':8658,
    'optimizer':'Adam','learning_rate':1e-5,'gradient_global_norm_clip':1.0,
    'batching':'Seeded shuffled exact-length buckets, maximum batch size 8; variable batch sizes; no frame padding, repetition or truncation beyond the existing input policy.',
    'head_l2':0.01,'additional_backbone_l2':0,'maximum_epochs':20,'early_stopping_patience':5,
    'checkpoint_selection':'Highest validation macro F1, tie broken by lower validation cross entropy; include the starting epoch-0 model as a fallback. No learning-rate or layer-scope search in this experiment.',
    'activation_cache':'Cache the frozen b4/l1 endpoint in float32 for train/validation only before model selection. Re-run b4/l2, b4/l3 and pooled head with real gradients each step. Verify equivalence to the full original model and after updating weights.',
    'augmentation':'No video augmentation or person cropping; classifier dropout 0.25 only.',
    'test_access':'Create new test activations and evaluate the selected checkpoint only after writing selection.json.',
    'device':'Native Windows TensorFlow CPU in Miniconda xai; same runtime as head-only baselines.',
    'limitations':'One previously evaluated researcher-defined subject split and seed; only two test subjects. Earlier test results already observed. Validation-only selection; no post-test tuning or significance claim.',
    'sources':['https://www.tensorflow.org/guide/keras/transfer_learning','https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet']}
shutil.copy2(BASE/'manifest.json',OUT/'manifest.json')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
(OUT/'source_verification.json').write_text(json.dumps({'source_sha256_checks_passed':540,
    'parent_head_sha256':protocol['initial_head_sha256'],'parent_manifest_sha256':protocol['parent_manifest_sha256'],
    'subject_split_unchanged':True,'test_not_used_for_protocol_or_checkpoint_selection':True},indent=2),encoding='utf-8')
print('FINETUNE_PROTOCOL_FROZEN',json.dumps(protocol),flush=True)
