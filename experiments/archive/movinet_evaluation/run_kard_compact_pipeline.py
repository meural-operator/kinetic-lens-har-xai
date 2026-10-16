"""Frozen official MoViNet + compact temporal KARD head; resumable experiment.

TensorFlow CPU replays immutable backbone features. Only the new compact module
and classifier are optimized, using PyTorch CUDA in the xai environment.
"""
import os
os.environ['KARD_TRAIN_FRACTION'] = '1.0'
os.environ['KARD_TARGET_FPS'] = '5'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import torch
import numpy as np
from kard_compact_model import CompactTemporalHead, load_compact_head
from kard_model import FrozenBackbone, read_clip, variable_digest, metrics, tf

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'kard_a0_head_100pct'
UPPER = ROOT/'kard_a0_upper_finetune'
OUT = ROOT/'kard_a0_compact_temporal_v1'
MANIFEST = json.loads((BASE/'manifest.json').read_text())
LABELS = MANIFEST['labels']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_torch(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def stem(record):
    return record['split']+'_'+record['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')


def status(phase, **extra):
    data = {'phase':phase, 'pid':os.getpid(), 'updated_unix':time.time(), **extra}
    write_json(OUT/'status.json', data)
    print(json.dumps(data), flush=True)


class FrozenSequences:
    def __init__(self):
        self.features = FrozenBackbone()
        self.backbone = self.features.backbone
        self.layers = [getattr(self.backbone, f'layer_with_weights-{i}') for i in (14,15)]
        assert [l.variables[0].name.split('/')[:2] for l in self.layers] == [['b4','l2'], ['b4','l3']]
        self.states = self.features.model.init_states(tf.constant([1,1,172,172,3]))
        self.replay = tf.function(self._replay, input_signature=[tf.TensorSpec([1,None,6,6,104],tf.float32)])
        self.full = tf.function(self._full, input_signature=[tf.TensorSpec([1,None,172,172,3],tf.float32)])
        self.model_sha = sha(ROOT/'models/a0_stream/variables/variables.data-00000-of-00001')
        self.manifest_sha = sha(BASE/'manifest.json')
        self.upper_protocol_sha = sha(UPPER/'protocol.json')
        assert json.loads((UPPER/'protocol.json').read_text())['original_backbone_file_sha256'] == self.model_sha

    def _replay(self, activation):
        states = self.states
        for layer in self.layers:
            activation, states = layer(activation, states=states, training=False)
        return tf.reduce_mean(activation, axis=[2,3])

    def _full(self, images):
        endpoints, _ = self.backbone({**self.states, 'image':images})
        return tf.reduce_mean(endpoints['b4/l3'], axis=[2,3])

    def source(self, record):
        source = UPPER/'activation_cache'/(stem(record)+'.npz')
        info = json.loads(source.with_suffix('.json').read_text())
        assert info['source_sha256'] == record['sha256']
        assert info['manifest_sha256'] == self.manifest_sha
        assert info['protocol_sha256'] == self.upper_protocol_sha
        assert info['original_backbone_file_sha256'] == self.model_sha
        assert info['cache_sha256'] == sha(source)
        assert sha(record['path']) == record['sha256']
        with np.load(source) as a:
            activation = a['activation']
            assert int(a['label']) == record['class_id'] and int(a['subject']) == record['subject_id']
        assert activation.ndim == 4 and activation.shape[1:] == (6,6,104)
        assert 0 < len(activation) <= 50 and np.isfinite(activation).all()
        return activation, info

    def sequence(self, record):
        activation, info = self.source(record)
        sequence = self.replay(tf.convert_to_tensor(activation[None])).numpy()[0]
        assert sequence.shape == (len(activation),104) and np.isfinite(sequence).all()
        return sequence.astype(np.float32), info

    def verify_unchanged(self):
        assert variable_digest(self.backbone.variables) == self.features.initial_hash


def pack(sequences, device):
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)
    padded = np.zeros((len(sequences), int(lengths.max().item()),104), np.float32)
    for i,s in enumerate(sequences):
        padded[i,:len(s)] = s
    return torch.as_tensor(padded, device=device), lengths


def preflight(extractor, device):
    records = [r for r in MANIFEST['records'] if r['split']=='train'][:3]
    sequences, errors = [], []
    for r in records:
        sequence, info = extractor.sequence(r)
        images, metadata = read_clip(r['path'], MANIFEST['protocol'])
        assert metadata['frame_indices'] == info['metadata']['frame_indices']
        full = extractor.full(images).numpy()[0]
        error = float(np.max(np.abs(full-sequence)))
        assert error < 1e-4, f'Cached/full mismatch: {error}'
        sequences.append(sequence)
        errors.append(error)
    torch.manual_seed(42)
    model = CompactTemporalHead(np.zeros(104),np.ones(104)).to(device)
    x,lengths = pack(sequences, device)
    model.eval()
    with torch.no_grad():
        batch = model(x,lengths)
        singles = torch.cat([model(x[i:i+1,:lengths[i]],lengths[i:i+1]) for i in range(len(sequences))])
        pad_error = float((batch-singles).abs().max())
        extended = torch.nn.functional.pad(x,(0,0,0,7))
        extra_pad_error = float((batch-model(extended,lengths)).abs().max())
        assert max(pad_error,extra_pad_error) < 2e-5
    model.train()
    before = {k:p.detach().clone() for k,p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    labels = torch.tensor([r['class_id'] for r in records],device=device)
    loss = torch.nn.functional.cross_entropy(model(x,lengths),labels)
    loss.backward()
    gradient_norms = {k:float(p.grad.norm()) for k,p in model.named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(v>0 for v in gradient_norms.values())
    optimizer.step()
    changes = {k:float((p.detach()-before[k]).abs().max()) for k,p in model.named_parameters()}
    assert all(v>0 for v in changes.values())
    model.eval()
    checkpoint = OUT/'preflight_weights.pt'
    save_torch(checkpoint,{'state_dict':cpu_state(model),'labels':LABELS})
    restored,_ = load_compact_head(checkpoint,device)
    with torch.no_grad():
        reload_error = float((model(x,lengths)-restored(x,lengths)).abs().max())
    assert reload_error == 0
    extractor.verify_unchanged()
    result = {'full_cached_sequence_max_errors':errors,'mixed_length_max_error':pad_error,
        'extra_right_padding_max_error':extra_pad_error,'checkpoint_reload_max_error':reload_error,
        'all_new_parameters_have_finite_nonzero_gradients':True,'gradient_norms':gradient_norms,
        'maximum_parameter_changes':changes,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'backbone_unchanged':True,'training_device':str(device),'gpu':torch.cuda.get_device_name(0),
        'torch_version':torch.__version__,'tensorflow_version':tf.__version__}
    write_json(OUT/'preflight.json', result)
    print('PREFLIGHT_PASSED',json.dumps(result),flush=True)
    return result


def cache_split(extractor, split):
    if split=='test' and not (OUT/'selection.json').exists():
        raise RuntimeError('Select the checkpoint before accessing test sequences.')
    records = [r for r in MANIFEST['records'] if r['split']==split]
    folder = OUT/'sequence_cache'
    folder.mkdir(exist_ok=True)
    for i,r in enumerate(records,1):
        path = folder/(stem(r)+'.npz')
        info_path = path.with_suffix('.json')
        if path.exists() and info_path.exists():
            info = json.loads(info_path.read_text())
            assert info['video_sha256']==r['sha256'] and info['model_sha256']==extractor.model_sha
            assert info['manifest_sha256']==extractor.manifest_sha and info['cache_sha256']==sha(path)
            assert info['endpoint']=='b4/l3_spatial_mean'
        else:
            sequence, source_info = extractor.sequence(r)
            np.savez_compressed(path,sequence=sequence,label=np.int32(r['class_id']),subject=np.int32(r['subject_id']))
            write_json(info_path,{'video':r['relative_path'],'video_sha256':r['sha256'],
                'model_sha256':extractor.model_sha,'manifest_sha256':extractor.manifest_sha,
                'cache_sha256':sha(path),'endpoint':'b4/l3_spatial_mean','shape':list(sequence.shape),
                'source_activation_sha256':source_info['cache_sha256'],'metadata':source_info['metadata']})
        if i%25==0 or i==len(records):
            status('extracting_sequences',split=split,clips_ready=i,clips_total=len(records))
    extractor.verify_unchanged()
    write_json(OUT/f'cache_verification_{split}.json',{'clips':len(records),'backbone_unchanged':True,
        'backbone_variable_digest':extractor.features.initial_hash,'test_selection_gate':split!='test' or (OUT/'selection.json').exists()})
    sequences=[]
    for r in records:
        with np.load(folder/(stem(r)+'.npz')) as data:
            assert int(data['label'])==r['class_id']
            sequences.append(data['sequence'])
    return sequences,np.asarray([r['class_id'] for r in records],np.int64),records


def predict(model, x, lengths, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+32].to(device),lengths[i:i+32].to(device)).cpu()
            for i in range(0,len(x),32)]).numpy()


def train(train_sequences,y,val_sequences,vy,device):
    # Equal clip weight: long clips do not dominate feature standardization.
    mean = np.mean([s.mean(0,dtype=np.float64) for s in train_sequences],axis=0)
    variance = np.mean([np.square(s.astype(np.float64)-mean).mean(0) for s in train_sequences],axis=0)
    scale = np.maximum(np.sqrt(variance),1e-3)
    np.savez(OUT/'train_normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
    x,lengths = pack(train_sequences,'cpu')
    vx,vl = pack(val_sequences,'cpu')
    targets = torch.tensor(y,dtype=torch.long,device=device)
    experiments=[]
    for l2 in (.0001,.001,.01):
        tag = f'l2_{l2:g}'
        best_path,last_path = OUT/f'best_{tag}.pt',OUT/f'last_{tag}.pt'
        history_path = OUT/f'history_{tag}.json'
        summary_path = OUT/f'summary_{tag}.json'
        if summary_path.exists():
            experiments.append(json.loads(summary_path.read_text()))
            continue
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = CompactTemporalHead(mean.astype(np.float32),scale.astype(np.float32)).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        initial = {k:p.detach().clone() for k,p in model.named_parameters()}
        rng = np.random.default_rng(42)
        history=[]
        best_key=(-1.,float('-inf'))
        best_epoch=stale=0
        start_epoch=1
        if last_path.exists():
            resume = torch.load(last_path,map_location=device,weights_only=False)
            model.load_state_dict(resume['state_dict'])
            optimizer.load_state_dict(resume['optimizer'])
            rng.bit_generator.state=resume['numpy_rng']
            torch.set_rng_state(resume['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([a.cpu() for a in resume['cuda_rng']])
            best_key=tuple(resume['best_key']);best_epoch=resume['best_epoch'];stale=resume['stale']
            start_epoch=resume['epoch']+1
            history=json.loads(history_path.read_text())[:resume['epoch']]
        start = time.perf_counter()
        for epoch in range(start_epoch,201):
            if stale>=25:
                break
            model.train()
            losses=[]
            permutation=rng.permutation(len(y))
            for offset in range(0,len(y),16):
                ids=permutation[offset:offset+16]
                index=torch.as_tensor(ids,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x[index].to(device),lengths[index].to(device))
                # Match the baseline's explicit kernel L2 convention, rather
                # than silently substituting AdamW decay. Biases/LN excluded.
                regularization=sum(p.square().sum() for name,p in model.named_parameters()
                    if name.endswith('weight') and p.ndim>1)
                loss=torch.nn.functional.cross_entropy(logits,targets[index.to(device)])+l2*regularization
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            tm=metrics(predict(model,x,lengths,device),y)
            vm=metrics(predict(model,vx,vl,device),vy)
            key=(vm['macro_f1'],-vm['cross_entropy'])
            entry={'epoch':epoch,'loss':float(np.mean(losses)),
                'train_accuracy':tm['accuracy'],'train_macro_f1':tm['macro_f1'],
                'validation_accuracy':vm['accuracy'],'validation_macro_f1':vm['macro_f1'],
                'validation_cross_entropy':vm['cross_entropy']}
            history.append(entry)
            if key>best_key:
                best_key=key;best_epoch=epoch;stale=0
                save_torch(best_path,{'state_dict':cpu_state(model),'labels':LABELS,'epoch':epoch,
                    'l2':l2,'architecture':'compact_temporal_v1','seed':42})
            else:
                stale+=1
            write_json(history_path,history)
            save_torch(last_path,{'state_dict':cpu_state(model),'optimizer':optimizer.state_dict(),
                'epoch':epoch,'best_key':best_key,'best_epoch':best_epoch,'stale':stale,
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()})
            if epoch==1 or epoch%5==0:
                status('training',candidate=tag,**entry,best_epoch=best_epoch,seconds=time.perf_counter()-start)
        best,_=load_compact_head(best_path,device)
        delta=max(float((p.detach()-initial[name]).abs().max()) for name,p in best.named_parameters())
        assert delta>1e-5
        summary={'l2':l2,'checkpoint':best_path.name,'best_epoch':best_epoch,
            'epochs_executed':len(history),'maximum_parameter_change':delta,
            'train':metrics(predict(best,x,lengths,device),y),
            'validation':metrics(predict(best,vx,vl,device),vy)}
        write_json(summary_path,summary)
        experiments.append(summary)
    winner=max(experiments,key=lambda e:(e['validation']['macro_f1'],-e['validation']['cross_entropy']))
    selected,_=load_compact_head(OUT/winner['checkpoint'],device)
    save_torch(OUT/'compact_kard_head.pt',{'state_dict':cpu_state(selected),'labels':LABELS,
        'architecture':'compact_temporal_v1','seed':42,'epoch':winner['best_epoch'],'l2':winner['l2']})
    result={'winner':winner,'candidates':experiments,'trainable_parameters':sum(p.numel() for p in selected.parameters()),
        'selection_rule':'Highest validation macro F1; lower validation cross entropy breaks ties.',
        'test_used_for_training_or_selection':False,'seed':42,'training_device':str(device),
        'manifest_sha256':sha(BASE/'manifest.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'checkpoint_sha256':sha(OUT/'compact_kard_head.pt')}
    write_json(OUT/'selection.json',result)
    return result


def report(evaluation,selection):
    baseline_selection=json.loads((BASE/'selection.json').read_text())
    baseline_evaluation=json.loads((BASE/'evaluation.json').read_text())
    bv=baseline_selection['winner']['validation']
    bt=baseline_evaluation['test']
    cv=selection['winner']['validation'];ct=evaluation['test']
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['model','validation_accuracy','validation_macro_f1','test_accuracy','test_macro_f1'])
        writer.writerow(['original',bv['accuracy'],bv['macro_f1'],bt['accuracy'],bt['macro_f1']])
        writer.writerow(['compact_temporal',cv['accuracy'],cv['macro_f1'],ct['accuracy'],ct['macro_f1']])
    text=f'''# KARD compact temporal architecture: first controlled experiment

The official MoViNet A0-Stream stem and all 15 bottlenecks remain frozen.
The original spatial 104-to-480 projection and pooled classifier are bypassed.
New path: b4/l3 -> spatial average per frame -> train-only 104-channel
standardization -> causal depthwise temporal convolution (kernel 3), hard-swish
and residual -> masked temporal average -> Linear(104,128) -> LayerNorm ->
hard-swish -> dropout 0.25 -> Linear(128,18).

Trainable parameters: {selection['trainable_parameters']:,}. Only the new module
and classifier are optimized; the original backbone is unchanged.
Preprocessing and subject split match the original full-training-pool baseline.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Original | {bv['accuracy']:.2%} | {bv['macro_f1']:.2%} | {bt['accuracy']:.2%} | {bt['macro_f1']:.2%} |
| Compact temporal | {cv['accuracy']:.2%} | {cv['macro_f1']:.2%} | {ct['accuracy']:.2%} | {ct['macro_f1']:.2%} |

The selected candidate uses L2 {selection['winner']['l2']:g}, epoch
{selection['winner']['best_epoch']}. Selection uses validation only. The selected
checkpoint is evaluated on test once. See protocol.json, preflight.json,
selection.json, histories, evaluation.json and predictions.csv for evidence.

## Interpretation limits

This is one seed and one previously observed researcher-defined subject split,
with two validation and two test subjects. It is not a blind benchmark,
significance result or a demonstrated novel method. Model, normalization,
optimizer and regularization differ from the original TensorFlow linear head;
this is a complete architecture comparison, not an isolated temporal-module
ablation. No improved latency, smaller end-to-end artifact or real-time behavior
is claimed until both complete execution graphs are exported and benchmarked.
The frozen extraction here is TensorFlow CPU; new module training is PyTorch CUDA.
The new inference helper requires both runtimes. Dataset and old artifacts are preserved.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training is required for this experiment.')
    device=torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not args.preflight_only and (OUT/'status.json').exists() and not args.resume:
        raise RuntimeError('Existing experiment preserved. Use --resume after confirming no job is running.')
    # Test records are not loaded by extraction/training before selection.
    train_records=[r for r in MANIFEST['records'] if r['split']=='train']
    val_records=[r for r in MANIFEST['records'] if r['split']=='validation']
    assert len(train_records)==324 and len(val_records)==108
    assert set(r['subject_id'] for r in train_records)=={1,3,4,6,7,8}
    assert set(r['subject_id'] for r in val_records)=={5,10}
    extractor=FrozenSequences()
    if args.preflight_only:
        preflight(extractor,device)
        return
    protocol={'version':1,'architecture':'compact_temporal_v1','seed':42,'dataset':'KARD',
        'parent_manifest_sha256':sha(BASE/'manifest.json'),'original_backbone_sha256':extractor.model_sha,
        'labels':LABELS,'subject_split':MANIFEST['split_subjects'],'train_clips':324,'validation_clips':108,'test_clips':108,
        'input':MANIFEST['protocol']['preprocessing'],'feature_endpoint':'b4/l3 spatial mean, [T,104]',
        'frozen':'Original official stem and blocks b0-b4, including BN; no original projection/classifier is trained.',
        'new_module':'Causal depthwise Conv1d(104,104,kernel=3), hard-swish residual; masked time mean; Linear(104,128), LayerNorm, hard-swish, dropout(.25), Linear(128,18).',
        'normalization':'Training clips only, equal clip weights for per-frame channel mean and variance.',
        'optimizer':'Adam lr=.001, explicit kernel L2 candidates [.0001,.001,.01], global gradient norm clip 1.',
        'batching':'16 clips nominal, causal computation and masked pooling exclude right padding; no duplicated frames.',
        'max_epochs':200,'patience':25,'selection':'Validation macro F1, then validation cross entropy.',
        'augmentation':'No video augmentation; dropout during training only.',
        'test_policy':'Access test sequence caches only after selection.json. Evaluate selected checkpoint only; no post-test tuning.',
        'previous_test_results_already_observed':True,'training_device':torch.cuda.get_device_name(0),
        'feature_extraction_device':'TensorFlow CPU; replay verified immutable existing b4/l1 activations.',
        'limitations':'One seed, previously observed subject split, two subjects per held-out partition; no novelty/speed claim.'}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:
        write_json(OUT/'protocol.json',protocol)
        (OUT/'manifest.json').write_bytes((BASE/'manifest.json').read_bytes())
        write_json(OUT/'labels.json',LABELS)
    if os.name=='nt':
        # A process-scoped request prevents automatic sleep during this job;
        # normal power settings resume when the process exits. Display may sleep.
        import ctypes
        sleep_request=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        if not sleep_request:
            print('Could not request awake execution; keep Windows awake manually.',flush=True)
    status('preflight')
    preflight(extractor,device)
    train_sequences,y,_=cache_split(extractor,'train')
    val_sequences,vy,_=cache_split(extractor,'validation')
    if (OUT/'selection.json').exists():
        selection=json.loads((OUT/'selection.json').read_text())
        assert selection['protocol_sha256']==sha(OUT/'protocol.json')
        assert selection['checkpoint_sha256']==sha(OUT/'compact_kard_head.pt')
    else:
        selection=train(train_sequences,y,val_sequences,vy,device)
    extractor.verify_unchanged()
    if (OUT/'evaluation.json').exists():
        evaluation=json.loads((OUT/'evaluation.json').read_text())
    else:
        status('evaluating_selected_checkpoint')
        sequences,y,records=cache_split(extractor,'test')
        assert len(y)==108 and set(r['subject_id'] for r in records)=={2,9}
        selected,_=load_compact_head(OUT/'compact_kard_head.pt',device)
        x,lengths=pack(sequences,'cpu')
        logits=predict(selected,x,lengths,device)
        evaluation={'test':metrics(logits,y),'checkpoint_sha256':sha(OUT/'compact_kard_head.pt'),
            'selection_sha256':sha(OUT/'selection.json'),'test_evaluated_after_selection':True}
        write_json(OUT/'evaluation.json',evaluation)
        np.savez(OUT/'test_logits.npz',logits=logits,labels=y)
        with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['video','subject','true_class','predicted_class','correct'])
            for r,p in zip(records,logits.argmax(1)):
                writer.writerow([r['relative_path'],r['subject_id'],r['class_name'],LABELS[p],int(p==r['class_id'])])
        # Verify the selected checkpoint on an uncached train clip through the
        # full original backbone. This does not perform another test evaluation.
        r=train_records[0]
        images,_=read_clip(r['path'],MANIFEST['protocol'])
        full_sequence=extractor.full(images).numpy()[0]
        cached_sequence=train_sequences[0]
        with torch.no_grad():
            fx,fl=pack([full_sequence],device);cx,cl=pack([cached_sequence],device)
            error=float((selected(fx,fl)-selected(cx,cl)).abs().max())
        assert error<1e-4
        write_json(OUT/'selected_inference_verification.json',{'full_cached_logits_max_error':error,
            'checkpoint_reload_verified':True,'backbone_unchanged':True,'video':r['relative_path']})
    extractor.verify_unchanged()
    report(evaluation,selection)
    status('complete',validation_macro_f1=selection['winner']['validation']['macro_f1'],
        test_macro_f1=evaluation['test']['macro_f1'],checkpoint='compact_kard_head.pt')
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            write_json(OUT/'failure.json',{'pid':os.getpid(),'time_unix':time.time(),'traceback':traceback.format_exc()})
        raise
