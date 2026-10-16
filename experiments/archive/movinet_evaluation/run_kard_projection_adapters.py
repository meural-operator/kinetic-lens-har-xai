"""Matched CUDA adaptation: ordinary LoRA, classifier-guided and random span.

All selections lock before any test cache access. Supports exact epoch resume.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import contextlib
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kard_projection_adapter import ProjectionAdapter, classifier_basis, hard_swish
from kard_ablation_model import state_digest
from run_kard_ablation import metrics, sha, write_json, save_torch, cpu_state, reset_rng, rng_state, restore_rng

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'kard_a0_projection_adapter_v1'
BASE=ROOT/'kard_a0_head_100pct'
CONTROL=ROOT/'kard_a0_controlled_ablation_v1'
SEEDS=(42,123,2026)
VARIANTS=('lora','guided','random')
L2S=(.0001,.001,.01)
MANIFEST=json.loads((BASE/'manifest.json').read_text())
LABELS=MANIFEST['labels']
MAX_EPOCHS=100
PATIENCE=15
LR=1e-4
MAX_RELATIVE_UPDATE=.02

def status(phase,**data):
    row={'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**data}
    write_json(OUT/'status.json',row);print(json.dumps(row),flush=True)

def control(seed):
    gate=json.loads((CONTROL/'selection_manifest.json').read_text())
    item=gate['selections'][f'A_seed_{seed}']
    path=CONTROL/item['checkpoint'];assert sha(path)==item['checkpoint_sha256']
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['variant']=='A' and checkpoint['seed']==seed
    return checkpoint['state_dict'],item

def projection():
    with np.load(OUT/'original_projection.npz') as data:
        return {k:torch.tensor(data[k].copy()) for k in data.files}

class AdaptedModel(nn.Module):
    def __init__(self,variant,seed):
        super().__init__();self.variant=variant;self.seed=seed
        head,_=control(seed);p=projection()
        for k,v in head.items():self.register_buffer(k.replace('.','_'),v.clone())
        for k in ('gamma','beta','moving_mean','moving_variance'):self.register_buffer(k,p[k])
        gain=p['gamma']/torch.sqrt(p['moving_variance']+.001)
        guided=classifier_basis(head['classifier.weight'].T,head['scale'],gain)
        assert guided.shape==(480,17)
        if variant=='guided':basis=guided
        elif variant=='lora':basis=torch.eye(480)
        elif variant=='random':
            generator=torch.Generator().manual_seed(seed+7919)
            basis=torch.linalg.qr(torch.randn(480,17,generator=generator,dtype=torch.float64),mode='reduced')[0].float()
        else:raise ValueError(variant)
        # Same input factor for all variants; independent of basis construction.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+104729)
            self.adapter=ProjectionAdapter(p['kernel'],basis,rank=4,alpha=4)
        self.dropout=nn.Dropout(.25)
        self.register_buffer('bn_gain',gain)

    def forward(self,positions,lengths,merged=None):
        # Fold a linear update into W before projection, including during
        # training. This avoids a separate 480x480 multiply for ordinary LoRA.
        kernel=self.adapter.merged_kernel() if merged is None else merged
        projected=hard_swish((positions@kernel-self.moving_mean)*self.bn_gain+self.beta)
        mask=torch.arange(positions.shape[1],device=positions.device)[None]<lengths[:,None]
        pooled=(projected*mask[...,None]).sum(1)/lengths[:,None]
        normalized=(pooled-self.mean)/self.scale
        return F.linear(self.dropout(normalized),self.classifier_weight,self.classifier_bias)

    def baseline_logits(self,wide):
        return F.linear((wide-self.mean)/self.scale,self.classifier_weight,self.classifier_bias)

    def constrain(self):
        # Bound the actual kernel perturbation, not scale-ambiguous factors.
        with torch.no_grad():
            limit=MAX_RELATIVE_UPDATE*self.adapter.kernel.norm()
            norm=self.adapter.delta().norm()
            if norm>limit:self.adapter.factor_out.mul_(limit/norm)

def protocol():
    initial={str(s):{'checkpoint_sha256':control(s)[1]['checkpoint_sha256'],
        'validation':control(s)[1]['validation']} for s in SEEDS}
    return {'version':1,'dataset':'KARD','model':'MoViNet A0-Stream',
        'seeds':list(SEEDS),'variants':list(VARIANTS),'l2_candidates':list(L2S),
        'trainable_parameters':{'lora':2336,'guided':484,'random':484},
        'baseline_checkpoints':initial,'manifest_sha256':sha(BASE/'manifest.json'),
        'original_projection_sha256':sha(OUT/'original_projection.npz'),
        'split_subjects':MANIFEST['split_subjects'],'split_counts':{'train':324,'validation':108,'test':108},
        'preprocessing':MANIFEST['protocol']['preprocessing'],
        'cache_inventories':{s:sha(OUT/f'cache_inventory_{s}.json') for s in ('train','validation')},
        'optimizer':{'name':'Adam','lr':LR,'betas':[.9,.999],'eps':1e-8,'batch_size':16,
            'clip_grad_norm':1.,'maximum_epochs':MAX_EPOCHS,'patience':PATIENCE},
        'adapter':{'rank':4,'alpha':4,'input_factor_initialization':'Normal(0,.01), shared per seed',
            'output_factor_initialization':'Zero','random_basis_seed_offset':7919,
            'maximum_relative_kernel_update_frobenius':MAX_RELATIVE_UPDATE,
            'penalty':'lambda * sum(delta_kernel**2); same three lambdas for all variants'},
        'frozen':'Stem, all 15 bottlenecks, original projection reference, BN including affine and statistics, selected affine KARD head and normalization.',
        'training':'Only adapter factors train; classifier dropout .25; fixed per-seed shuffled batches; no augmentation.',
        'selection':'Validation macro F1 then CE (tie tolerance 1e-7); include exact unchanged baseline at epoch zero; earlier epoch/grid order on tie; no best-seed selection.',
        'test_gate':'All nine variant/seed selections and their hashes persist before any test cache read/preparation. Only selected checkpoints evaluated.',
        'runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'deterministic':True,'tf32':False,'dtype':'float32','torch_threads':2},
        'limits':'Previously observed fixed subject split; three seeds quantify optimization variation, not population confidence intervals. Novelty and improved deployment latency unestablished.'}

def validate_gate():
    path=OUT/'selection_manifest.json'
    if not path.exists():raise RuntimeError('Test access blocked until all nine selections are locked.')
    gate=json.loads(path.read_text());assert gate['all_selections_locked'] and not gate['test_used_for_selection']
    assert gate['protocol_sha256']==sha(OUT/'protocol.json')
    assert set(gate['selections'])=={f'{v}_seed_{s}' for v in VARIANTS for s in SEEDS}
    for item in gate['selections'].values():assert sha(OUT/item['checkpoint'])==item['checkpoint_sha256']
    return gate

def stem(r):return r['split']+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')

def load_data(split,device):
    if split=='test':validate_gate()
    records=[r for r in MANIFEST['records'] if r['split']==split and r['selected']]
    inventory=json.loads((OUT/f'cache_inventory_{split}.json').read_text())['files']
    sequences=[];wide=[]
    assert len(records)=={'train':324,'validation':108,'test':108}[split]
    assert np.bincount([r['class_id'] for r in records],minlength=18).tolist()==[len(records)//18]*18
    for r in records:
        path=OUT/'activation_cache'/(stem(r)+'.npz');info=inventory[stem(r)]
        assert sha(path)==info['cache_sha256'] and info['video_sha256']==r['sha256']
        assert sha(r['path'])==r['sha256']
        assert info['projection_sha256']==sha(OUT/'original_projection.npz')
        assert json.loads(path.with_suffix('.json').read_text())==info
        with np.load(path,allow_pickle=False) as data:
            a=data['activation'].copy();w=data['wide'].copy()
            assert int(data['label'])==r['class_id'] and int(data['subject'])==r['subject_id']
        assert list(a.shape)==info['shape'] and a.dtype==np.float32 and np.isfinite(a).all()
        assert a.shape[1:]==(6,6,104) and w.shape==(480,) and np.isfinite(w).all()
        sequences.append(a.reshape(-1,104));wide.append(w)
    padded=np.zeros((len(sequences),max(map(len,sequences)),104),dtype=np.float32)
    for i,a in enumerate(sequences):padded[i,:len(a)]=a
    return {'positions':torch.tensor(padded,device=device),
        'lengths':torch.tensor([len(a) for a in sequences],device=device),
        'wide':torch.tensor(np.stack(wide),device=device),
        'labels':torch.tensor([r['class_id'] for r in records],device=device),'records':records}

def predict(model,data,merged=None):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(data['positions'][i:i+16],data['lengths'][i:i+16],merged)
            for i in range(0,len(data['labels']),16)]).cpu().numpy()

def evaluate(model,data,baseline=False):
    values=model.baseline_logits(data['wide']).detach().cpu().numpy() if baseline else predict(model,data)
    return metrics(values,data['labels'].cpu().numpy())

def key_better(value,best):
    return value['macro_f1']>best['macro_f1'] or (value['macro_f1']==best['macro_f1'] and value['cross_entropy']<best['cross_entropy']-1e-7)

def step(model,optimizer,data,ids,l2):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(data['positions'][ids],data['lengths'][ids]),data['labels'][ids])+l2*model.adapter.delta().square().sum()
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();model.constrain()
    return float(loss.detach())

def preflight(train,validation,device):
    evidence={'test_cache_accesses':0,'variants':{},'initial_parity':{},'baseline':{},'paired_input_initialization':{}}
    for seed in SEEDS:
        factors=[]
        for variant in VARIANTS:
            model=AdaptedModel(variant,seed).to(device);model.eval()
            assert sum(p.numel() for p in model.parameters())=={'lora':2336,'guided':484,'random':484}[variant]
            factors.append(model.adapter.factor_in.detach().cpu())
            if variant=='lora':
                evidence['baseline'][str(seed)]={s:evaluate(model,d,True) for s,d in [('train',train),('validation',validation)]}
                expected=control(seed)[1]['validation']
                actual=evidence['baseline'][str(seed)]['validation']
                assert actual['macro_f1']==expected['macro_f1'] and abs(actual['cross_entropy']-expected['cross_entropy'])<1e-5
            assert torch.equal(model.adapter.merged_kernel(),model.adapter.kernel)
            parity={}
            for split,data in [('train',train),('validation',validation)]:
                projected=predict(model,data)
                original=model.baseline_logits(data['wide']).detach().cpu().numpy()
                error=float(np.max(np.abs(projected-original)));assert error<1e-4,error
                assert np.array_equal(projected.argmax(1),original.argmax(1))
                parity[split]={'logit_max_error':error,'prediction_exact':True}
            evidence['initial_parity'][f'{variant}_{seed}']=parity
        assert all(torch.equal(factors[0],a) for a in factors[1:])
        evidence['paired_input_initialization'][str(seed)]=state_digest({'factor':factors[0]})
    ids=torch.tensor([0,1,20,100,200,323],device=device)
    small={k:v[ids] for k,v in train.items() if isinstance(v,torch.Tensor)}
    for variant in VARIANTS:
        model=AdaptedModel(variant,42).to(device);reset_rng(42)
        before=cpu_state(model);model.eval()
        with torch.no_grad():
            ref=model(small['positions'],small['lengths'])
            pad=F.pad(small['positions'],(0,0,0,17),value=987.)
            padding_error=float((ref-model(pad,small['lengths'])).abs().max());assert padding_error<1e-4
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        step(model,optimizer,small,slice(None),.001)
        assert model.adapter.factor_out.grad.abs().sum()>0
        step(model,optimizer,small,slice(None),.001)
        assert model.adapter.factor_in.grad.abs().sum()>0
        for k,v in before.items():
            if not k.startswith('adapter.factor_'):assert torch.equal(v,cpu_state(model)[k])
        model.eval()
        adapted=predict(model,small);merged=predict(model,small,model.adapter.merged_kernel().detach())
        merge_error=float(np.max(np.abs(adapted-merged)));assert merge_error==0
        # Check explicit factor path against a genuinely merged projection.
        x=small['positions'][0,:int(small['lengths'][0])]
        with torch.no_grad():
            explicit_error=float((model.adapter(x)-x@model.adapter.merged_kernel()).abs().max())
            assert torch.allclose(model.adapter(x),x@model.adapter.merged_kernel(),rtol=1e-4,atol=1e-4)
        saved={'state':cpu_state(model),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_state(np.random.default_rng(42))}
        save_torch(OUT/f'preflight_{variant}.pt',saved)
        reset_rng(123);rng=np.random.default_rng(123)
        saved={'state':cpu_state(model),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_state(rng)}
        resume_path=OUT/f'preflight_resume_{variant}.pt'
        save_torch(resume_path,saved)
        losses=[step(model,optimizer,small,torch.tensor(rng.permutation(6),device=device),.001) for _ in range(2)]
        final=cpu_state(model)
        saved=torch.load(resume_path,map_location=device,weights_only=False)
        restored=AdaptedModel(variant,42).to(device);restored.load_state_dict(saved['state'])
        opt=torch.optim.Adam(restored.parameters(),lr=LR);opt.load_state_dict(saved['optimizer'])
        rrng=np.random.default_rng();restore_rng(saved['rng'],rrng)
        resumed=[step(restored,opt,small,torch.tensor(rrng.permutation(6),device=device),.001) for _ in range(2)]
        assert losses==resumed and all(torch.equal(v,cpu_state(restored)[k]) for k,v in final.items())
        model.adapter.factor_out.data.mul_(1e6);model.constrain()
        relative=float((model.adapter.delta().norm()/model.adapter.kernel.norm()).detach());assert relative<=MAX_RELATIVE_UPDATE+1e-6
        evidence['variants'][variant]={'padding_max_error':padding_error,'merged_logit_error':merge_error,
            'explicit_vs_merged_projection_error':explicit_error,'resume_exact':True,'frozen_buffers_unchanged':True,
            'relative_update_constraint_verified':relative,'trainable_parameters':sum(p.numel() for p in model.parameters())}
    try:load_data('test',device)
    except RuntimeError:evidence['test_gate_rejected_before_read']=True
    else:raise AssertionError('Gate should reject before selections')
    # Small timing only estimates an epoch; not an end-to-end inference benchmark.
    model=AdaptedModel('guided',42).to(device);opt=torch.optim.Adam(model.parameters(),lr=LR)
    reset_rng(42);torch.cuda.synchronize();start=time.perf_counter()
    for i in range(0,len(train['labels']),16):step(model,opt,train,slice(i,i+16),.001)
    evaluate(model,train);evaluate(model,validation);torch.cuda.synchronize()
    evidence['measured_one_train_and_validation_epoch_seconds']=time.perf_counter()-start
    evidence['hashes']={name:sha(ROOT/name) for name in ('run_kard_projection_adapters.py','kard_projection_adapter.py','prepare_kard_adapter_cache.py','write_kard_adapter_report.py','run_kard_ablation.py','kard_ablation_model.py')}
    evidence['protocol_sha256']=sha(OUT/'protocol.json')
    write_json(OUT/'preflight.json',evidence)
    print('PREFLIGHT_PASSED',json.dumps({k:v for k,v in evidence.items() if k not in ('baseline','initial_parity')}),flush=True)

def candidate(variant,seed,l2,train,validation,device):
    folder=OUT/'candidates'/f'{variant}_seed_{seed}'/f'l2_{l2:g}';folder.mkdir(parents=True,exist_ok=True)
    summary_path=folder/'summary.json'
    if summary_path.exists():
        saved=json.loads(summary_path.read_text());assert saved['protocol_sha256']==sha(OUT/'protocol.json')
        assert sha(OUT/saved['checkpoint'])==saved['checkpoint_sha256'];return saved
    model=AdaptedModel(variant,seed).to(device);reset_rng(seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR);rng=np.random.default_rng(seed)
    initial=state_digest(cpu_state(model));base=evaluate(model,validation,True)
    best_state=cpu_state(model);best_metrics=base;best_epoch=0;stale=0;start_epoch=1;history=[]
    last=folder/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        assert saved['initial_sha256']==initial and saved['protocol_sha256']==sha(OUT/'protocol.json')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],rng)
        best_state={k:v.cpu() for k,v in saved['best_state'].items()};best_metrics=saved['best_metrics']
        best_epoch=saved['best_epoch'];history=saved['history'];stale=saved['stale'];start_epoch=saved['epoch']+1
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        if stale>=PATIENCE:break
        epoch_start=time.perf_counter();permutation=rng.permutation(len(train['labels']))
        losses=[step(model,optimizer,train,torch.tensor(permutation[i:i+16],device=device),l2) for i in range(0,len(permutation),16)]
        tm=evaluate(model,train);vm=evaluate(model,validation)
        relative=float((model.adapter.delta().norm()/model.adapter.kernel.norm()).detach())
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'train':tm,'validation':vm,
            'relative_update':relative,'seconds':time.perf_counter()-epoch_start,
            'permutation_sha256':hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()})
        if key_better(vm,best_metrics):best_metrics,best_state,best_epoch,stale=vm,cpu_state(model),epoch,0
        else:stale+=1
        save_torch(last,{'state':cpu_state(model),'optimizer':optimizer.state_dict(),'rng':rng_state(rng),
            'epoch':epoch,'history':history,'best_state':best_state,'best_metrics':best_metrics,
            'best_epoch':best_epoch,'stale':stale,'initial_sha256':initial,'protocol_sha256':sha(OUT/'protocol.json')})
        write_json(folder/'history.json',history)
        if epoch==1 or epoch%5==0:status('training',variant=variant,seed=seed,l2=l2,epoch=epoch,
            validation_macro_f1=vm['macro_f1'],best_epoch=best_epoch,epoch_seconds=history[-1]['seconds'])
    model.load_state_dict(best_state);model.eval()
    train_metrics=evaluate(model,train,best_epoch==0)
    best_path=folder/'best.pt'
    save_torch(best_path,{'state_dict':best_state,'variant':variant,'seed':seed,'l2':l2,'epoch':best_epoch,
        'merged_kernel':model.adapter.merged_kernel().detach().cpu(),'labels':LABELS,'baseline_checkpoint_sha256':control(seed)[1]['checkpoint_sha256']})
    result={'variant':variant,'seed':seed,'l2':l2,'best_epoch':best_epoch,'epochs_executed':len(history),
        'baseline_validation':base,'validation':best_metrics,'train':train_metrics,
        'checkpoint':str(best_path.relative_to(OUT)).replace('\\','/'),'checkpoint_sha256':sha(best_path),
        'protocol_sha256':sha(OUT/'protocol.json'),'initial_sha256':initial,
        'relative_kernel_update':float((model.adapter.delta().norm()/model.adapter.kernel.norm()).detach())}
    assert result['validation']['macro_f1']>=base['macro_f1']
    write_json(summary_path,result);return result

def select_all(train,validation,device):
    if (OUT/'selection_manifest.json').exists():return validate_gate()
    selections={}
    for seed in SEEDS:
        for variant in VARIANTS:
            candidates=[candidate(variant,seed,l2,train,validation,device) for l2 in L2S]
            winner=candidates[0]
            for item in candidates[1:]:
                if key_better(item['validation'],winner['validation']):winner=item
            name=f'{variant}_seed_{seed}';path=OUT/'selected'/(name+'.pt');path.parent.mkdir(exist_ok=True)
            checkpoint=torch.load(OUT/winner['checkpoint'],map_location='cpu',weights_only=True);save_torch(path,checkpoint)
            selections[name]={**winner,'candidate_checkpoint':winner['checkpoint'],'candidates':candidates,
                'checkpoint':str(path.relative_to(OUT)).replace('\\','/'),'checkpoint_sha256':sha(path)}
            write_json(path.with_suffix('.json'),selections[name]);status('selection_locked',variant=variant,seed=seed,selections_locked=len(selections),selections_total=9)
    write_json(OUT/'selection_manifest.json',{'all_selections_locked':True,'test_used_for_selection':False,
        'protocol_sha256':sha(OUT/'protocol.json'),'locked_unix':time.time(),'selections':selections})
    return validate_gate()

def evaluate_selected(gate,device):
    validate_gate()
    if not (OUT/'cache_inventory_test.json').exists():
        status('preparing_test_after_selection')
        subprocess.run([sys.executable,'-u','-B',str(ROOT/'prepare_kard_adapter_cache.py'),'--split','test'],check=True)
    test=load_data('test',device);results={};baselines={}
    for seed in SEEDS:
        model=AdaptedModel('guided',seed).to(device);baselines[str(seed)]=evaluate(model,test,True)
    for name,item in gate['selections'].items():
        checkpoint=torch.load(OUT/item['checkpoint'],map_location=device,weights_only=True)
        model=AdaptedModel(item['variant'],item['seed']).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
        values=model.baseline_logits(test['wide']).detach().cpu().numpy() if item['best_epoch']==0 else predict(model,test)
        merged=predict(model,test,checkpoint['merged_kernel'].to(device))
        merge_error=float(np.max(np.abs(values-merged)));assert merge_error<1e-4
        folder=OUT/'evaluation'/name;folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'test_logits.npz',logits=values,labels=test['labels'].cpu().numpy())
        result={'variant':item['variant'],'seed':item['seed'],'test':metrics(values,test['labels'].cpu().numpy()),
            'baseline_test':baselines[str(item['seed'])],'merged_logit_max_error':merge_error,
            'checkpoint_sha256':item['checkpoint_sha256'],'selection_manifest_sha256':sha(OUT/'selection_manifest.json'),
            'test_evaluated_after_all_selections':True}
        write_json(folder/'metrics.json',result);results[name]=result
    write_json(OUT/'evaluation.json',{'results':results,'baselines':baselines});return results,baselines

@contextlib.contextmanager
def lock():
    kernel=ctypes.windll.kernel32;kernel.CreateMutexW.restype=ctypes.c_void_p
    handle=kernel.CreateMutexW(None,False,'Local\\KARD_Projection_Adapter_'+hashlib.sha256(str(OUT).encode()).hexdigest()[:20])
    if not handle:raise ctypes.WinError()
    try:
        if kernel.GetLastError()==183:raise RuntimeError('Another adapter experiment is running.')
        yield
    finally:
        kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    with lock():
        proposed=protocol()
        if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==proposed
        else:write_json(OUT/'protocol.json',proposed)
        if not args.preflight_only:
            checked=json.loads((OUT/'preflight.json').read_text())
            assert checked['protocol_sha256']==sha(OUT/'protocol.json')
            for name,digest in checked['hashes'].items():assert sha(ROOT/name)==digest
            if (OUT/'status.json').exists() and not args.resume:raise RuntimeError('Use --resume for existing experiment')
        awake=False
        try:
            if not args.preflight_only:
                awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
                status('startup_verified',gpu=torch.cuda.get_device_name(0),sleep_prevention_requested=awake,monitor_policy='Check only when requested by user.')
            train=load_data('train',device);validation=load_data('validation',device)
            if args.preflight_only:preflight(train,validation,device);return
            gate=select_all(train,validation,device)
            status('evaluating_locked_selections',selections=9)
            results,baselines=evaluate_selected(gate,device)
            # Verify train/validation file inventories again before reporting.
            for split in ('train','validation'):
                assert sha(OUT/f'cache_inventory_{split}.json')==proposed['cache_inventories'][split]
                for name,info in json.loads((OUT/f'cache_inventory_{split}.json').read_text())['files'].items():
                    assert sha(OUT/'activation_cache'/(name+'.npz'))==info['cache_sha256']
            from write_kard_adapter_report import report
            report(OUT,gate,results,baselines)
            status('complete',selected_checkpoints=9,candidates=27,report='REPORT.md')
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',{'pid':os.getpid(),'time_unix':time.time(),'traceback':traceback.format_exc()});raise
