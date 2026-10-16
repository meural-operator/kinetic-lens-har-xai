"""Check real-weight identity, gradients and merge equivalence; no training run."""
import os
os.environ['KARD_TRAIN_FRACTION']='1.0'
import json
from pathlib import Path
import numpy as np
import torch
from kard_model import FrozenBackbone, tf, variable_digest
from kard_projection_adapter import ProjectionAdapter, classifier_basis, frozen_projection
from kard_evidence_pool import EvidencePool

root=Path(__file__).resolve().parent
out=root/'architecture_challenge_20261010'
f=FrozenBackbone()
p=getattr(f.backbone,'layer_with_weights-16')
variables={v.name:v.numpy() for v in p.variables}
kernel=torch.tensor(next(v for k,v in variables.items() if 'kernel' in k).reshape(104,480))
def bn(name):
    return torch.tensor(next(v for k,v in variables.items() if k.endswith('/'+name+':0')))
gamma,beta,mean,var=[bn(k) for k in ('gamma','beta','moving_mean','moving_variance')]
head=tf.keras.models.load_model(root/'kard_a0_head_100pct'/'kard_head.keras')
classifier=torch.tensor(head.get_layer('kard_classifier').kernel.numpy())
norm=np.load(root/'kard_a0_head_100pct'/'train_normalization.npz')
scale=torch.tensor(norm['scale'])
torch.manual_seed(2718)
basis=classifier_basis(classifier,scale,gamma/torch.sqrt(var+.001))
adapter=ProjectionAdapter(kernel,basis)
manifest=json.loads((root/'kard_a0_head_100pct'/'manifest.json').read_text())
r=next(r for r in manifest['records'] if r['split']=='train')
stem=r['split']+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')
cache=root/'kard_a0_upper_finetune'/'activation_cache'/(stem+'.npz')
a=tf.constant(np.load(cache)['activation'][None])
states=f.model.init_states(tf.constant([1,1,172,172,3]))
for i in (14,15):
    a,states=getattr(f.backbone,f'layer_with_weights-{i}')(a,states=states,training=False)
original,_=p(a,states=states,training=False)
x=torch.tensor(a.numpy())
baseline=frozen_projection(x,kernel,gamma,beta,mean,var).mean((1,2,3))
reference=torch.tensor(original.numpy().reshape(1,480))
reference_error=(baseline-reference).abs().max().item()
assert reference_error<1e-4,reference_error
assert torch.equal(adapter.merged_kernel(),kernel)
assert torch.equal(adapter(x),x@kernel)
adapted=frozen_projection(x,adapter.merged_kernel(),gamma,beta,mean,var).mean((1,2,3))
assert torch.equal(adapted,baseline)
assert variable_digest(f.backbone.variables)==f.initial_hash
target=torch.tensor([r['class_id']])
normal_mean=torch.tensor(norm['mean'])
classifier_bias=torch.tensor(head.get_layer('kard_classifier').bias.numpy())
logits=((adapted-normal_mean)/scale)@classifier+classifier_bias
loss=torch.nn.functional.cross_entropy(logits,target)
loss.backward()
grad_out=adapter.factor_out.grad.abs().sum().item()
grad_in=adapter.factor_in.grad.abs().sum().item()
assert grad_out>0 and grad_in==0
# Synthetic update is only a numerical merge check, not a trained checkpoint.
with torch.no_grad():
    adapter.factor_out.normal_(std=.01)
explicit=adapter(x)
merged=x@adapter.merged_kernel()
merge_error=(explicit-merged).abs().max().item()
assert torch.allclose(explicit,merged,rtol=1e-4,atol=1e-4)
# Verify the merged kernel in the real restored TensorFlow projection graph,
# then restore the pretrained variable even if a comparison fails.
kernel_variable=next(v for v in p.variables if 'kernel' in v.name)
saved_kernel=kernel_variable.numpy().copy()
try:
    kernel_variable.assign(adapter.merged_kernel().detach().numpy().reshape(saved_kernel.shape))
    merged_reference,_=p(a,states=states,training=False)
    merged_features=frozen_projection(x,adapter.merged_kernel(),gamma,beta,mean,var).mean((1,2,3))
    restored_graph_error=(merged_features-torch.tensor(merged_reference.numpy().reshape(1,480))).abs().max().item()
    assert restored_graph_error<1e-4
finally:
    kernel_variable.assign(saved_kernel)
assert variable_digest(f.backbone.variables)==f.initial_hash
loss2=adapter(x).square().mean()
adapter.zero_grad();loss2.backward()
assert adapter.factor_in.grad.abs().sum()>0
arrays=np.load(out/'train_validation_features.npz')
lengths=torch.tensor(arrays['lengths']);frames=torch.tensor(arrays['frames'])
evidence=((frames-normal_mean)/scale)@classifier+classifier_bias
padded=torch.zeros(len(lengths),int(lengths.max()),18)
offset=0
for i,n in enumerate(lengths):
    padded[i,:n]=evidence[offset:offset+n];offset+=n
wide=torch.tensor(arrays['wide'])
original_logits=((wide-normal_mean)/scale)@classifier+classifier_bias
pool=EvidencePool(np.zeros(36),np.ones(36))
pooled,weights=pool(original_logits,padded,lengths,return_weights=True)
assert torch.equal(pooled,original_logits)
assert torch.allclose(weights.sum(1),torch.ones(len(lengths)))
assert torch.isfinite(pooled).all()
result={'test_accessed':False,'real_train_clip':r['relative_path'],
    'classifier_basis_dimension':basis.shape[1],
    'projection_adapter_trainable_parameters':sum(p.numel() for p in adapter.parameters()),
    'standard_rank4_LoRA_parameters':4*(104+480),
    'frozen_projection_vs_original_max_error':reference_error,
    'identity_kernel_bitwise':True,'identity_output_bitwise':True,
    'factor_out_gradient_at_initialization':grad_out,
    'factor_in_gradient_at_initialization':grad_in,
    'factor_in_gradient_after_nonzero_update':float(adapter.factor_in.grad.abs().sum()),
    'synthetic_nonzero_update_merge_max_error':merge_error,
    'synthetic_merged_kernel_vs_real_saved_graph_feature_max_error':restored_graph_error,
    'merged_kernel_shape':list(adapter.merged_kernel().shape),
    'backbone_unchanged':True,
    'evidence_pool_identity_clips':len(lengths),
    'evidence_pool_identity_bitwise':True,
    'accuracy_improvement_tested':False}
(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
np.savez(out/'projection_adapter_initialization.npz',kernel=kernel.numpy(),basis=basis.numpy(),
    factor_in=adapter.factor_in.detach().numpy(),factor_out=np.zeros_like(adapter.factor_out.detach().numpy()))
print(json.dumps(result,indent=2),flush=True)
