"""Controlled KARD ablations on immutable features, with an all-selections test gate.

Run --preflight-only before launching --resume. No TensorFlow is imported.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import copy
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import random
import platform
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from kard_ablation_model import PARAMETERS, NAMES, initialize, state_digest, kernel_l2

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'kard_a0_head_100pct'
COMPACT = ROOT / 'kard_a0_compact_temporal_v1'
UPPER = ROOT / 'kard_a0_upper_finetune'
OUT = ROOT / 'kard_a0_controlled_ablation_v1'
SEEDS = (42, 123, 2026)
VARIANTS = ('A', 'B', 'C', 'D')
L2S = (.0001, .001, .01)
MANIFEST = json.loads((BASE / 'manifest.json').read_text())
LABELS = MANIFEST['labels']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save_torch(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def status(phase, **data):
    result = {'phase': phase, 'pid': os.getpid(), 'updated_unix': time.time(), **data}
    write_json(OUT / 'status.json', result)
    print(json.dumps(result), flush=True)


def protocol():
    return {'version': 1, 'dataset': 'KARD', 'seeds': list(SEEDS), 'variants': NAMES,
        'changed_part_parameters': PARAMETERS, 'l2_candidates': list(L2S),
        'manifest_sha256': sha(BASE / 'manifest.json'),
        'model_sha256': sha(ROOT / 'models/a0_stream/variables/variables.data-00000-of-00001'),
        'subject_split': MANIFEST['split_subjects'], 'split_counts': {'train': 324, 'validation': 108, 'test': 108},
        'preprocessing': MANIFEST['protocol']['preprocessing'],
        'optimizer': {'name': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
                      'batch_size': 16, 'global_gradient_norm_clip': 1., 'max_epochs': 200, 'patience': 25},
        'runtime': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                    'numpy': np.__version__, 'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                    'cudnn': torch.backends.cudnn.version(), 'deterministic_algorithms': True,
                    'tf32': False, 'torch_threads': 2},
        'loss': 'Cross entropy plus explicit L2 times summed squared Linear/Conv kernels; bias and LayerNorm excluded.',
        'normalization_104': 'Train only; equal clip weights for per-frame channel moments; scale floor .001; identical B/C.',
        'normalization_480': 'Train-only float32 mean/std with scale floor .001, independently compared to saved historical normalization.',
        'initialization': 'PyTorch defaults; identical B/C common modules per seed; C temporal RNG isolated. All variants share per-seed minibatch permutations and reset dropout RNG.',
        'selection': 'Checkpoint and L2 per variant per seed: highest validation macro F1, then lowest validation cross entropy; exact tie chooses earlier epoch/grid order. No seed selection.',
        'test_gate': 'No test cache access until all 12 variant/seed selections persisted with verified selected checkpoint SHA256. Evaluate only those checkpoints; no post-test tuning.',
        'attention': 'None. C uses a causal depthwise temporal convolution with hard-swish residual.',
        'frozen': 'All original stem and 15 bottlenecks frozen via immutable features. A/D retain frozen104-to480 projection; B/C bypass it.',
        'historical_baseline': 'Context only: TensorFlow CPU; fresh A is an original-style PyTorch control, not an exact historical reproduction (initialization, optimizer and clipping differ).',
        'limits': 'Previously observed subject split; descriptive test only. Three-seed sample SD estimates training variability, not population CI or subject generalization. D matches changed-part capacity only, not full-model compute. No end-to-end speed claim.'}


def stem(record):
    return record['split'] + '_' + record['relative_path'].replace('\\', '_').replace('/', '_').removesuffix('.mp4')


def candidate_dir(variant, seed, l2):
    return OUT / 'candidates' / f'{variant}_seed_{seed}' / f'l2_{l2:g}'


def selection_key(variant, seed):
    return f'{variant}_seed_{seed}'


def validate_selection_gate():
    path = OUT / 'selection_manifest.json'
    if not path.exists():
        raise RuntimeError('Test access blocked: all twelve selections must be locked first.')
    gate = json.loads(path.read_text())
    assert gate['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert set(gate['selections']) == {selection_key(v, s) for v in VARIANTS for s in SEEDS}
    assert gate['all_selections_locked'] and not gate['test_used_for_selection']
    for item in gate['selections'].values():
        assert sha(OUT / item['checkpoint']) == item['checkpoint_sha256']
    return gate


def load_split(split, expected_inventory=None):
    # This is the single entry point for both representations. The gate executes
    # before any test cache sidecar, hash or array is read.
    if split == 'test':
        validate_selection_gate()
    records = [r for r in MANIFEST['records'] if r['split'] == split and r['selected']]
    expected_counts = {'train': 324, 'validation': 108, 'test': 108}
    expected_subjects = {'train': {1, 3, 4, 6, 7, 8}, 'validation': {5, 10}, 'test': {2, 9}}
    assert len(records) == expected_counts[split]
    assert {r['subject_id'] for r in records} == expected_subjects[split]
    assert np.bincount([r['class_id'] for r in records], minlength=18).tolist() == [len(records)//18]*18
    manifest_sha = sha(BASE / 'manifest.json')
    model_sha = sha(ROOT / 'models/a0_stream/variables/variables.data-00000-of-00001')
    upper_sha = sha(UPPER / 'protocol.json')
    wide, sequences, inventory = [], [], {}
    for r in records:
        name = stem(r)
        paths = {'wide': BASE / 'feature_cache' / (name + '.npz'),
                 'sequence': COMPACT / 'sequence_cache' / (name + '.npz')}
        assert sha(r['path']) == r['sha256']
        row = {'video_sha256': r['sha256'], 'subject': r['subject_id'], 'class_id': r['class_id']}
        infos = {}
        for kind, path in paths.items():
            info = json.loads(path.with_suffix('.json').read_text())
            infos[kind] = info
            assert info['video_sha256'] == r['sha256']
            assert info['manifest_sha256'] == manifest_sha
            assert info['model_sha256'] == model_sha
            digest = sha(path)
            if kind == 'sequence':
                assert info['cache_sha256'] == digest
                assert info['endpoint'] == 'b4/l3_spatial_mean'
            row[kind + '_cache_sha256'] = digest
            row[kind + '_sidecar_sha256'] = sha(path.with_suffix('.json'))
            with np.load(path, allow_pickle=False) as data:
                assert int(data['label']) == r['class_id'] and int(data['subject']) == r['subject_id']
                values = data['features' if kind == 'wide' else 'sequence'].copy()
            assert values.dtype == np.float32 and np.isfinite(values).all()
            if kind == 'wide':
                assert values.shape == (480,)
                wide.append(values)
            else:
                assert values.ndim == 2 and values.shape[1] == 104 and 0 < len(values) <= 50
                assert info['shape'] == list(values.shape)
                assert len(values) == info['metadata']['sampled_frames']
                sequences.append(values)
        # Confirm both representations use precisely the same sampled frames.
        assert infos['wide']['frame_indices'] == infos['sequence']['metadata']['frame_indices']
        assert infos['wide']['state_head_frame_count'] == len(sequences[-1])
        source = UPPER / 'activation_cache' / (name + '.npz')
        source_info = json.loads(source.with_suffix('.json').read_text())
        source_sha = sha(source)
        assert source_sha == infos['sequence']['source_activation_sha256'] == source_info['cache_sha256']
        assert source_info['source_sha256'] == r['sha256']
        assert source_info['manifest_sha256'] == manifest_sha
        assert source_info['original_backbone_file_sha256'] == model_sha
        assert source_info['protocol_sha256'] == upper_sha
        row['source_activation_sha256'] = source_sha
        inventory[name] = row
        if expected_inventory is not None:
            assert row == expected_inventory[name], f'Immutable cache changed: {name}'
    inventory_path = OUT / f'cache_inventory_{split}.json'
    if inventory_path.exists():
        assert json.loads(inventory_path.read_text())['files'] == inventory
    else:
        write_json(inventory_path, {'split': split, 'manifest_sha256': manifest_sha,
            'model_sha256': model_sha, 'files': inventory,
            'wide_hash_limit': 'Historical480D sidecars lack stored NPZ checksum; these independently established hashes detect changes from this audit onward, not retrospectively.',
            'sequence_stored_cache_hash_verified': True, 'source_activation_hash_verified': True})
    return {'wide': np.stack(wide), 'sequence': sequences,
            'labels': np.asarray([r['class_id'] for r in records], np.int64), 'records': records}


def normalization(train):
    wide = train['wide']
    wm, ws = wide.mean(0), np.maximum(wide.std(0), 1e-3)
    mean = np.mean([s.mean(0, dtype=np.float64) for s in train['sequence']], axis=0)
    variance = np.mean([np.square(s.astype(np.float64)-mean).mean(0) for s in train['sequence']], axis=0)
    sm, ss = mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-3).astype(np.float32)
    evidence = {}
    for kind, reference, current in [('480', BASE/'train_normalization.npz', (wm, ws)),
                                     ('104', COMPACT/'train_normalization.npz', (sm, ss))]:
        with np.load(reference) as ref:
            errors = {key: float(np.max(np.abs(values-ref[key]))) for key, values in zip(('mean', 'scale'), current)}
            assert all(np.array_equal(values, ref[key]) for key, values in zip(('mean', 'scale'), current))
        evidence[kind] = {'reference_sha256': sha(reference), 'max_abs_errors': errors, 'bitwise_equal': True}
        target = OUT / f'train_normalization_{kind}.npz'
        if target.exists():
            with np.load(target) as saved:
                assert np.array_equal(saved['mean'], current[0]) and np.array_equal(saved['scale'], current[1])
        else:
            np.savez(target, mean=current[0], scale=current[1])
    write_json(OUT / 'normalization_verification.json', evidence)
    return {'A': (wm, ws), 'D': (wm, ws), 'B': (sm, ss), 'C': (sm, ss)}


def tensors(data, device):
    sequences = data['sequence']
    padded = np.zeros((len(sequences), max(map(len, sequences)), 104), np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = sequence
    return {'wide': torch.as_tensor(data['wide'], device=device),
            'sequence': torch.as_tensor(padded, device=device),
            'lengths': torch.tensor([len(s) for s in sequences], device=device),
            'labels': torch.as_tensor(data['labels'], device=device)}


def logits(model, data, ids=None):
    if ids is None:
        ids = slice(None)
    return model(data['sequence'][ids], data['lengths'][ids]) if model.variant in ('B', 'C') else model(data['wide'][ids])


def metrics(values, labels):
    values, labels = np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=int)
    prediction = values.argmax(1)
    matrix = np.zeros((18, 18), dtype=int)
    np.add.at(matrix, (labels, prediction), 1)
    tp, actual, predicted = np.diag(matrix), matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(18), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(18), where=actual > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(18), where=precision+recall > 0)
    shifted = values-values.max(1, keepdims=True)
    ce = (np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)), labels]).mean()
    return {'n': len(labels), 'accuracy': float((prediction == labels).mean()), 'macro_f1': float(f1.mean()),
            'cross_entropy': float(ce), 'balanced_accuracy': float(recall.mean()), 'confusion_matrix': matrix.tolist(),
            'precision_per_class': precision.tolist(), 'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(), 'support_per_class': actual.tolist()}


def predict(model, data):
    model.eval()
    with torch.no_grad():
        return torch.cat([logits(model, data, slice(i, i+32)).cpu()
                          for i in range(0, len(data['labels']), 32)]).numpy()


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dropout_rng_digest():
    values = {'cpu': torch.get_rng_state()}
    values.update({f'cuda_{i}': state for i, state in enumerate(torch.cuda.get_rng_state_all())})
    return state_digest(values)


def rng_state(rng):
    return {'numpy_generator': copy.deepcopy(rng.bit_generator.state), 'python': random.getstate(),
            'numpy_global': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state, rng):
    rng.bit_generator.state = state['numpy_generator']
    random.setstate(state['python'])
    np.random.set_state(state['numpy_global'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def train_step(model, optimizer, data, ids, l2):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits(model, data, ids), data['labels'][ids]) + l2*kernel_l2(model)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def preflight(train, norms, device):
    evidence = {'device': str(device), 'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
                'test_cache_accesses': 0, 'variants': {}, 'common_initialization': {}}
    for seed in SEEDS:
        reset_rng(seed)
        before_rng = dropout_rng_digest()
        b = initialize('B', *norms['B'], seed)
        c = initialize('C', *norms['C'], seed)
        assert dropout_rng_digest() == before_rng
        common = {k: v for k, v in c.state_dict().items() if not k.startswith('temporal.')}
        assert all(torch.equal(b.state_dict()[k], v) for k, v in common.items())
        evidence['common_initialization'][str(seed)] = {'B_sha256': state_digest(b.state_dict()),
            'C_common_sha256': state_digest(common), 'exact_equal': True,
            'temporal_sha256': state_digest({k: v for k, v in c.state_dict().items() if k.startswith('temporal.')}),
            'initialization_preserves_cpu_cuda_rng': True,
            'training_start_dropout_rng_sha256': before_rng}
    # Six real clips with mixed lengths and labels, no synthetic training data.
    ids = sorted(set([0, 1, 3, len(train['sequence'])//3, len(train['sequence'])//2, len(train['sequence'])-1]))
    small = {'wide': train['wide'][ids], 'sequence': [train['sequence'][i] for i in ids],
             'labels': train['labels'][ids]}
    data = tensors(small, device)
    for variant in VARIANTS:
        reset_rng(42)
        model = initialize(variant, *norms[variant], 42).to(device)
        reset_rng(42)
        initial = cpu_state(model)
        model.eval()
        with torch.no_grad():
            reference = logits(model, data)
            padding_error = causality_error = 0.
            if variant in ('B', 'C'):
                singles = torch.cat([model(data['sequence'][i:i+1, :int(data['lengths'][i])], data['lengths'][i:i+1]) for i in range(len(ids))])
                extended = F.pad(data['sequence'], (0, 0, 0, 7), value=987.)
                padding_error = max(float((reference-singles).abs().max()),
                                    float((reference-model(extended, data['lengths'])).abs().max()))
                assert padding_error < 3e-5
                prefix = int(data['lengths'].min())//2
                original, _ = model.temporal_features(data['sequence'], data['lengths'])
                changed = data['sequence'].clone()
                changed[:, prefix:] += 37.
                perturbed, _ = model.temporal_features(changed, data['lengths'])
                causality_error = float((original[:, :prefix]-perturbed[:, :prefix]).abs().max())
                assert causality_error == 0.
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        train_step(model, optimizer, data, slice(None), .001)
        gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
        changes = {n: float((p.detach().cpu()-initial[n]).abs().max()) for n, p in model.named_parameters()}
        assert all(v > 0 for v in gradients.values()) and all(v > 0 for v in changes.values())
        path = OUT / f'preflight_{variant}.pt'
        save_torch(path, {'state_dict': cpu_state(model)})
        restored = initialize(variant, *norms[variant], 42).to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state_dict'])
        model.eval(); restored.eval()
        with torch.no_grad():
            reload_error = float((logits(model, data)-logits(restored, data)).abs().max())
        assert reload_error == 0
        # Exact continuation with dropout, optimizer and all RNG restored.
        reset_rng(123)
        rng = np.random.default_rng(123)
        model = initialize(variant, *norms[variant], 123).to(device)
        reset_rng(123)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(2):
            train_step(model, optimizer, data, torch.as_tensor(rng.permutation(len(ids)), device=device), .001)
        saved = {'state_dict': cpu_state(model), 'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': rng_state(rng)}
        save_torch(OUT / f'preflight_resume_{variant}.pt', saved)
        uninterrupted_losses = []
        for _ in range(2):
            uninterrupted_losses.append(train_step(model, optimizer, data, torch.as_tensor(rng.permutation(len(ids)), device=device), .001))
        final_state = cpu_state(model)
        expected_rng = rng_state(rng)
        checkpoint = torch.load(OUT / f'preflight_resume_{variant}.pt', map_location=device, weights_only=False)
        resumed = initialize(variant, *norms[variant], 123).to(device)
        resumed.load_state_dict(checkpoint['state_dict'])
        ropt = torch.optim.Adam(resumed.parameters(), lr=.001)
        ropt.load_state_dict(checkpoint['optimizer'])
        rrng = np.random.default_rng()
        restore_rng(checkpoint['rng'], rrng)
        resumed_losses = []
        for _ in range(2):
            resumed_losses.append(train_step(resumed, ropt, data, torch.as_tensor(rrng.permutation(len(ids)), device=device), .001))
        resume_error = max(float((v-cpu_state(resumed)[k]).abs().max()) for k, v in final_state.items())
        assert resume_error == 0 and uninterrupted_losses == resumed_losses
        assert state_digest({'rng': torch.get_rng_state()}) == state_digest({'rng': expected_rng['torch']})
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(torch.cuda.get_rng_state_all(), expected_rng['cuda']))
        evidence['variants'][variant] = {'parameters': PARAMETERS[variant], 'padding_max_error': padding_error,
            'causal_prefix_max_error': causality_error, 'reload_max_error': reload_error,
            'resume_state_max_error': resume_error, 'resume_loss_exact': True,
            'gradient_norms': gradients, 'maximum_parameter_changes': changes}
    # Gate must fail before its first test sidecar read.
    if not (OUT/'selection_manifest.json').exists():
        try:
            load_split('test')
        except RuntimeError as error:
            assert 'Test access blocked' in str(error)
            evidence['test_gate_rejection_verified'] = True
        else:
            raise AssertionError('Test gate did not reject')
    evidence['runner_sha256'] = sha(Path(__file__))
    evidence['model_code_sha256'] = sha(ROOT/'kard_ablation_model.py')
    evidence['report_code_sha256'] = sha(ROOT/'write_kard_ablation_report.py')
    evidence['protocol_sha256'] = sha(OUT/'protocol.json')
    write_json(OUT / 'preflight.json', evidence)
    print('PREFLIGHT_PASSED', json.dumps(evidence), flush=True)


def train_candidate(variant, seed, l2, train, validation, norms, device):
    folder = candidate_dir(variant, seed, l2)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        assert summary['protocol_sha256'] == sha(OUT/'protocol.json')
        assert summary['checkpoint_sha256'] == sha(folder/'best.pt')
        return summary
    reset_rng(seed)
    model = initialize(variant, *norms[variant], seed).to(device)
    # Reset after construction: all candidates start the same dropout streams.
    reset_rng(seed)
    dropout_start_sha = dropout_rng_digest()
    checked = json.loads((OUT/'preflight.json').read_text())
    assert dropout_start_sha == checked['common_initialization'][str(seed)]['training_start_dropout_rng_sha256']
    initial_sha = state_digest(cpu_state(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    history, best_key, best_epoch, stale, start_epoch = [], (-1., -float('inf')), 0, 0, 1
    best_state = None
    last = folder/'last.pt'
    if last.exists():
        resume = torch.load(last, map_location=device, weights_only=False)
        assert resume['protocol_sha256'] == sha(OUT/'protocol.json')
        assert resume['initial_sha256'] == initial_sha
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['rng'], rng)
        history = resume['history']
        best_key, best_epoch, stale = tuple(resume['best_key']), resume['best_epoch'], resume['stale']
        best_state = {k: v.cpu() for k, v in resume['best_state'].items()}
        start_epoch = resume['epoch']+1
    for epoch in range(start_epoch, 201):
        if stale >= 25:
            break
        permutation = rng.permutation(len(train['labels']))
        losses = [train_step(model, optimizer, train, torch.as_tensor(permutation[offset:offset+16], device=device), l2)
                  for offset in range(0, len(permutation), 16)]
        tm = metrics(predict(model, train), train['labels'].cpu().numpy())
        vm = metrics(predict(model, validation), validation['labels'].cpu().numpy())
        key = (vm['macro_f1'], -vm['cross_entropy'])
        entry = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train': tm, 'validation': vm,
                 'permutation_sha256': hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()}
        history.append(entry)
        if key > best_key:
            best_key, best_epoch, stale, best_state = key, epoch, 0, cpu_state(model)
        else:
            stale += 1
        # Authoritative atomic epoch checkpoint includes history AND best weights,
        # so interruption cannot leave mixed-epoch history/best/optimizer files.
        save_torch(last, {'state_dict': cpu_state(model), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'best_key': best_key, 'best_epoch': best_epoch, 'stale': stale,
            'best_state': best_state, 'history': history, 'rng': rng_state(rng),
            'initial_sha256': initial_sha, 'protocol_sha256': sha(OUT/'protocol.json')})
        write_json(folder/'history.json', history)
        save_torch(folder/'best.pt', {'state_dict': best_state, 'variant': variant, 'seed': seed,
            'l2': l2, 'epoch': best_epoch, 'labels': LABELS})
        if epoch == 1 or epoch % 5 == 0:
            status('training', variant=variant, seed=seed, l2=l2, epoch=epoch,
                   validation_macro_f1=vm['macro_f1'], best_epoch=best_epoch)
    assert best_state is not None
    model.load_state_dict(best_state)
    save_torch(folder/'best.pt', {'state_dict': best_state, 'variant': variant, 'seed': seed,
        'l2': l2, 'epoch': best_epoch, 'labels': LABELS})
    write_json(folder/'history.json', history)
    summary = {'variant': variant, 'seed': seed, 'l2': l2, 'best_epoch': best_epoch,
        'epochs_executed': len(history), 'initial_sha256': initial_sha,
        'training_start_dropout_rng_sha256': dropout_start_sha,
        'checkpoint': str((folder/'best.pt').relative_to(OUT)).replace('\\', '/'),
        'checkpoint_sha256': sha(folder/'best.pt'), 'protocol_sha256': sha(OUT/'protocol.json'),
        'train': metrics(predict(model, train), train['labels'].cpu().numpy()),
        'validation': metrics(predict(model, validation), validation['labels'].cpu().numpy())}
    write_json(summary_path, summary)
    return summary


def select_all(train, validation, norms, device):
    if (OUT/'selection_manifest.json').exists():
        return validate_selection_gate()
    selections = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            candidates = [train_candidate(variant, seed, l2, train, validation, norms, device) for l2 in L2S]
            winner = max(candidates, key=lambda item: (item['validation']['macro_f1'], -item['validation']['cross_entropy']))
            selected_path = OUT/'selected'/f'{selection_key(variant, seed)}.pt'
            selected_path.parent.mkdir(exist_ok=True)
            selected = torch.load(OUT/winner['checkpoint'], map_location='cpu', weights_only=True)
            save_torch(selected_path, selected)
            item = {**winner, 'candidate_checkpoint': winner['checkpoint'],
                    'checkpoint': str(selected_path.relative_to(OUT)).replace('\\', '/'),
                    'checkpoint_sha256': sha(selected_path), 'candidates': candidates}
            selections[selection_key(variant, seed)] = item
            write_json(selected_path.with_suffix('.json'), item)
            status('selection_locked', variant=variant, seed=seed, selections_locked=len(selections), selections_total=12)
    gate = {'all_selections_locked': True, 'test_used_for_selection': False,
            'protocol_sha256': sha(OUT/'protocol.json'), 'locked_unix': time.time(), 'selections': selections}
    write_json(OUT/'selection_manifest.json', gate)
    return validate_selection_gate()


def evaluate_selected(gate, norms, device):
    test = load_split('test')
    td = tensors(test, device)
    results = {}
    for key, selected in gate['selections'].items():
        folder = OUT/'evaluation'/key
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder/'metrics.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result['checkpoint_sha256'] == selected['checkpoint_sha256']
            assert result['selection_manifest_sha256'] == sha(OUT/'selection_manifest.json')
            results[key] = result
            continue
        checkpoint = torch.load(OUT/selected['checkpoint'], map_location=device, weights_only=True)
        model = initialize(selected['variant'], *norms[selected['variant']], selected['seed']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        values = predict(model, td)
        np.savez(folder/'test_logits.npz', logits=values, labels=test['labels'])
        with (folder/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['video', 'subject', 'true_id', 'true_class', 'predicted_id', 'predicted_class', 'correct'])
            for r, predicted in zip(test['records'], values.argmax(1)):
                writer.writerow([r['relative_path'], r['subject_id'], r['class_id'], r['class_name'], predicted,
                                 LABELS[predicted], int(predicted == r['class_id'])])
        result = {'variant': selected['variant'], 'seed': selected['seed'], 'test': metrics(values, test['labels']),
                  'checkpoint_sha256': selected['checkpoint_sha256'],
                  'selection_manifest_sha256': sha(OUT/'selection_manifest.json'), 'test_evaluated_after_all_selections': True}
        write_json(result_path, result)
        results[key] = result
    write_json(OUT/'evaluation.json', results)
    return results


@contextlib.contextmanager
def single_process_lock():
    # Windows named mutex automatically releases if a crashed process exits.
    name = 'Local\\KARD_Ablation_' + hashlib.sha256(str(OUT.resolve()).encode()).hexdigest()[:24]
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError()
    try:
        if kernel.GetLastError() == 183:
            raise RuntimeError('Another ablation process already owns the experiment lock.')
        yield
    finally:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    with single_process_lock():
        proposed = protocol()
        if (OUT/'protocol.json').exists():
            assert json.loads((OUT/'protocol.json').read_text()) == proposed
        else:
            write_json(OUT/'protocol.json', proposed)
            (OUT/'manifest.json').write_bytes((BASE/'manifest.json').read_bytes())
            write_json(OUT/'labels.json', LABELS)
        if not args.preflight_only:
            checked = json.loads((OUT/'preflight.json').read_text())
            assert checked['runner_sha256'] == sha(Path(__file__))
            assert checked['model_code_sha256'] == sha(ROOT/'kard_ablation_model.py')
            assert checked['report_code_sha256'] == sha(ROOT/'write_kard_ablation_report.py')
            assert checked['protocol_sha256'] == sha(OUT/'protocol.json')
            if (OUT/'status.json').exists() and not args.resume:
                raise RuntimeError('Use --resume for existing experiment.')
        awake = False
        try:
            if not args.preflight_only:
                awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
                status('startup_verified', gpu=torch.cuda.get_device_name(0), sleep_prevention_requested=awake,
                       monitor_policy='Check only when user asks.')
            train_data = load_split('train')
            validation_data = load_split('validation')
            norms = normalization(train_data)
            if args.preflight_only:
                preflight(train_data, norms, device)
                return
            train, validation = tensors(train_data, device), tensors(validation_data, device)
            gate = select_all(train, validation, norms, device)
            status('evaluating_locked_selections', selections=12)
            evaluation = evaluate_selected(gate, norms, device)
            # Re-hash train/validation inputs to detect mutation during the run.
            for split in ('train', 'validation'):
                inventory = json.loads((OUT/f'cache_inventory_{split}.json').read_text())['files']
                load_split(split, inventory)
            from write_kard_ablation_report import report
            report(OUT, gate, evaluation, LABELS, PARAMETERS, NAMES, SEEDS)
            status('complete', selected_checkpoints=12, candidates=36, report='REPORT.md')
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            write_json(OUT/'failure.json', {'pid': os.getpid(), 'time_unix': time.time(), 'traceback': traceback.format_exc()})
        raise
