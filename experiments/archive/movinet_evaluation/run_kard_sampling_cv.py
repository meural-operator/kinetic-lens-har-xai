"""Prespecified five subject folds x three seeds, paired frozen A0 heads."""
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kard_sampling_cv'
OUT.mkdir(exist_ok=True)
(OUT / 'tmp').mkdir(exist_ok=True)
from kard_model import tf, np, build_head, metrics
from kard_config import LABELS
tempfile.tempdir = str(OUT / 'tmp')
tf.config.experimental.enable_op_determinism()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def immutable(path, data):
    if path.exists():
        assert read(path) == data, f'Existing experiment differs: {path}'
    else:
        write(path, data)

def load_verified_features():
    sources = {5: ROOT/'kard_a0_head_100pct', 10: ROOT/'kard_a0_head_100pct_10fps'}
    manifests = {fps: read(folder/'manifest.json') for fps, folder in sources.items()}
    records = manifests[5]['records']
    assert records == manifests[10]['records'] and len(records) == 540
    assert len({r['sha256'] for r in records}) == 540
    assert len({r['relative_path'] for r in records}) == 540
    # Stateless extraction from a fixed pretrained backbone is split independent.
    # No old standardization, labels-trained heads or predictions are reused.
    for r in records:
        assert digest(Path(r['path'])) == r['sha256']
    official_hash = digest(ROOT/'models/a0_stream/variables/variables.data-00000-of-00001')
    rows = []
    features = {}
    for fps, folder in sources.items():
        mhash = digest(folder/'manifest.json')
        protocol = manifests[fps]['protocol']
        assert protocol['target_fps'] == fps and protocol['max_frames'] == fps*10
        e = read(folder/'evaluation.json')
        assert e['export_verification']['backbone_weights_unchanged']
        arrays = []
        for r in records:
            stem = r['split']+'_'+r['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')
            cache = folder/'feature_cache'/stem
            meta_path, data_path = cache.with_suffix('.json'), cache.with_suffix('.npz')
            meta = read(meta_path)
            assert meta['manifest_sha256'] == mhash and meta['video_sha256'] == r['sha256']
            assert meta['model_sha256'] == official_hash
            if fps == 10:
                assert meta['target_fps'] == 10 and meta['same_window_as_5fps_verified']
                assert meta['5fps_indices_match_every_other_10fps_frame']
            with np.load(data_path) as data:
                assert int(data['label']) == r['class_id'] and int(data['subject']) == r['subject_id']
                feature = data['features'].copy()
            assert feature.shape == (480,) and np.isfinite(feature).all()
            arrays.append(feature)
            rows.append({'fps': fps, 'video': r['relative_path'], 'video_sha256': r['sha256'],
                         'cache': str(data_path.relative_to(ROOT)), 'cache_sha256': digest(data_path),
                         'metadata_sha256': digest(meta_path)})
        features[fps] = np.stack(arrays)
    audit = {'clips': 540, 'feature_vectors': 1080, 'all_source_hashes_verified': True,
             'fixed_official_backbone_sha256': official_hash,
             'reuse': 'Only raw split-independent frozen pretrained features; no previous normalization or trained head.',
             'records': rows}
    immutable(OUT/'feature_provenance.json', audit)
    return records, features

def make_protocol(records):
    pairs = [[1,2], [3,4], [5,6], [7,8], [9,10]]
    folds = []
    subjects = np.array([r['subject_id'] for r in records])
    labels = np.array([r['class_id'] for r in records], dtype=np.int32)
    for i, test in enumerate(pairs):
        val = pairs[(i+1) % 5]
        train = sorted(set(range(1,11))-set(test)-set(val))
        assignments = {'train': train, 'validation': val, 'test': test}
        ids = {split: np.flatnonzero(np.isin(subjects, people)).tolist() for split, people in assignments.items()}
        assert set(ids['train']).isdisjoint(ids['validation']) and set(ids['train']).isdisjoint(ids['test'])
        assert set(ids['validation']).isdisjoint(ids['test'])
        assert sorted(ids['train']+ids['validation']+ids['test']) == list(range(540))
        for split, n in [('train',18), ('validation',6), ('test',6)]:
            assert np.all(np.bincount(labels[ids[split]], minlength=18) == n)
        folds.append({'fold': i+1, 'subjects': assignments, 'record_indices': ids})
    assert sorted(j for f in folds for j in f['record_indices']['test']) == list(range(540))
    protocol = {'version': 1, 'model': 'Official frozen MoViNet A0-Stream', 'labels': LABELS,
        'rates_fps': [5,10], 'frame_caps': [50,100], 'maximum_window_seconds': 10,
        'resolution': 172, 'feature_dimension': 480, 'head_trainable_parameters': 8658,
        'training_seeds': [42,123,2026], 'folds': folds,
        'subject_partition_rule': 'Sorted consecutive subject pairs; test pair rotates once, validation is the next pair cyclically, remaining six subjects train. Fixed before these training/test runs.',
        'normalization': 'Separate mean and std (floor 0.001) fitted only on each fold/rate training features.',
        'head': 'Dropout 0.25, Dense 18 logits, L2 kernel penalty; initialized afresh for every fold/rate/seed/candidate.',
        'optimizer': 'Adam learning rate 0.001; batch 16; seeded shuffle; maximum 200 epochs; patience 25.',
        'l2_candidates': [0.0001,0.001,0.01],
        'selection': 'Checkpoint and L2 by validation macro F1, tie lower validation cross entropy; sampling recommendation by the same rule.',
        'phase_order': 'All 30 head and 15 sampling selections frozen and hashed before any new-fold test predictions.',
        'previous_dataset_and_test_results_observed': True,
        'limitations': 'Exploratory researcher-defined protocol, not an official KARD benchmark or fresh external test. Seeds share data; folds overlap in training. Descriptive variation only, no significance/CI claims. No augmentation, backbone adaptation or new privacy technique.',
        'code_sha256': digest(Path(__file__)), 'feature_provenance_sha256': digest(OUT/'feature_provenance.json')}
    immutable(OUT/'protocol.json', protocol)
    return protocol, labels

def fit_run(folder, x, y, vx, vy, seed, context):
    if (folder/'selection.json').exists():
        s = read(folder/'selection.json')
        assert s['context'] == context and s['protocol_sha256'] == digest(OUT/'protocol.json')
        assert s['selected_head_sha256'] == digest(folder/'kard_head.keras')
        return s
    folder.mkdir(parents=True, exist_ok=True)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-3)
    np.savez(folder/'train_normalization.npz', mean=mean, scale=scale)
    candidates = []
    start = time.perf_counter()
    for l2 in [0.0001,0.001,0.01]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        head = build_head(mean, scale, l2)
        initial = [v.numpy().copy() for v in head.trainable_variables]
        initial_hash = hashlib.sha256(b''.join(v.tobytes() for v in initial)).hexdigest()
        optimizer = tf.keras.optimizers.Adam(0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        rng = np.random.default_rng(seed)
        best_key = (-1., float('-inf'))
        stale, best_epoch = 0, 0
        history = []
        @tf.function(reduce_retracing=True)
        def step(bx, by):
            with tf.GradientTape() as tape:
                logits = head(bx, training=True)
                loss = loss_fn(by, logits) + tf.add_n(head.losses)
            gradients = tape.gradient(loss, head.trainable_variables)
            for g in gradients:
                tf.debugging.assert_all_finite(g, 'Nonfinite head gradient')
            optimizer.apply_gradients(zip(gradients, head.trainable_variables))
            return loss
        for epoch in range(1,201):
            ids = rng.permutation(len(y))
            losses = [float(step(x[ids[j:j+16]], y[ids[j:j+16]])) for j in range(0,len(y),16)]
            train = metrics(head(x, training=False).numpy(), y)
            val = metrics(head(vx, training=False).numpy(), vy)
            history.append({'epoch': epoch, 'loss': float(np.mean(losses)),
                'train_accuracy': train['accuracy'], 'train_macro_f1': train['macro_f1'],
                'validation_accuracy': val['accuracy'], 'validation_macro_f1': val['macro_f1'],
                'validation_cross_entropy': val['cross_entropy']})
            key = (val['macro_f1'], -val['cross_entropy'])
            if key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best_weights = head.get_weights()
            else:
                stale += 1
            if stale >= 25:
                break
        delta = max(float(np.max(np.abs(v.numpy()-old))) for v, old in zip(head.trainable_variables, initial))
        assert delta > 1e-5
        head.set_weights(best_weights)
        checkpoint = folder/f'candidate_l2_{l2:g}.keras'
        head.save(checkpoint)
        write(folder/f'history_l2_{l2:g}.json', history)
        candidates.append({'l2': l2, 'checkpoint': checkpoint.name, 'best_epoch': best_epoch,
            'epochs_executed': len(history), 'optimizer_steps': int(optimizer.iterations.numpy()),
            'maximum_trainable_weight_change': delta, 'initial_trainable_sha256': initial_hash,
            'train': metrics(head(x, training=False).numpy(), y),
            'validation': metrics(head(vx, training=False).numpy(), vy)})
    assert len({c['initial_trainable_sha256'] for c in candidates}) == 1
    winner = max(candidates, key=lambda c:(c['validation']['macro_f1'], -c['validation']['cross_entropy']))
    selected = tf.keras.models.load_model(folder/winner['checkpoint'])
    selected.save(folder/'kard_head.keras')
    before = selected(vx, training=False).numpy()
    reloaded = tf.keras.models.load_model(folder/'kard_head.keras')
    reload_error = float(np.max(np.abs(before-reloaded(vx, training=False).numpy())))
    assert reload_error < 1e-6
    selection = {'context': context, 'protocol_sha256': digest(OUT/'protocol.json'),
        'test_used_for_training_or_selection': False, 'normalization_fit_clips': len(y),
        'selected_head_sha256': digest(folder/'kard_head.keras'), 'head_reload_max_logit_error': reload_error,
        'winner': winner, 'candidates': candidates, 'training_seconds': time.perf_counter()-start,
        'tensorflow_version': tf.__version__, 'device': 'CPU'}
    write(folder/'selection.json', selection)
    print('HEAD_COMPLETE', json.dumps({**context, 'epoch': winner['best_epoch'], 'l2': winner['l2'],
        'validation_macro_f1': winner['validation']['macro_f1']}), flush=True)
    return selection

def main():
    records, features = load_verified_features()
    protocol, labels = make_protocol(records)
    selections = []
    recommendations = []
    start = time.perf_counter()
    for fold in protocol['folds']:
        ids = fold['record_indices']
        for seed in protocol['training_seeds']:
            pair = []
            for fps in protocol['rates_fps']:
                context = {'fold': fold['fold'], 'seed': seed, 'fps': fps, 'subjects': fold['subjects']}
                folder = OUT/f"fold_{fold['fold']}_seed_{seed}_fps_{fps}"
                s = fit_run(folder, features[fps][ids['train']], labels[ids['train']],
                            features[fps][ids['validation']], labels[ids['validation']], seed, context)
                pair.append(s)
                selections.append({'folder': folder.name, 'selection_sha256': digest(folder/'selection.json'),
                                   'head_sha256': digest(folder/'kard_head.keras')})
            assert pair[0]['winner']['initial_trainable_sha256'] == pair[1]['winner']['initial_trainable_sha256']
            selected_rate = max(pair, key=lambda s:(s['winner']['validation']['macro_f1'],
                                                   -s['winner']['validation']['cross_entropy']))['context']['fps']
            recommendations.append({'fold': fold['fold'], 'seed': seed, 'validation_recommended_fps': selected_rate})
    lock = {'protocol_sha256': digest(OUT/'protocol.json'), 'heads': selections, 'sampling_choices': recommendations,
            'test_used_for_selection': False, 'all_selections_completed_before_new_test_scoring': True}
    immutable(OUT/'selection_lock.json', lock)
    print('ALL_SELECTIONS_LOCKED', flush=True)
    all_rows = []
    for selected in selections:
        folder = OUT/selected['folder']
        assert digest(folder/'selection.json') == selected['selection_sha256']
        assert digest(folder/'kard_head.keras') == selected['head_sha256']
        s = read(folder/'selection.json')
        context = s['context']
        fold, fps = protocol['folds'][context['fold']-1], context['fps']
        ids = fold['record_indices']['test']
        head = tf.keras.models.load_model(folder/'kard_head.keras')
        logits = head(features[fps][ids], training=False).numpy()
        test = metrics(logits, labels[ids])
        assert test['support_per_class'] == [6]*18
        row = {**{k: context[k] for k in ['fold','seed','fps']},
               'validation_accuracy': s['winner']['validation']['accuracy'],
               'validation_macro_f1': s['winner']['validation']['macro_f1'],
               'test_accuracy': test['accuracy'], 'test_macro_f1': test['macro_f1'],
               'selected_epoch': s['winner']['best_epoch'], 'selected_l2': s['winner']['l2']}
        predictions = [{'video': records[j]['relative_path'], 'subject': records[j]['subject_id'],
                        'true_class': int(labels[j]), 'predicted_class': int(logits[k].argmax())} for k,j in enumerate(ids)]
        result = {'context': context, 'test': test, 'selection_lock_sha256': digest(OUT/'selection_lock.json'),
                  'test_used_for_selection': False, 'predictions': predictions}
        immutable(folder/'evaluation.json', result)
        np.savez(folder/'test_logits.npz', logits=logits, labels=labels[ids], record_indices=ids)
        all_rows.append(row)
        print('TEST_COMPLETE', json.dumps(row), flush=True)
    with (OUT/'run_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    write(OUT/'execution.json', {'runs_completed': len(all_rows), 'candidate_fits': 90,
          'protocol_sha256': digest(OUT/'protocol.json'), 'selection_lock_sha256': digest(OUT/'selection_lock.json'),
          'wall_seconds_this_invocation': time.perf_counter()-start, 'all_head_reload_checks_passed': True,
          'test_score_checks_passed': True})
    print('CV_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
