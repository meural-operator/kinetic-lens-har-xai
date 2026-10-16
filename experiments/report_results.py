"""Verify published evidence and recompute reported aggregates without training.

Uses the Python standard library only. No videos, model downloads or GPU needed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def csv_rows(path):
    with (ROOT / path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'{label}: {actual} != {expected}')


def aggregate(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def verify(require_private=False):
    publication = read('docs/research/publication_manifest.json')
    private = read('models/private_artifacts.json')
    private_entries = {entry['path']: entry for entry in private['files']}
    assert len(private_entries) == 26 and private['visibility_requirement'] == 'private'
    verified_private = 0
    if require_private:
        for entry in private_entries.values():
            path = (ROOT / entry['path']).resolve()
            assert path.is_relative_to(ROOT), entry['path']
            content = path.read_bytes()
            assert len(content) == entry['bytes'] and hashlib.sha256(content).hexdigest() == entry['sha256'], entry['path']
            verified_private += 1
    public_count = private_record_count = 0
    for record in publication['records']:
        if record.get('distribution') == 'private_huggingface':
            entry = private_entries[record['published']]
            assert record['published_sha256'] == entry['sha256']
            assert record['published_bytes'] == entry['bytes']
            assert record['storage'] == {k: private[k] for k in ('repo_id', 'repo_type', 'revision')} | {'hf_path': entry['hf_path']}
            private_record_count += 1
            continue
        path = (ROOT / record['published']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Published artifact escapes repository')
        content = path.read_bytes()
        if len(content) != record['published_bytes'] or hashlib.sha256(content).hexdigest() != record['published_sha256']:
            raise ValueError(f'Published artifact changed: {record["published"]}')
        public_count += 1

    sampling = read('docs/research/sampling/summary.json')
    runs = csv_rows('docs/research/sampling/run_results.csv')
    assert len(runs) == 30
    assert {(int(r['fold']), int(r['seed']), int(r['fps'])) for r in runs} == {
        (fold, seed, fps) for fold in range(1, 6) for seed in (42, 123, 2026) for fps in (5, 10)}
    for rate in sampling['rates']:
        for metric in ('validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1'):
            values = [100 * statistics.mean(float(r[metric]) for r in runs
                      if int(r['fold']) == fold and int(r['fps']) == rate['fps']) for fold in range(1, 6)]
            close(statistics.mean(values), rate[metric + '_mean_pct'], 'sampling mean')
            close(statistics.stdev(values), rate[metric + '_fold_sd_pp'], 'sampling fold SD')

    ablation = read('docs/research/architecture-ablations/summary.json')
    aruns = csv_rows('docs/research/architecture-ablations/per_seed_metrics.csv')
    assert len(aruns) == 12
    for variant in 'ABCD':
        for split in ('validation', 'test'):
            for metric in ('accuracy', 'macro_f1', 'cross_entropy'):
                values = [float(r[split + '_' + metric]) for r in aruns if r['variant'] == variant]
                assert len(values) == 3
                close(statistics.mean(values), ablation[variant][split][metric]['mean'], 'ablation mean')
                close(statistics.stdev(values), ablation[variant][split][metric]['sample_sd'], 'ablation seed SD')

    adapters = read('docs/research/projection-adapters/summary.json')
    pruns = csv_rows('docs/research/projection-adapters/per_seed_metrics.csv')
    assert len(pruns) == 12
    for variant in ('baseline', 'lora', 'guided', 'random'):
        for split, metrics in [('validation', ('macro_f1',)), ('test', ('macro_f1', 'accuracy', 'cross_entropy'))]:
            for metric in metrics:
                values = [float(r[split + '_' + metric]) for r in pruns if r['variant'] == variant]
                assert len(values) == 3
                close(statistics.mean(values), adapters[variant][split][metric]['mean'], 'adapter mean')
                close(statistics.stdev(values), adapters[variant][split][metric]['sample_sd'], 'adapter seed SD')
    for study, variants in [('architecture-ablations', ('A', 'B', 'C', 'D')),
                            ('projection-adapters', ('lora', 'guided', 'random'))]:
        gate = read(f'docs/research/{study}/selection_manifest.json')
        assert gate['all_selections_locked'] and not gate['test_used_for_selection']
        assert len(gate['selections']) == len(variants) * 3
        for variant in variants:
            for seed in (42, 123, 2026):
                item = gate['selections'][f'{variant}_seed_{seed}']
                checkpoint = ROOT / 'docs/research' / study / item['checkpoint']
                relative = checkpoint.relative_to(ROOT).as_posix()
                assert private_entries[relative]['sha256'] == item['checkpoint_sha256']
                if require_private:
                    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == item['checkpoint_sha256']

    checked_timings = 0
    for session in ('latency_benchmark', 'latency_benchmark_fp16'):
        data = read(f'docs/benchmarks/vit-bilstm/{session}/summary.json')
        for model in data['models']:
            for backend, stats in model['stats'].items():
                values = [r['latency_ms'] for r in model['timing_observations'] if r['backend'] == backend]
                assert len(values) == stats['n'] == 45
                close(statistics.mean(values), stats['mean_ms'], 'latency mean')
                close(statistics.stdev(values), stats['std_ms'], 'latency call SD')
                checked_timings += len(values)
    sizes = csv_rows('docs/benchmarks/size-latency/model_size_latency.csv')
    assert len(sizes) == 31
    assert sum(bool(r['artifact_bytes']) for r in sizes) == 26
    for row in sizes:
        if row['artifact_bytes']:
            close(int(row['artifact_bytes']) / 1e6, float(row['artifact_mb']), 'decimal MB')
    identity = read('models/kard/provenance.json')
    assert identity['source_head_sha256'] == 'e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70'
    assert identity['validation_correct'] == 96 and identity['test_correct'] == 94
    result = {'published_files_verified': public_count, 'private_publication_records_checked': private_record_count,
              'private_artifact_records_checked': len(private_entries), 'private_files_verified': verified_private,
              'sampling_runs': 30,
              'ablation_selected_runs': 12, 'adapter_rows_including_baselines': 12,
              'selected_checkpoint_hash_records_checked': 21,
              'serialized_selected_checkpoints_verified': 21 if require_private else 0,
              'raw_vit_timing_calls_verified_including_superseded_gpu_session': checked_timings,
              'size_latency_forms': 31, 'serialized_inference_artifact_sizes': 26,
              'training_or_camera_executed': False}
    return result, sampling, ablation, adapters


def summary_document(result, sampling, ablation, adapters):
    def fmt(stats, percent=True):
        factor = 100 if percent else 1
        return f"{factor * stats['mean']:.2f} ± {factor * stats['sample_sd']:.2f}"
    lines = ['# Independently recomputed study summary', '',
             'Generated with `python experiments/report_results.py --output docs/research/VERIFIED_SUMMARY.md`.',
             'This verifies public archived results and hash records; it does not retrain models or test a camera.',
             'Private checkpoints are excluded from the default offline verification. Use `--require-private` '
             'after authenticated setup to verify their actual bytes; otherwise their manifest identities alone are checked.', '',
             '## Temporal sampling', '',
             'Five fold means; three seeds averaged within each fold. Percentages, mean ± fold SD.', '',
             '| FPS | Test accuracy | Test macro F1 |', '|---|---:|---:|']
    for rate in sampling['rates']:
        lines.append(f"| {rate['fps']} | {rate['test_accuracy_mean_pct']:.2f} ± {rate['test_accuracy_fold_sd_pp']:.2f} | "
                     f"{rate['test_macro_f1_mean_pct']:.2f} ± {rate['test_macro_f1_fold_sd_pp']:.2f} |")
    lines += ['', '## Architecture ablations', '', 'Percentages, mean ± training-seed SD on the same fixed split.', '',
              '| Variant | Test accuracy | Test macro F1 |', '|---|---:|---:|']
    for variant in 'ABCD':
        lines.append(f"| {variant} | {fmt(ablation[variant]['test']['accuracy'])} | {fmt(ablation[variant]['test']['macro_f1'])} |")
    lines += ['', '## Projection adapters', '',
              '| Variant | Test accuracy | Test macro F1 |', '|---|---:|---:|']
    for variant in ('baseline', 'lora', 'guided', 'random'):
        lines.append(f"| {variant} | {fmt(adapters[variant]['test']['accuracy'])} | {fmt(adapters[variant]['test']['macro_f1'])} |")
    lines += ['', 'SDs are descriptive and are not confidence intervals. Sampling-fold and seed SDs capture different variation. '
              'The default deployed head is the historical 94/108-test-correct model; these later controlled heads '
              'and adapters do not replace it. Previously observed data limit independent generalization claims.', '',
              'Verification counts:', '', '```json', json.dumps(result, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional generated Markdown summary; relative to repository root')
    parser.add_argument('--require-private', action='store_true', help='Also require and hash all 26 authenticated local weight artifacts')
    args = parser.parse_args()
    result, sampling, ablation, adapters = verify(args.require_private)
    if args.output:
        target = args.output if args.output.is_absolute() else ROOT / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(summary_document(result, sampling, ablation, adapters), encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
