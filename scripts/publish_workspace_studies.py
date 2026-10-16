"""Import explicitly selected completed research evidence from a local workspace.

No training/inference is executed. Videos, caches, raw frames, installed packages,
launch logs and unrelated working files are never enumerated for publication.
Text paths are normalized; original and published hashes are recorded separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.ps1', '.md', '.json', '.csv', '.tex', '.svg'}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    mov = workspace / 'movinet_evaluation'
    published = []

    def clean(text):
        for prefix in (str(workspace), workspace.as_posix()):
            text = text.replace(prefix, '<local-workspace>')
        # JSON encodes backslashes twice.
        text = text.replace(str(workspace).replace('\\', '\\\\'), '<local-workspace>')
        text = re.sub(r'C:[/\\]+Users[/\\]+Xerox[/\\]+miniconda ?3[/\\]+envs[/\\]+xai[/\\]+python\.exe',
                      'python', text, flags=re.IGNORECASE)
        # Older benchmark manifests used a separate showcase checkout.
        text = re.sub(r'C:[/\\]+Users[/\\]+Xerox[/\\]+Desktop[/\\]+MeitY_Research_Showcase',
                      '<original-showcase>', text, flags=re.IGNORECASE)
        return text

    def copy(source, destination, *, unchanged=False):
        source, destination = Path(source), REPO / destination
        original = source.read_bytes()
        if source.suffix.lower() in TEXT and not unchanged:
            # Canonical LF matches repository attributes on Windows and Linux.
            # Exact archival source snapshots use unchanged=True instead.
            text = clean(original.decode('utf-8-sig')).replace('\r\n', '\n').replace('\r', '\n')
            if source.suffix.lower() == '.svg':
                # Matplotlib emits trailing spaces in multiline path data.
                # Newline separators preserve SVG geometry after trimming.
                text = '\n'.join(line.rstrip(' \t') for line in text.split('\n'))
            if source == mov / 'REPORT.md':
                text = text.replace('](CROP_REPORT.md)', '](../../research/person-crop/REPORT.md)')
            if source == workspace / 'mindgraph_audit/edge_exports/README.md':
                text = text.replace('](../latency_benchmark_fp16/REPORT.md)',
                                    '](latency_benchmark_fp16/REPORT.md)')
            content = text.encode('utf-8')
        else:
            content = original
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        published.append({'source': source.relative_to(workspace).as_posix(),
                          'published': destination.relative_to(REPO).as_posix(),
                          'source_sha256': digest(original), 'published_sha256': digest(content),
                          'source_bytes': len(original), 'published_bytes': len(content),
                          'text_normalized': original != content})

    def files(source, target, names):
        for name in names:
            copy(source / name, Path(target) / name)

    def figures(source, target, stems):
        for stem in stems:
            for ext in ('.pdf', '.svg', '.png'):
                path = source / (stem + ext)
                if path.exists():
                    copy(path, Path(target) / path.name)

    data = mov / 'kard_a0_head_100pct'
    target = 'docs/research/data-efficiency'
    files(data, target, ['REPORT.md', 'comparison.json', 'class_comparison.csv', 'protocol.json',
                          'selection.json', 'split_manifest.csv', 'fraction_comparison.png',
                          'confusion_matrix.png'])
    for fraction in (25, 50):
        folder = mov / f'kard_a0_head_{fraction}pct'
        for name in ('protocol.json', 'selection.json'):
            copy(folder / name, Path(target) / f'{fraction}pct_{name}')
    paper = mov / 'paper_assets/kard_a0_frozen_head'
    files(paper, Path(target) / 'paper_assets',
          ['captions.md', 'results.csv', 'per_action_results.csv', 'table_results.tex',
           'table_actions.tex', 'include_figures.tex', 'plot_data.json'])
    figures(paper, Path(target) / 'paper_assets',
            ['fig01_training_fraction', 'fig02_protocol', 'fig03_action_recall',
             'fig04_confusion_100pct', 'fig05_training_curves', 'fig06_subject_accuracy'])

    sampling = mov / 'kard_sampling_cv'
    target = 'docs/research/sampling'
    files(sampling, target, ['REPORT.md', 'summary.json', 'summary_results.csv',
         'run_results.csv', 'fold_results.csv', 'subject_results.csv', 'seed_results.csv',
         'action_results.csv', 'paired_changes.csv', 'protocol.json', 'selection_lock.json',
         'execution.json'])
    files(sampling / 'paper_assets', Path(target) / 'paper_assets',
          ['include_figures.tex', 'table_results.tex', 'plot_data.json'])
    figures(sampling / 'paper_assets', Path(target) / 'paper_assets',
            ['cv01_fold_comparison', 'cv02_subject_comparison', 'cv03_subject_protocol'])
    for name in ('REPORT.md', 'comparison.json', 'sampling_selection.json', 'protocol.json'):
        copy(mov / 'kard_a0_head_100pct_10fps' / name, Path(target) / 'single_split' / name)

    ablation = mov / 'kard_a0_controlled_ablation_v1'
    target = 'docs/research/architecture-ablations'
    files(ablation, target, ['REPORT.md', 'summary.json', 'summary_metrics.csv', 'per_seed_metrics.csv',
         'paired_C_minus_B.csv', 'per_class_per_seed.csv', 'per_class_summary.csv', 'protocol.json',
         'preflight.json', 'selection_manifest.json', 'evaluation.json', 'historical_context.json',
         'normalization_verification.json', 'figures_for_paper.tex'])
    for variant in 'ABCD':
        for split in ('test', 'validation'):
            filename = f'{variant}_{split}_mean_confusion.csv'
            copy(ablation / filename, Path(target) / filename)
        for seed in (42, 123, 2026):
            filename = f'A_seed_{seed}.pt' if variant == 'A' else f'{variant}_seed_{seed}.pt'
            copy(ablation / 'selected' / filename, Path(target) / 'selected' / filename, unchanged=True)
    figures(ablation / 'figures', Path(target) / 'figures',
            ['macro_f1_comparison', 'paired_temporal_delta', 'test_class_recall',
             'test_mean_confusions', 'validation_class_recall'])
    files(mov / 'kard_a0_compact_temporal_v1', Path(target) / 'single_seed_pilot',
          ['REPORT.md', 'comparison.csv', 'protocol.json', 'selection.json', 'evaluation.json'])

    adapters = mov / 'kard_a0_projection_adapter_v1'
    target = 'docs/research/projection-adapters'
    files(adapters, target, ['REPORT.md', 'summary.json', 'paired_differences.csv', 'paired_differences.json',
          'per_seed_metrics.csv', 'protocol.json', 'preflight.json', 'selection_manifest.json', 'evaluation.json'])
    for variant in ('lora', 'guided', 'random'):
        for seed in (42, 123, 2026):
            filename = f'{variant}_seed_{seed}.pt'
            copy(adapters / 'selected' / filename, Path(target) / 'selected' / filename, unchanged=True)
    figures(adapters / 'figures', Path(target) / 'figures', ['adapter_comparison'])
    # Train/validation-only representation intervention, distinct from retrained
    # compact-head performance. The feature NPZ is intentionally not published.
    files(mov / 'architecture_challenge_20261010', Path(target) / 'diagnosis',
          ['REPORT.md', 'diagnosis.json', 'verification.json'])
    figures(mov / 'architecture_challenge_20261010', Path(target) / 'diagnosis',
            ['architecture_diagnosis'])

    upper = mov / 'kard_a0_upper_finetune'
    target = 'docs/research/upper-finetuning'
    files(upper, target, ['REPORT.md', 'protocol.json', 'selection.json', 'results_comparison.csv',
         'history.json', 'gradient_verification.json', 'updated_backbone_verification.json',
         'comparison.json', 'source_verification.json'])
    files(upper / 'paper_assets', Path(target) / 'paper_assets',
          ['include_figures.tex', 'table_results.tex', 'plot_data.json'])
    figures(upper / 'paper_assets', Path(target) / 'paper_assets',
            ['finetune01_comparison', 'finetune02_training', 'finetune03_validation_action_f1'])

    copy(mov / 'CROP_REPORT.md', 'docs/research/person-crop/REPORT.md')
    copy(mov / 'crop_comparison.csv', 'docs/research/person-crop/crop_comparison.csv')
    copy(mov / 'REPORT.md', 'docs/benchmarks/official-movinet/REPORT.md')
    copy(mov / 'latency_table.csv', 'docs/benchmarks/official-movinet/latency.csv')
    copy(mov / 'downloads.json', 'docs/benchmarks/official-movinet/downloads.json')
    copy(mov / 'timings.csv', 'docs/benchmarks/official-movinet/timings.csv')

    article = workspace / 'deliverables/har_size_latency_article'
    target = 'docs/benchmarks/size-latency'
    files(article, target, ['README.md', 'model_size_latency.csv', 'size_latency_audit.json',
         'article_tables.tex', 'table_vit_size_latency.tex', 'table_movinet_kard_size_latency.tex',
         'table_movinet_scale_size_latency.tex', 'table_temporal_sampling.tex',
         'temporal_sampling.csv', 'temporal_sampling_audit.json', 'temporal_sampling.md'])
    figures(article, target, ['size_latency_tradeoff'])
    # Separate original benchmark sessions, so GPU reruns are never pooled.
    for name in ('latency_benchmark', 'latency_benchmark_fp16'):
        source = workspace / 'mindgraph_audit' / name
        folder = Path('docs/benchmarks/vit-bilstm') / name
        files(source, folder, ['summary.json', 'timings.csv'])
    copy(workspace / 'mindgraph_audit/latency_benchmark_fp16/REPORT.md',
         'docs/benchmarks/vit-bilstm/latency_benchmark_fp16/REPORT.md')
    copy(workspace / 'mindgraph_audit/edge_exports/README.md', 'docs/benchmarks/vit-bilstm/EXPORTS.md')
    for model in ('kth', 'uav', 'kard', 'hon4d', 'ucf45'):
        files(workspace / 'mindgraph_audit/edge_exports' / model,
              Path('docs/benchmarks/vit-bilstm/exports') / model,
              ['manifest.json', 'fp16_manifest.json', 'validation.json'])

    # Exact source archive: not silently refactored into a claimed portable
    # trainer. Its workspace/dependency requirements are documented separately.
    sources = [
        'kard_config.py', 'kard_model.py', 'runtime.py', 'download_models.py',
        'prepare_kard.py', 'prepare_kard_50.py', 'prepare_kard_100.py', 'prepare_kard_10fps.py',
        'extract_kard_features.py', 'train_kard_head.py', 'evaluate_kard_head.py',
        'run_kard_sampling_cv.py', 'write_kard_sampling_cv_report.py',
        'kard_ablation_model.py', 'run_kard_ablation.py', 'write_kard_ablation_report.py',
        'kard_compact_model.py', 'run_kard_compact_pipeline.py', 'predict_kard_compact.py',
        'kard_projection_adapter.py', 'run_kard_projection_adapters.py',
        'prepare_kard_adapter_cache.py', 'write_kard_adapter_report.py',
        'kard_evidence_pool.py', 'verify_projection_adapter.py',
        'diagnose_architecture_challenge.py', 'plot_architecture_diagnosis.py',
        'kard_finetune.py', 'prepare_kard_finetune.py', 'cache_kard_upper_inputs.py',
        'check_kard_upper_gradients.py', 'train_kard_upper.py', 'evaluate_kard_upper.py',
        'verify_kard_updated_backbone.py', 'write_kard_upper_report.py',
        'launch_kard_ablation_background.ps1', 'launch_kard_adapters_background.ps1',
    ]
    for name in sources:
        copy(mov / name, Path('experiments/archive/movinet_evaluation') / name, unchanged=True)
    for name in ('build_size_latency_tables.py', 'add_temporal_sampling_table.py'):
        copy(article / name, Path('experiments/archive/article') / name, unchanged=True)
    for name in ('benchmark_har_devices.py', 'benchmark_har_fp16.py', 'export_har_onnx.py',
                 'format_latency_table.py', 'format_fp16_results.py'):
        source = workspace / 'mindgraph_audit' / name
        if source.exists():
            copy(source, Path('experiments/archive/mindgraph_audit') / name, unchanged=True)
    publication = {
        'schema_version': 1,
        'purpose': 'Completed studies and source snapshots; no model retraining or deployment change.',
        'normalization': 'Published text substitutes local workspace/showcase paths and uses LF. Generated SVG trailing spaces are trimmed without geometry changes. Exact source archive bytes are retained.',
        'historical_hashes': 'Hashes inside copied reports/protocols identify original workspace artifacts, not necessarily normalized published text. Entries below preserve both hashes.',
        'excluded': ['datasets/videos/images', 'activation/feature caches', 'installed packages',
                     'launch/status logs', 'large ViT weights', 'training optimizer checkpoints'],
        'records': published}
    manifest = REPO / 'docs/research/publication_manifest.json'
    manifest.write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'published_files': len(published), 'total_bytes': sum(r['published_bytes'] for r in published),
                      'manifest': str(manifest)}, indent=2))


if __name__ == '__main__':
    main()
