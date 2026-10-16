# Reproducing research evidence

The application defaults and experimental research checkpoints have different
roles. The application still uses the historical selected KARD head identified
in [its model card](../models/kard/MODEL_CARD.md). The three-seed controls and
adapter checkpoints are evaluation artifacts, not replacement dashboard weights.

## Recompute the published tables without a dataset or GPU

From the repository root, with Python 3.11 or later:

```sh
python experiments/report_results.py
python experiments/report_results.py --output docs/research/VERIFIED_SUMMARY.md
```

This uses only the standard library. It verifies every imported artifact against
the publication manifest; recomputes sampling fold means/SDs from all 30 run
rows; recomputes ablation and adapter seed statistics; verifies all 21 selected
experimental checkpoints against their selection records; and checks the
recorded raw ViT latency means/SDs and decimal model sizes. It does not claim
to reproduce training, recognition accuracy, hardware timings or camera behavior.

The [study index](../docs/research/README.md) links protocols, per-run results,
checkpoint selections, controls and article figures. The [benchmark index](../docs/benchmarks/README.md)
distinguishes window lengths, state policies and transfer costs.

## Historical training and measurement source snapshots

`archive/movinet_evaluation/` retains the exact scripts used for the original
training, sampling, ablation, adapter and fine-tuning studies.
`archive/mindgraph_audit/` retains the available original ViT measurement/export
scripts. `archive/article/` retains the table-building and sampling-table scripts.
Original source-byte hashes and normalized published-report hashes are recorded
separately in [publication_manifest.json](../docs/research/publication_manifest.json).

These scripts are an inspectable source archive, **not a portable one-command
training CLI**. They rely on the historical workspace layout, locally acquired
KARD data, feature/activation caches, exact dependencies and original artifacts.
Do not run them directly inside the archive and assume that missing files or a
new package version reproduces the historical experiment.

To rerun them, use a separate workspace with:

```text
<workspace>/
  datasets/KARD/<action>/<video>.mp4
  movinet_evaluation/
    <archived .py scripts>
    models/a0_stream/                 exact official pretrained SavedModel
    kard_a0_head_25pct/
    kard_a0_head_50pct/
    kard_a0_head_100pct/
    kard_a0_head_100pct_10fps/
    kard_a0_upper_finetune/
    kard_a0_compact_temporal_v1/
    kard_a0_controlled_ablation_v1/
    kard_a0_projection_adapter_v1/
```

The study `protocol.json` files specify the original environments and hashes.
The TensorFlow runs used native Windows TensorFlow 2.21 CPU; controlled
ablation/adapter optimization used PyTorch 2.13.0+cu130 CUDA. Archived
`runtime.py` first looks for the historical local `tf_deps` folder, then normal
installed packages. Report renderers additionally require Matplotlib and, for
some original paper catalogs, PyMuPDF/pypdf. Data curation uses SciPy where
specified in the source. Installed dependencies are not vendored here.

Order matters: construct/audit the subject split, curate nested training subsets,
extract frozen features with their provenance, train fresh heads with train-only
normalization, and lock validation selections before test reads. Sampling
cross-validation fits normalization and heads separately in each fold. Upper
adaptation requires frozen b4/l1 activation caches; the compact/adapter studies
derive their own immutable endpoints. The archived scripts and reports preserve
the checks, rather than treating any final CSV as evidence that these steps ran.

Typical historical runner stages, **after the prerequisite layout/artifacts
exist**, are:

```sh
python run_kard_sampling_cv.py
python write_kard_sampling_cv_report.py

python run_kard_ablation.py --preflight-only
python run_kard_ablation.py --resume

python prepare_kard_adapter_cache.py
python run_kard_projection_adapters.py --preflight-only
python run_kard_projection_adapters.py --resume
```

Run these from that separate workspace's `movinet_evaluation` folder. Baseline
preparation and cache-generation code is included for inspection; the archive
is not tested as a fresh end-to-end training installation. The exact full
historical workspace and caches are not distributed. This is an explicit
reproducibility limitation.

## Models and data

Selected compact experimental `.pt` checkpoints are bundled under each study's
`selected/` directory with recorded hashes. They require the corresponding
archived model definitions and official backbone; they are not self-contained
video inference models. PyTorch serialization should be loaded with
`weights_only=True` where supported, as the archived inference paths do.
Optimizer/resume checkpoints, videos, raw frames, activation/feature caches,
large ViT exports and installed packages are excluded.

Dataset and pretrained-weight terms continue to apply. See
[third-party notices](../THIRD_PARTY_NOTICES.md). A hash proves identity of an
artifact, not independent correctness of its scientific interpretation.
