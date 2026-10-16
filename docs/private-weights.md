# Private model storage

Learned artifacts are stored in the **private** Hugging Face model repository
`meural-operator/xai-weights`, under `kinetic-lens/`. The repository also retains
the earlier five private ViT checkpoints at their existing paths. GitHub holds
code, labels, research reports, numerical results and artifact identities.

[models/private_artifacts.json](../models/private_artifacts.json) pins an exact
Hugging Face commit and all 26 artifact paths, byte counts and SHA256 hashes:

| Group | Contents | Files |
|---|---|---:|
| `default` | Historical KARD head, concept probes and selected channel groups | 3 |
| `cuda` | Complete KARD MoViNet FP32/FP16 ONNX graphs | 2 |
| `research` | Twelve architecture-ablation and nine projection-adapter checkpoints | 21 |

The experimental checkpoints are not self-contained backbone/video models and do
not replace the deployed checkpoint. Their research selection records still
identify the same exact learned bytes.

## Authorized local setup

Obtain access from the model-repository owner. Authenticate through the local
Hugging Face CLI or a locally configured `HF_TOKEN`; do not put a token in a
config, command committed to Git, issue, screenshot or public CI log.

```sh
python -m pip install -e ".[movinet]"
hf auth login
python scripts/setup_models.py
```

The default retrieves the three private head/XAI files and the separately
available hash-pinned official Google backbone. For the complete CUDA graphs:

```sh
python -m pip install -e ".[cuda]"
python scripts/setup_models.py --group cuda --skip-backbone
```

For all learned artifacts, including research checkpoints:

```sh
python scripts/setup_models.py --all --skip-backbone
python scripts/setup_models.py --all --skip-backbone --verify-only
python experiments/report_results.py --require-private
```

`--verify-only` is offline. Existing mismatched artifacts are rejected instead
of silently overwritten. Downloads are pinned to the manifest's exact commit
and verified before atomic placement. The downloader rejects a public model
repository. Standard public CI runs aggregate/metadata checks without any
credential or weight download.

`--from-workspace /path/to/original/workspace` is an offline option for owners
with verified local backups. It does not grant other users access. The official
pretrained backbone remains publicly obtainable from its original provider;
this policy controls access to the trained/combined/model-specific artifacts.
Private storage does not revoke copies someone already acquired.

## Publication and history status

The weight migration uploaded 26 files, downloaded and hash-verified each one,
confirmed repository privacy and rejected anonymous access. Local originals and
an independent backup were preserved. The original five Hugging Face files
were retained. [Migration receipt](private_weight_migration.json) records the
non-secret verification facts.

**Current GitHub source excludes learned files. Earlier public commits contained
them.** A normal deletion commit does not erase those historical blobs.
Rewriting Git history and force-updating published references is a separate
cleanup step that requires owner approval because it changes existing commit
identities and affects clones. Previously downloaded copies cannot be revoked.
Until that cleanup is approved and performed, this repository must not be
described as historically free of public weight files.

The private model repository's owner must keep its visibility private. Never
publish a public Space route that serves these files or a write-capable token.
