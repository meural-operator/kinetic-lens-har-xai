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

**The owner-approved history cleanup is complete for the published `main`
history.** All seven existing commits were rewritten to remove the 26 learned
artifact paths. A fresh clone from GitHub verified that none of the former
weight-blob versions is reachable. The current code/results tree was identical
before and after the rewrite. A private local Git bundle and the learned-file
backups were preserved.

Existing clones made before the cleanup should be replaced with a fresh clone
or carefully aligned with the cleaned history; merging an old branch can
reintroduce removed history. The research machine's main checkout was aligned
without changing its working files or private weights.

**GitHub retained-object/cache removal is not complete.** A header-only check
of one old raw-file URL still returned HTTP 200 after the rewrite; no weight
payload was downloaded for that check. Exact historical URLs are kept in the
owner's local cleanup records rather than republished here. GitHub can retain
access through unreferenced commit IDs/cached views, and ordinary Git pushes
cannot purge those server-side copies. A Support request draft has been prepared
locally for the owner; it has not been submitted. GitHub reviews eligibility and
does not promise removal of non-sensitive data. See its
[history and cached-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
Previously downloaded copies cannot be revoked. Do not describe this cleanup as
proof that every historical public copy has disappeared.

The private model repository's owner must keep its visibility private. Never
publish a public Space route that serves these files or a write-capable token.
