# Verification and reproducibility

Core tests exercise independent numerical oracles, plugin compatibility, class-margin reconstruction, video sampling, bounded live buffers, stale context reset, malformed-input rejection, host/origin checks, REST predictions and camera WebSocket predictions. Optional-runtime tests use small exported fixtures; they are not HAR accuracy benchmarks.

`scripts/verify_workspace_model.py --workspace /path/to/original/workspace` compares the refactored KARD adapter against the original Keras head on cached features for **all 108 validation clips**. It also executes the five explanation techniques on the same Two hand wave clip and repeats inference afterward. Full numerical results are recorded in [verification_results.json](verification_results.json). The script requires the original dataset/caches locally; these are deliberately not distributed.

`scripts/export_from_workspace.py` records the original `.keras` hash and exports its exact normalization, dense weights and bias into a compact NumPy artifact. It does not retrain, fine-tune or quantize. The default adapter uses TensorFlow operations for preprocessing/backbone/head arithmetic. Small floating-point residuals are expected; replay/equivalence tolerance is `1e-4` in logit units. The portable head and source head have different file hashes because their serialization formats differ.

The official backbone setup checks the published archive and variable-data hashes. Defaults never silently substitute a new backbone if the download changes. `models/kard/provenance.json` identifies the original selected weights and reported train/validation/test split.

The local server/browser and actual-model checks are documented in the repository release verification record. WebSocket tests send synthetic frames through the actual buffering and inference protocol. These check application behavior, not continuous-action accuracy or physical camera hardware. Webcam accuracy remains unvalidated, even where capture and inference work.

Chrome camera capture, predictions and a temporal explanation were observed on the local trained-model dashboard. The optional automatic-XAI controls and animated matched-window heatmap were added afterward. The user elected to perform their camera/runtime validation; those additions have not been tested with physical camera input. Their JavaScript lifecycle, scheduler and matching-frame overlay regression checks passed during the CUDA publication update.

The original extended XAI reports are preserved in `docs/research/`. These studies reused validation data after checkpoint selection and remain exploratory. They support bounded intervention effects; high proxy AUC or a plausible heatmap alone does not establish independently meaningful semantic concepts or complete circuits.

## CUDA publication update

The original selected SavedModel was exported to fixed 20-frame ONNX FP32 and FP16 graphs. The paired original-wrapper trial covers one validation window per KARD class, with numerical agreement and independent CUDA operation profiling. A separate saved-video desktop check measures preprocessing plus inference. See [CUDA deployment](cuda.md) and the archived [methods/results](benchmarks/cuda/REPORT.md) for timing scopes and limitations. The original TensorFlow default and its gradient hooks remain available.

The published `movinet-onnx` adapter retains the checked artifact hashes, CUDA provider options, and exact TensorFlow CPU pixel transforms. `scripts/verify_cuda_migration.py` checks its migration on the original 18 saved windows. The desktop runner reuses the original bounded camera/display loop and accepts framework configurations. The CUDA paths have no physical camera validation in this update, per user preference; browser CUDA/XAI performance is not claimed from the saved-video checks.

Publication checks: 21 Python tests passed, including the actual ONNX/TorchScript fixture runtimes and two artifact-integrity checks. The JavaScript dashboard regression checks passed, and a wheel build succeeded. The packaged CUDA adapter preserved both original pixel tensors exactly and top-one predictions on all 18 migration windows. The packaged desktop runner completed 21 saved-video measurements with stable reset-state output and a verified bounded mailbox. [Publication record](benchmarks/cuda/publication_verification.json), [adapter migration](benchmarks/cuda/adapter_migration.json), and [desktop smoke check](benchmarks/cuda/desktop_publication_smoke.json) preserve the evidence. The first restricted-process API test stalled; rerunning under host access passed the full suite.

## Research-evidence publication

The later documentation publication adds sampling, data-efficiency, controlled
architecture, projection-adapter and negative fine-tuning results. It does not
retrain, replace the deployed weights, rerun camera capture or establish fresh
hardware timings. [The study index](research/README.md) separates those model
identities and protocols.

`python experiments/report_results.py` verifies the imported publication-file
hashes and all 21 selected experimental-head hashes, recomputes fold/seed
aggregates from the per-run CSVs, and checks raw ViT timing means/SDs and model
byte-to-MB conversion. [The generated summary](research/VERIFIED_SUMMARY.md)
records those checks. This is evidence reconciliation, not independent
reproduction of training or new scientific validation.

[The reproduction guide](../experiments/README.md) explicitly separates
self-contained report reproduction from historical training snapshots that
need an external dataset and the original workspace artifacts. Original
source hashes and normalized-publication hashes are recorded separately.
