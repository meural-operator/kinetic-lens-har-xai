# Verification and reproducibility

Core tests exercise independent numerical oracles, plugin compatibility, class-margin reconstruction, video sampling, bounded live buffers, stale context reset, malformed-input rejection, host/origin checks, REST predictions and camera WebSocket predictions. Optional-runtime tests use small exported fixtures; they are not HAR accuracy benchmarks.

`scripts/verify_workspace_model.py --workspace /path/to/original/workspace` compares the refactored KARD adapter against the original Keras head on cached features for **all 108 validation clips**. It also executes the five explanation techniques on the same Two hand wave clip and repeats inference afterward. Full numerical results are recorded in [verification_results.json](verification_results.json). The script requires the original dataset/caches locally; these are deliberately not distributed.

`scripts/export_from_workspace.py` records the original `.keras` hash and exports its exact normalization, dense weights and bias into a compact NumPy artifact. It does not retrain, fine-tune or quantize. The default adapter uses TensorFlow operations for preprocessing/backbone/head arithmetic. Small floating-point residuals are expected; replay/equivalence tolerance is `1e-4` in logit units. The portable head and source head have different file hashes because their serialization formats differ.

The official backbone setup checks the published archive and variable-data hashes. Defaults never silently substitute a new backbone if the download changes. `models/kard/provenance.json` identifies the original selected weights and reported train/validation/test split.

The local server/browser and actual-model checks are documented in the repository release verification record. WebSocket tests send synthetic frames through the actual buffering and inference protocol. These check application behavior, not continuous-action accuracy or physical camera hardware. Webcam accuracy remains unvalidated, even where capture and inference work.

Chrome camera capture, predictions and a temporal explanation were observed on the local trained-model dashboard. The optional automatic-XAI controls and animated matched-window heatmap were added afterward. The user elected to perform their camera/runtime validation; those additions have not been runtime-tested. Earlier passing test counts apply to the preceding framework version, not to an execution of the newly extended browser regression checks.

The original extended XAI reports are preserved in `docs/research/`. These studies reused validation data after checkpoint selection and remain exploratory. They support bounded intervention effects; high proxy AUC or a plausible heatmap alone does not establish independently meaningful semantic concepts or complete circuits.
