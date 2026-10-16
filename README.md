# KineticLens — HAR & XAI Studio

**Recognize actions in video, inspect the decisions, and swap models without rebuilding the dashboard.**

KineticLens connects trained human activity recognition (HAR) models to a local camera dashboard and a small, extensible explanation interface. Its default model is **Google's official MoViNet A0-Stream backbone with our trained 18-class KARD classification head**—the selected model used in the accompanying XAI studies. The original Kinetics-600 classifier is bypassed. No retraining is needed to run it.

## Run the trained KARD model

Python 3.11–3.13. Run commands from the repository root. A new environment is recommended; an existing compatible environment also works.

```sh
python -m pip install -e ".[movinet,dev]"
hf auth login
python scripts/setup_models.py
kinetic-lens --config configs/movinet_kard.json serve
```

Open **http://127.0.0.1:8765**, choose **Start camera**, and allow browser camera access. The first prediction appears after a four-second window is collected. Stop releases the camera and clears the session. Setup downloads the exact official backbone and retrieves the trained head and model-specific XAI assets from **private Hugging Face storage**, using an authenticated account with access. GitHub contains code, labels, manifests and research results; learned artifacts are downloaded locally and ignored by Git. See [private weights and access](docs/private-weights.md).

In the Explanation Lab, choose a **Technique** and either **Explain latest window** once or enable **Automatically explain new windows**. Automatic mode is optional, off by default, and waits 5, 10 or 30 seconds after each completed explanation. Spatial attribution displays an animated replay of the matching original sampled window with an adjustable heatmap; its age is shown separately from the live camera. **Show heatmap** toggles the overlay. Other techniques display their measurements below. Explanations can pause inference, and automatic mode stops on errors. See [runtime explanations](docs/runtime-xai.md) for what each display means.

Windows PowerShell with your Miniconda environment:

```powershell
conda activate xai
python -m pip install -e ".[movinet,dev]"
hf auth login
python scripts/setup_models.py
kinetic-lens --config configs/movinet_kard.json serve
```

No camera is needed for offline prediction or explanations:

The first call can include several seconds of TensorFlow graph tracing. Later calls reuse the traced graph. The dashboard reports processing time and result age; benchmark warm calls separately from startup. Camera permission is specific to this localhost URL, so another dashboard's permission may not carry over.

```sh
kinetic-lens --config configs/movinet_kard.json predict --video /path/to/clip.mp4 --output outputs/prediction.json
kinetic-lens --config configs/movinet_kard.json explain --video /path/to/clip.mp4 --method temporal --output outputs/temporal.json
kinetic-lens --config configs/movinet_kard.json explain --video /path/to/clip.mp4 --method contrastive --target 8 --rival 13
kinetic-lens --config configs/movinet_kard.json benchmark --video /path/to/clip.mp4 --repeats 10
```

For a dependency-light UI smoke test, install `.[dev]` and use `configs/demo.json`. That configuration is explicitly a **synthetic RGB fixture, not a trained HAR model**. Missing trained weights never silently select the fixture.

## CUDA inference and the original desktop camera UI

The selected MoViNet/KARD model also has checked **ONNX CUDA FP16 and FP32** exports in private Hugging Face storage. Install `.[cuda]` with compatible CUDA/cuDNN libraries, authenticate, and retrieve them into the ignored local `models/kard/onnx/` folder:

```sh
hf auth login
python scripts/setup_models.py --group cuda --skip-backbone
python scripts/desktop/realtime_kard.py --config configs/movinet_kard_cuda_fp16.json
kinetic-lens --config configs/movinet_kard_cuda_fp16.json serve
```

The first command opens the original desktop window with predictions overlaid on the feed; the second uses the browser dashboard. Windows `xai` launchers are in `scripts/desktop/`. Change the config to `movinet_kard_cuda_fp32.json` for FP32. The ONNX adapter preserves the original TensorFlow pixel preprocessing and requires a four-second/20-sample window. It supports prediction and temporal perturbations; retain the TensorFlow config for spatial, concept, contrastive and channel explanations.

On RTX 4000 Ada, recorded model invocation medians were 26.69 ms for FP16 and 31.74 ms for FP32; full app processing medians were 103.17 and 91.90 ms in a separate workload. Both exports preserved 18/18 sampled original predictions and were profiled on CUDA. Camera testing of the GPU paths remains user-led. See [CUDA setup, evidence and reproduction](docs/cuda.md).

## What is implemented?

| Component | Function | Scope |
|---|---|---|
| MoViNet + KARD head | Real clip/window predictions; original-model equivalence checks | A0-Stream, 172² padded RGB, 5 fps, max 50 frames; fresh states per window |
| Live dashboard | Browser camera, bounded temporal buffer, top-three scores, latency and result age; optional automatic XAI and matching spatial heatmap replay | Local rolling-window inference; no persistent video recording |
| Temporal attribution | Reversal, shuffle, static-frame and multi-reference segment interventions | Any adapter exposing prediction |
| Spatial attribution | Video Grad-CAM or signed class-pair maps | MoViNet gradient hook; coarse maps need faithfulness controls |
| Contrastive explanation | Exact feature contributions to a class-pair logit margin | Models exposing an affine head; feature coordinates are not semantic concepts |
| Concept attribution | Training-fitted proxy directions and feature deletions | Bundled action-family proxies; semantic validation remains partial |
| Mechanistic explanation | Removal/retention of training-selected channel groups | MoViNet internal hook; selections for Drink, Two hand wave and Forward Kick |
| ONNX adapter | Stateless single-input video logits; explicit CPU/CUDA providers | Configure layout, labels, normalization and fixed frame count |
| MoViNet CUDA adapter | Checked KARD ONNX FP16/FP32 exports, original pixel pipeline, CPU fallback disabled | Fixed 20 samples at 5 fps; prediction/temporal hooks; no gradients |
| TorchScript adapter | Stateless video logits; explicit CPU/CUDA device | Exported wrapper returning `[1, num_classes]` logits |

Unsupported methods are shown as unavailable. An adapter does not automatically gain gradients, semantic concepts or channel hooks by supplying predictions.

## A small structure with clear boundaries

```text
configs/                  model and input policies
models/kard/              compact trained head, labels, XAI assets, provenance
src/kinetic_lens/
  contracts.py            RGB clip, logits and explanation contracts
  registry.py             built-ins and module:Class plugins
  models/                 runtime-specific adapters
  xai/                    independent explanation methods
  video.py                offline sampling and live window buffer
  server.py               local API and camera sessions
  web/                    browser dashboard (no Node build required)
  cli.py                  prediction, XAI, benchmark and server commands
examples/                 minimal model/XAI plugins
tests/                    numerical, adapter, API and streaming checks
scripts/                  setup, export and model verification
docs/                     architecture, extension guides and research findings
```

The dashboard depends on the model contract, not MoViNet internals. Add a model by selecting an existing adapter or implementing `predict(frames)` and `provenance()` with its input specification, labels and capabilities. Add a technique by implementing `explain(model, frames, ...)` and declaring required capabilities. See [adding models](docs/adding-models.md) and [adding XAI](docs/adding-xai.md).

## Evidence and limits

The deployed head was trained on **324 KARD clips**, with subject-disjoint validation/test partitions of 108 clips each. Saved results are **96/108 validation (88.89%)** and **94/108 test (87.04%)**. The [model card](models/kard/MODEL_CARD.md) and [provenance](models/kard/provenance.json) identify that checkpoint. This is a researcher-defined split with two subjects per held-out partition, not an official benchmark or a new blind evaluation. The [XAI interpretations](docs/research/INTERPRETATION.md) distinguish executed effects from semantic and circuit claims.

### Recorded deployment measurements

The same historical KARD checkpoint has the following warm model-invocation
results on a Xeon w7-2595X / RTX 4000 Ada workstation:

| Runtime | Serialized size (MB) | Mean inference (ms) |
|---|---:|---:|
| TensorFlow CPU FP32 | 26.363 | 121.30 |
| ONNX CUDA FP32 | 8.430 | 34.61 |
| ONNX CUDA FP16 | 4.997 | 28.73 |

Batch 1, 20 RGB samples at 172², fresh states, 90 timed calls per backend.
ONNX timing includes transfers and synchronous CPU output; decoding,
preprocessing, camera/display and XAI are excluded. TensorFlow size is the full
SavedModel package, while ONNX size is a graph file; MB means 1,000,000 bytes,
not runtime memory. The [benchmark index](docs/benchmarks/README.md) separates
model invocation from the slower full app-processing workload and provides the
[complete size/precision tables](docs/benchmarks/size-latency/README.md), including
the five ViT–BiLSTM checkpoints and three official MoViNet scales.

### Completed research comparisons

- **Sampling density:** five subject folds and three seeds per fold gave
  87.78 ± 4.90% test accuracy at 5 fps, versus 85.43 ± 4.41% at 10 fps.
  SD is across fold means after averaging seeds. Frame caps are 50/100 over
  at most ten seconds; this study did not benchmark latency.
- **Architecture ablations:** the fresh three-seed affine control achieved
  91.93 ± 2.03% test macro F1; the compact temporal variant achieved
  86.17 ± 4.35%. The initial promising single-seed pilot is retained alongside
  the subsequent controlled negative result.
- **Projection adapters:** ordinary LoRA, classifier-guided and random-subspace
  adapters achieved mean test macro F1 of 92.52%, 92.19% and 92.46% respectively,
  versus 91.93% for their matched control. Gains were concentrated in one seed;
  classifier guidance did not beat the equal-budget random-subspace control.
- **Data efficiency and fine-tuning:** nested 81/162/324-clip training subsets
  and the unsuccessful upper-backbone fine-tuning run are also reported.
  The latter selected its unchanged epoch-zero fallback.

These later controls/adapters are **different experimental heads, not the
deployed dashboard/XAI checkpoint**. Their seed-averaged scores must not be
paired with the deployed model's latency as one model result. See the
[study index](docs/research/README.md), [article-ready tables](docs/benchmarks/size-latency/article_tables.tex)
and [reproduction guide](experiments/README.md). Protocols, per-run CSVs,
selection records, negative results and PDF/SVG/PNG figures are public. Selected
experimental heads/adapters are private; their pinned identities remain in the
public manifests. Recompute the bundled aggregate tables and verify public
artifact hashes without videos, credentials or a GPU:

```sh
python experiments/report_results.py
```

To verify the private learned files as well, an authorized user can run
`python scripts/setup_models.py --all --skip-backbone` followed by
`python experiments/report_results.py --require-private`.

The KARD model has **no trained idle/unknown class**, and displayed softmax scores are uncalibrated. Clip-level KARD accuracy does not establish accuracy on continuous webcam actions. The live window is four seconds by default, while offline inference can use up to ten seconds. This repository provides HAR/XAI tooling; it does not claim differential privacy, encrypted inference, federated training or certified privacy protection.

The server binds to loopback. Camera frames stay in memory and are sent only to this local server. Explanation JSON is saved only when requested through the CLI or dashboard download. Datasets, recordings, original TensorFlow backbone downloads, learned files and generated outputs are excluded from Git. Checked ONNX/head/adapter identities are pinned in the public private-artifact manifest; the bytes are stored in the private model repository. Earlier public Git history contained weights; see the migration status in [private-weights.md](docs/private-weights.md).

## Testing and devices

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

[Verification details](docs/verification.md) distinguish framework tests, actual-model equivalence, optional-runtime tests and live-camera validation. GitHub Actions runs core tests and builds wheels on Windows/Linux; optional-runtime tests run in a separate job.

Native Windows TensorFlow 2.21 uses CPU. For TensorFlow GPU use a compatible Linux/WSL2 installation. ONNX Runtime CUDA and PyTorch CUDA are configurable when their GPU runtimes/drivers are installed. Install **one** of `.[onnx]` or `.[onnx-gpu]`, not both. GPU availability is not a guarantee that every operation executes on GPU; profile your actual model. See [device and model guide](docs/adding-models.md).

## Attribution

The default backbone is Google's [MoViNet A0-Stream](https://github.com/tensorflow/models/tree/master/official/projects/movinet), pretrained on Kinetics-600. The KARD classifier and XAI studies are documented with artifact hashes in `models/kard/provenance.json`. The backbone and dataset retain their respective terms; the dataset is not redistributed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
