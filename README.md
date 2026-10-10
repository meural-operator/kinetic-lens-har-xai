# KineticLens — HAR & XAI Studio

**Recognize actions in video, inspect the decisions, and swap models without rebuilding the dashboard.**

KineticLens connects trained human activity recognition (HAR) models to a local camera dashboard and a small, extensible explanation interface. Its default model is **Google's official MoViNet A0-Stream backbone with our trained 18-class KARD classification head**—the selected model used in the accompanying XAI studies. The original Kinetics-600 classifier is bypassed. No retraining is needed to run it.

## Run the trained KARD model

Python 3.11–3.13. Run commands from the repository root. A new environment is recommended; an existing compatible environment also works.

```sh
python -m pip install -e ".[movinet,dev]"
python scripts/setup_models.py
kinetic-lens --config configs/movinet_kard.json serve
```

Open **http://127.0.0.1:8765**, choose **Start camera**, and allow browser camera access. The first prediction appears after a four-second window is collected. Stop releases the camera and clears the session. The setup script downloads and verifies the exact official backbone; the compact trained head, labels, concept proxies and channel selections are already versioned in `models/kard/`.

Windows PowerShell with your Miniconda environment:

```powershell
conda activate xai
python -m pip install -e ".[movinet,dev]"
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

## What is implemented?

| Component | Function | Scope |
|---|---|---|
| MoViNet + KARD head | Real clip/window predictions; original-model equivalence checks | A0-Stream, 172² padded RGB, 5 fps, max 50 frames; fresh states per window |
| Live dashboard | Browser camera, bounded temporal buffer, top-three scores, latency and result age | Local rolling-window inference; no persistent video recording |
| Temporal attribution | Reversal, shuffle, static-frame and multi-reference segment interventions | Any adapter exposing prediction |
| Spatial attribution | Video Grad-CAM or signed class-pair maps | MoViNet gradient hook; coarse maps need faithfulness controls |
| Contrastive explanation | Exact feature contributions to a class-pair logit margin | Models exposing an affine head; feature coordinates are not semantic concepts |
| Concept attribution | Training-fitted proxy directions and feature deletions | Bundled action-family proxies; semantic validation remains partial |
| Mechanistic explanation | Removal/retention of training-selected channel groups | MoViNet internal hook; selections for Drink, Two hand wave and Forward Kick |
| ONNX adapter | Stateless single-input video logits; explicit CPU/CUDA providers | Configure layout, labels, normalization and fixed frame count |
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

The selected head was trained on **324 KARD clips**, with subject-disjoint validation/test partitions of 108 clips each. Saved results are **96/108 validation** and **94/108 test**. This is a researcher-defined split with two subjects per held-out partition, not an official benchmark or a new blind evaluation. The [XAI interpretations](docs/research/INTERPRETATION.md) distinguish executed effects from semantic and circuit claims.

The KARD model has **no trained idle/unknown class**, and displayed softmax scores are uncalibrated. Clip-level KARD accuracy does not establish accuracy on continuous webcam actions. The live window is four seconds by default, while offline inference can use up to ten seconds. This repository provides HAR/XAI tooling; it does not claim differential privacy, encrypted inference, federated training or certified privacy protection.

The server binds to loopback. Camera frames stay in memory and are sent only to this local server. Explanation JSON is saved only when requested through the CLI or dashboard download. Datasets, recordings, full backbone weights and generated outputs are excluded from Git.

## Testing and devices

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

[Verification details](docs/verification.md) distinguish framework tests, actual-model equivalence, optional-runtime tests and live-camera validation. GitHub Actions runs core tests and builds wheels on Windows/Linux; optional-runtime tests run in a separate job.

Native Windows TensorFlow 2.21 uses CPU. For TensorFlow GPU use a compatible Linux/WSL2 installation. ONNX Runtime CUDA and PyTorch CUDA are configurable when their GPU runtimes/drivers are installed. Install **one** of `.[onnx]` or `.[onnx-gpu]`, not both. GPU availability is not a guarantee that every operation executes on GPU; profile your actual model. See [device and model guide](docs/adding-models.md).

## Attribution

The default backbone is Google's [MoViNet A0-Stream](https://github.com/tensorflow/models/tree/master/official/projects/movinet), pretrained on Kinetics-600. The KARD classifier and XAI studies are documented with artifact hashes in `models/kard/provenance.json`. The backbone and dataset retain their respective terms; the dataset is not redistributed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
