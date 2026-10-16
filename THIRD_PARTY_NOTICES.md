# Attribution and artifact terms

- The official MoViNet backbone is provided by Google/TensorFlow Model Garden, pretrained on Kinetics-600: https://github.com/tensorflow/models/tree/master/official/projects/movinet. The exact handle and hashes are in `models/kard/provenance.json`. The backbone is downloaded separately; consult its provider's current artifact terms. Model Garden source is Apache-2.0.
- KARD videos are not included. Obtain the dataset under its own terms. Model evaluation and research reports describe a researcher-defined partition and do not imply an official benchmark.
- The KARD head is the selected original trained classifier exported without parameter changes. The export's original artifact hash and normalization are recorded. Explanation probes/groups are model-specific auxiliary research assets, not universal concepts/circuits.
- The combined inference graphs in `models/kard/onnx/` include parameters derived from that official Google MoViNet backbone and our selected KARD head. Their source/artifact hashes and FP16 conversion details are recorded in the accompanying manifest and CUDA report. These graphs do not change the original backbone provider's artifact terms; consult the official handle and Model Garden attribution above.
- MoViNet/KARD runtime and explanation implementation portions derive from the local DIAT HAR research work. The original MIT notice is retained in `LICENSE`. No MindGraph dashboard source or federated/privacy claims are included.
- NumPy, TensorFlow, OpenCV, Pillow, FastAPI, Uvicorn, PyTorch and ONNX Runtime retain their respective licenses. They are installed as dependencies, not vendored into this repository.

Method references and interpretation limits are preserved in `docs/research/` and linked extension guides.

The research publication additionally retains selected compact experimental
heads/adapters derived from the same official MoViNet features. These are
identified by selection hashes under their research reports; they do not replace
the default model or change the backbone/dataset terms. Large original ViT
checkpoints and converted ViT model binaries are not redistributed in this
update; only benchmark records, conversion manifests and original source
snapshots are included. The historical source archive can reference an external
showcase checkout and is not a bundled training-data or dependency distribution.
