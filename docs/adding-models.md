# Add or replace a HAR model

## Use an existing adapter

Copy `configs/onnx.example.json` or `configs/torchscript.example.json` to an ignored `configs/local_model.json`. Point to your own weights and a JSON list of labels in exact output order. Set fps, resolution, maximum/fixed frames, layout and normalization **from that model's training/export policy**, rather than copying the illustrative values blindly.

```sh
python -m pip install -e ".[onnx]"
kinetic-lens --config configs/local_model.json predict --video /path/to/known_clip.mp4
kinetic-lens --config configs/local_model.json serve --window-seconds 3.2
```

For a fixed 16-frame model at 5 fps, use a 3.2-second live window. Validate both accuracy and timing on your own data. Changing preprocessing or frame policy can invalidate the original model even if its weights load successfully.

ONNX accepts `NCTHW`, `NTHWC` and `NTCHW`, float32/float16 inputs, explicit output names, and stretch/padded resize. Output must be raw `[1,classes]` logits. Multi-input/stateful graphs need a custom adapter. An ONNX graph that exposes probabilities instead of logits needs an explicitly adjusted wrapper.

TorchScript accepts the same axis layouts, OpenCV bilinear stretch and explicit normalization. It runs `eval()` and `inference_mode()`. Export a wrapper returning one logits tensor. A `.pth` state dictionary alone is insufficient because it does not identify the architecture; load it with your architecture in a custom adapter or export a trusted TorchScript logits wrapper first. Treat model artifacts as trusted executable artifacts, not arbitrary uploads.

## Devices

For ONNX CUDA, install `.[onnx-gpu]` instead of `.[onnx]` and set:

```json
"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

The adapter rejects unavailable requested providers. Some operators can still fall back to CPU; verify placement using the official runtime profiler. Do not label an entire run GPU-only from a provider list. For TorchScript, use a compatible CUDA PyTorch build and `"device": "cuda:0"`; the adapter rejects unavailable CUDA. FP16 support depends on the graph and device. INT8 requires an appropriate quantized graph and supported kernels; selecting CUDA does not quantize a model.

The bundled MoViNet runtime uses TensorFlow 2.21. Native Windows runs it on CPU. Linux/WSL2 can use a compatible TensorFlow GPU environment. The framework itself does not export, quantize or retrain weights.

## A custom adapter

Start with `examples/custom_model.py`. Supply these attributes and methods:

```python
class MyModel:
    name = "My trained HAR model"
    labels = ["action A", "action B"]
    input_spec = InputSpec(fps=5, max_frames=16, resolution=224)
    capabilities = frozenset({"predict"})

    def __init__(self, config):
        # Load trusted weights and preserve the training preprocessing.
        ...

    def predict(self, frames):
        # Sampled RGB uint8 [T,H,W,3] -> finite logits [2].
        # Reset recurrent state before every independent window.
        ...

    def provenance(self):
        # Include artifact hashes, actual device and preprocessing.
        ...
```

Set `"adapter": "my_package.adapter:MyModel"` in local config. This leaves camera handling, prediction rendering, benchmarks and generic temporal XAI unchanged. The extension package must be importable (installed or on your local Python path).

Adding a different MoViNet version is not necessarily a filename change: feature dimensions, resolution and restored internal layers may differ. The supplied adapter deliberately validates A0's layout. Use a logits-only adapter for a new variant first, then add gradient/channel hooks and replay tests as needed. Model-specific internal XAI naturally needs more integration than black-box prediction.

Primary runtime references: [TensorFlow SavedModel loading](https://www.tensorflow.org/api_docs/python/tf/saved_model/load), [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), [PyTorch TorchScript loading](https://pytorch.org/docs/stable/generated/torch.jit.load.html).
