"""Export supplied HAR checkpoints and validate CPU ONNX/INT8 inference.

Run with the xai Python executable. Conversion dependencies are isolated in
the adjacent optional_deps directory. Original checkpoints are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
(HERE / 'conversion_temp').mkdir(exist_ok=True)
tempfile.tempdir = str(HERE / 'conversion_temp')
sys.path.insert(0, str(HERE / 'optional_deps'))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def sample_video(path, sequence_length, limit=3):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot read video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / 4))
    windows, frames, indices = [], [], []
    index = 0
    while len(windows) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(frame)
            indices.append(index)
            if len(frames) == sequence_length:
                windows.append((frames, indices))
                frames, indices = [], []
        index += 1
    cap.release()
    return windows


def cpu_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=-1, keepdims=True)


def run_model(model_id, args, model_api, registry, samples):
    spec = registry.MODELS[model_id]
    folder = args.output / model_id
    folder.mkdir(parents=True, exist_ok=True)
    fp32_path = folder / f'{model_id}-fp32.onnx'
    int8_path = folder / f'{model_id}-int8.onnx'
    report = {'model_id': model_id, 'checkpoint': str(spec.weight_path),
              'checkpoint_bytes': spec.weight_path.stat().st_size,
              'input_shape': [1, args.sequence_length, 3, 224, 224], 'stages': {}}
    report_path = folder / 'validation.json'

    def status(stage, value):
        report['stages'][stage] = value
        write_json(report_path, report)
        print(json.dumps({'model': model_id, 'stage': stage,
                          **{key: val for key, val in value.items() if key != 'results'}}), flush=True)

    model = model_api.load_model(spec, torch.device('cpu'))
    model.vit.config.output_attentions = False
    tensors = [model_api.preprocess_bgr(frames).unsqueeze(0) for _, frames, _ in samples]
    refs = []
    with torch.inference_mode():
        for tensor in tensors:
            refs.append(model(tensor).numpy())
    status('pytorch_reference', {'status': 'PASS', 'windows': len(refs),
                               'parameter_count': sum(p.numel() for p in model.parameters())})

    if not fp32_path.exists() or args.overwrite:
        started = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(model, tensors[0], str(fp32_path), dynamo=False,
                              opset_version=17, input_names=['frames'], output_names=['logits'],
                              do_constant_folding=True)
        status('fp32_export', {'status': 'PASS', 'seconds': round(time.perf_counter()-started, 3)})
    del model
    gc.collect()
    onnx.checker.check_model(str(fp32_path))
    graph = onnx.load(str(fp32_path))
    operators = dict(Counter(node.op_type for node in graph.graph.node))
    del graph
    gc.collect()
    status('fp32_graph', {'status': 'PASS', 'bytes': fp32_path.stat().st_size, 'operators': operators})

    if not int8_path.exists() or args.overwrite:
        started = time.perf_counter()
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                         per_channel=True, op_types_to_quantize=['MatMul', 'LSTM'],
                         extra_options={'MatMulConstBOnly': True})
        status('int8_quantization', {'status': 'PASS', 'seconds': round(time.perf_counter()-started, 3)})
    onnx.checker.check_model(str(int8_path))
    graph = onnx.load(str(int8_path))
    operators = dict(Counter(node.op_type for node in graph.graph.node))
    del graph
    gc.collect()
    status('int8_graph', {'status': 'PASS', 'bytes': int8_path.stat().st_size, 'operators': operators,
                        'reduction_vs_checkpoint_percent': round(100*(1-int8_path.stat().st_size/report['checkpoint_bytes']), 3)})

    for variant, path in [('fp32', fp32_path), ('int8', int8_path)]:
        session = cpu_session(path, args.threads)
        session.run(None, {'frames': tensors[0].numpy()})
        results = []
        durations = []
        for (name, _, indices), tensor, reference in zip(samples, tensors, refs):
            elapsed = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                actual = session.run(None, {'frames': tensor.numpy()})[0]
                elapsed.append(time.perf_counter()-started)
            assert actual.shape == (1, spec.num_classes) and np.isfinite(actual).all()
            probability, ref_probability = softmax(actual), softmax(reference)
            actual_label, ref_label = int(actual.argmax()), int(reference.argmax())
            results.append({'video': name, 'frame_indices': indices,
                            'pytorch_label': spec.labels[ref_label], 'onnx_label': spec.labels[actual_label],
                            'top1_agreement': actual_label == ref_label,
                            'max_abs_logit_error': float(np.abs(actual-reference).max()),
                            'max_abs_probability_error': float(np.abs(probability-ref_probability).max()),
                            'softmax_score': float(probability.max()),
                            'cpu_median_ms': round(statistics.median(elapsed)*1000, 3)})
            durations.extend(elapsed)
        worst = max(item['max_abs_logit_error'] for item in results)
        if variant == 'fp32':
            assert worst < 0.002, f'FP32 export parity error too high: {worst}'
        status(f'{variant}_runtime', {'status': 'PASS', 'provider': 'CPUExecutionProvider',
                                      'threads': args.threads, 'windows': len(results),
                                      'top1_agreement_count': sum(r['top1_agreement'] for r in results),
                                      'max_abs_logit_error': worst,
                                      'max_abs_probability_error': max(r['max_abs_probability_error'] for r in results),
                                      'cpu_median_ms': round(statistics.median(durations)*1000, 3),
                                      'results': results})
        del session
        gc.collect()
    manifest = {'model_id': model_id, 'dataset': spec.dataset, 'architecture': 'ViT-B/16 + additive residual BiLSTM',
                'labels': list(spec.labels), 'input_name': 'frames', 'input_shape': report['input_shape'],
                'input_dtype': 'float32', 'input_layout': 'BTCHW', 'output_name': 'logits',
                'preprocessing': {'color': 'RGB', 'resize': [224, 224], 'range_before_normalize': [0, 1],
                                  'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                'sampling': {'target_fps': 4, 'sequence_length': args.sequence_length,
                             'note': 'Sampled demonstration protocol; not the reported benchmark protocol.'},
                'onnx_files': {'fp32': fp32_path.name, 'int8': int8_path.name},
                'quantization': 'Mixed precision: dynamic INT8 constant MatMul and LSTM weights; other operations remain float.',
                'runtime': 'ONNX Runtime CPU; INT8 LSTM uses com.microsoft.DynamicQuantizeLSTM.',
                'validation_scope': 'Numerical and prediction agreement on supplied demonstrations; not held-out accuracy.'}
    with spec.weight_path.open('rb') as handle:
        manifest['checkpoint_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    write_json(folder / 'manifest.json', manifest)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--showcase', type=Path, default=Path(r'C:\Users\Xerox\Desktop\MeitY_Research_Showcase'))
    parser.add_argument('--models', nargs='+', default=['kth'])
    parser.add_argument('--output', type=Path, default=HERE / 'edge_exports')
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.sequence_length <= 0 or args.threads <= 0 or args.repeats <= 0:
        parser.error('sequence length, threads and repeats must be positive')
    sys.path.insert(0, str(args.showcase))
    from backend import model, registry
    if any(value not in registry.MODELS for value in args.models):
        parser.error(f'Known models: {list(registry.MODELS)}')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    for name in ['kth-boxing', 'kth-running', 'kard-drink', 'kard-kick']:
        path = args.showcase / 'static' / 'media' / name / 'original.mp4'
        for frames, indices in sample_video(path, args.sequence_length):
            samples.append((name, frames, indices))
    assert samples, 'No evaluation samples found'
    summary = {'platform': platform.platform(), 'torch': torch.__version__, 'onnx': onnx.__version__,
               'onnxruntime': ort.__version__, 'models': []}
    summary_path = args.output / 'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['models'] = [item for item in previous.get('models', []) if item['model_id'] not in args.models]
    for model_id in args.models:
        try:
            result = run_model(model_id, args, model, registry, samples)
            summary['models'].append(result)
            write_json(summary_path, summary)
        except Exception as exc:
            summary['models'].append({'model_id': model_id, 'status': 'FAIL', 'error': repr(exc)})
            write_json(summary_path, summary)
            raise


if __name__ == '__main__':
    main()
