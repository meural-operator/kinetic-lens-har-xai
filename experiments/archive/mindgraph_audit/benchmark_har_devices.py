"""Paired steady-state CPU/GPU latency benchmark for all five HAR models."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['KMP_BLOCKTIME'] = '0'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'optional_deps'))
sys.path.insert(0, str(HERE / 'gpu_deps'))
import cv2
import numpy as np
import torch
import onnxruntime as ort


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def get_samples(root):
    result = []
    for name in ['kth-boxing', 'kth-running', 'kard-drink', 'kard-kick']:
        capture = cv2.VideoCapture(str(root / 'static/media' / name / 'original.mp4'))
        fps = capture.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps/4))
        index = 0
        frames, indices = [], []
        count = 0
        while count < 3:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(frame)
                indices.append(index)
                if len(frames) == 8:
                    result.append((name, frames, indices))
                    frames, indices = [], []
                    count += 1
            index += 1
        capture.release()
    assert len(result) == 9
    return result


def session_options(threads, profile=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    return options


def stats(values):
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'std_ms': statistics.stdev(values), 'median_ms': statistics.median(values),
            'p95_ms': float(np.percentile(values, 95)), 'min_ms': min(values), 'max_ms': max(values)}


def measure_model(model_id, args, api, registry, samples, cpu_inputs, gpu_inputs, output_dir):
    folder = HERE / 'edge_exports' / model_id
    spec = registry.MODELS[model_id]
    model_cpu = api.load_model(spec, torch.device('cpu'))
    model_cpu.vit.config.output_attentions = False
    model_gpu = api.load_model(spec, torch.device('cuda'))
    model_gpu.vit.config.output_attentions = False
    fp32_cpu = ort.InferenceSession(str(folder / f'{model_id}-fp32.onnx'), session_options(args.threads),
                                    providers=['CPUExecutionProvider'])
    int8_cpu = ort.InferenceSession(str(folder / f'{model_id}-int8.onnx'), session_options(args.threads),
                                    providers=['CPUExecutionProvider'])
    cuda_options = {'device_id': '0', 'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC',
                    'user_compute_stream': str(torch.cuda.current_stream().cuda_stream)}
    gpu_session = ort.InferenceSession(str(folder / f'{model_id}-fp32.onnx'), session_options(args.threads),
                                      providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'])
    gpu_session.disable_fallback()
    assert 'CUDAExecutionProvider' in gpu_session.get_providers()
    outputs = [torch.empty((1, spec.num_classes), device='cuda', dtype=torch.float32) for _ in gpu_inputs]
    bindings = []
    for tensor, output in zip(gpu_inputs, outputs):
        binding = gpu_session.io_binding()
        binding.bind_input('frames', 'cuda', 0, np.float32, tuple(tensor.shape), tensor.data_ptr())
        binding.bind_output('logits', 'cuda', 0, np.float32, tuple(output.shape), output.data_ptr())
        bindings.append(binding)
    arrays = [tensor.numpy() for tensor in cpu_inputs]
    names = ['pytorch_cpu_fp32', 'onnx_cpu_fp32', 'onnx_cpu_int8', 'pytorch_gpu_fp32', 'onnx_gpu_fp32']

    def forward(name, index):
        if name == 'pytorch_cpu_fp32':
            return model_cpu(cpu_inputs[index])
        if name == 'onnx_cpu_fp32':
            return fp32_cpu.run(None, {'frames': arrays[index]})[0]
        if name == 'onnx_cpu_int8':
            return int8_cpu.run(None, {'frames': arrays[index]})[0]
        if name == 'pytorch_gpu_fp32':
            return model_gpu(gpu_inputs[index])
        gpu_session.run_with_iobinding(bindings[index])
        return outputs[index]

    comparisons = []
    with torch.inference_mode():
        for name in names:
            for index in range(args.warmups):
                forward(name, index % len(samples))
        torch.cuda.synchronize()
        for index, (video, _, indices) in enumerate(samples):
            reference = forward('pytorch_cpu_fp32', index).numpy().copy()
            entry = {'video': video, 'frame_indices': indices, 'backends': {}}
            for name in names:
                actual = forward(name, index)
                if isinstance(actual, torch.Tensor):
                    actual = actual.cpu().numpy().copy()
                assert np.isfinite(actual).all() and actual.shape == reference.shape
                error = float(np.abs(actual-reference).max())
                entry['backends'][name] = {'max_abs_logit_error': error,
                                          'top1_agreement': bool(actual.argmax() == reference.argmax())}
                if name != 'onnx_cpu_int8':
                    assert error < 0.002, (model_id, name, error)
            comparisons.append(entry)
        torch.cuda.synchronize()
        timings = {name: [] for name in names}
        observations = []
        rng = random.Random(args.seed)
        for device, group in [('CPU', names[:3]), ('GPU', names[3:])]:
            if device == 'GPU':
                # A fresh warm-up immediately precedes the uninterrupted GPU
                # block; no long CPU inference calls are inserted into it.
                for _ in range(args.gpu_warmups):
                    for name in group:
                        forward(name, 0)
                torch.cuda.synchronize()
            for repeat in range(args.repeats):
                indices = list(range(len(samples)))
                rng.shuffle(indices)
                for index in indices:
                    order = list(group)
                    rng.shuffle(order)
                    for name in order:
                        is_gpu = '_gpu_' in name
                        if is_gpu:
                            torch.cuda.synchronize()
                        started = time.perf_counter_ns()
                        result = forward(name, index)
                        if is_gpu:
                            torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter_ns()-started)/1e6
                        timings[name].append(elapsed_ms)
                        observations.append({'backend': name, 'repeat': repeat, 'sample_index': index,
                                             'video': samples[index][0], 'latency_ms': elapsed_ms})
                        del result
                print(json.dumps({'model':model_id, 'device':device, 'timed_repeat_completed':repeat+1,
                                  'of':args.repeats}), flush=True)
    record = {'model_id': model_id, 'dataset': spec.dataset, 'stats': {name:stats(values) for name,values in timings.items()},
              'parity_checks': comparisons, 'timing_observations': observations}
    record['speedups'] = {
        'cpu_fp32': record['stats']['pytorch_cpu_fp32']['mean_ms']/record['stats']['onnx_cpu_fp32']['mean_ms'],
        'cpu_int8': record['stats']['pytorch_cpu_fp32']['mean_ms']/record['stats']['onnx_cpu_int8']['mean_ms'],
        'gpu_fp32': record['stats']['pytorch_gpu_fp32']['mean_ms']/record['stats']['onnx_gpu_fp32']['mean_ms']}
    dump(output_dir / f'{model_id}.json', record)
    print(json.dumps({'model': model_id, 'stats':record['stats'], 'speedups':record['speedups']}), flush=True)

    # Verify placement independently after all timed calls. Profiling is never
    # enabled in benchmark sessions, so it does not inflate measured latency.
    del model_cpu, model_gpu, fp32_cpu, int8_cpu, bindings, outputs, gpu_session
    gc.collect()
    torch.cuda.empty_cache()
    profile_session = ort.InferenceSession(str(folder/f'{model_id}-fp32.onnx'),
        session_options(args.threads, output_dir/f'{model_id}_placement'),
        providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'])
    profile_session.disable_fallback()
    profile_session.run(None, {'frames': arrays[0]})
    torch.cuda.synchronize()
    profile_path = Path(profile_session.end_profiling())
    events = json.loads(profile_path.read_text())
    nodes = [event for event in events if event.get('cat')=='Node' and event.get('args',{}).get('provider')]
    assignment = Counter(event['args']['provider'] for event in nodes)
    assert assignment.get('CUDAExecutionProvider',0)>0
    record['gpu_placement'] = {'executed_node_counts': dict(assignment), 'profile_path':str(profile_path),
                               'cpu_operations':dict(Counter(event['args'].get('op_name') for event in nodes
                                                           if event['args']['provider']=='CPUExecutionProvider'))}
    dump(output_dir / f'{model_id}.json', record)
    del profile_session
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--showcase', type=Path, default=Path(r'C:\Users\Xerox\Desktop\MeitY_Research_Showcase'))
    parser.add_argument('--models', nargs='+', default=['kth','uav','kard','hon4d','ucf45'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--gpu-warmups', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    assert args.threads>0 and args.repeats>1 and args.warmups>0
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    ort.preload_dlls(directory=str(Path(torch.__file__).parent/'lib'))
    assert 'CUDAExecutionProvider' in ort.get_available_providers()
    sys.path.insert(0, str(args.showcase))
    from backend import model as api, registry
    samples = get_samples(args.showcase)
    cpu_inputs = [api.preprocess_bgr(frames).unsqueeze(0) for _,frames,_ in samples]
    gpu_inputs = [tensor.to('cuda') for tensor in cpu_inputs]
    hardware_path = HERE/'benchmark_hardware.json'
    hardware = json.loads(hardware_path.read_text(encoding='utf-8-sig')) if hardware_path.exists() else {}
    output_dir = HERE/'latency_benchmark'
    output_dir.mkdir(exist_ok=True)
    summary = {'hardware':hardware, 'gpu':torch.cuda.get_device_name(),
               'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
               'platform':platform.platform(), 'python':platform.python_version(),
               'torch':torch.__version__, 'torch_cuda':torch.version.cuda,
               'cudnn':torch.backends.cudnn.version(), 'onnxruntime':ort.__version__,
               'protocol':{'batch_size':1,'sequence_length':8,'input_shape':[1,8,3,224,224],
                           'intra_op_threads':args.threads,'inter_op_threads':1,'warmup_calls_per_backend':args.warmups,
                           'additional_warmup_calls_per_gpu_backend_before_gpu_block':args.gpu_warmups,
                           'distinct_windows':len(samples),'repeats_per_window':args.repeats,
                           'timed_calls_per_backend':len(samples)*args.repeats,'seed':args.seed,
                           'order':'separate CPU and GPU blocks; randomized input/backend order within each device block',
                           'gpu_timing':'synchronized wall time; inputs and outputs GPU resident through I/O binding',
                           'exclusions':['model load','session creation','video decode','preprocessing','host-device copies'],
                           'precision':'FP32 with TF32 disabled; dynamic INT8 CPU is mixed precision',
                           'sample_sources':[{'video':name,'frame_indices':indices} for name,_,indices in samples],
                           'limitation':'single workstation session, not isolated from normal OS background activity'},
               'models':[]}
    for model_id in args.models:
        print('BENCHMARK_START '+model_id,flush=True)
        record = measure_model(model_id,args,api,registry,samples,cpu_inputs,gpu_inputs,output_dir)
        summary['models'].append(record)
        dump(output_dir/'summary.json',summary)


if __name__=='__main__':
    main()
