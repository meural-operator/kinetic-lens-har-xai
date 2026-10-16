"""Convert all HAR exports to FP16 and compare four official GPU backends.

Run with the xai interpreter. This uses showcase architecture/checkpoints, not
MindGraph inference code. Preserve the older CPU/GPU measurements separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import time

from benchmark_har_devices import (HERE, dump, get_samples, np, ort,
                                   session_options, stats, torch)
import onnx
from onnxconverter_common import float16


def convert(model_id):
    folder = HERE / 'edge_exports' / model_id
    source = folder / f'{model_id}-fp32.onnx'
    target = folder / f'{model_id}-fp16.onnx'
    graph = onnx.load(str(source))
    graph = float16.convert_float_to_float16(
        graph, keep_io_types=False, min_positive_val=0.0, max_finite_val=65504.0)
    # This converter updates tensor types but leaves the original attention
    # Cast(to=FLOAT) attributes untouched. Match each Cast target to its declared
    # output type; otherwise ORT correctly rejects the inconsistent graph.
    tensor_types = {v.name:v.type.tensor_type.elem_type for v in
                    [*graph.graph.value_info,*graph.graph.input,*graph.graph.output]}
    cast_repairs = []
    for node in graph.graph.node:
        if node.op_type=='Cast' and tensor_types.get(node.output[0])==onnx.TensorProto.FLOAT16:
            for attr in node.attribute:
                if attr.name=='to' and attr.i==onnx.TensorProto.FLOAT:
                    attr.i = onnx.TensorProto.FLOAT16
                    cast_repairs.append(node.name)
    onnx.checker.check_model(graph,full_check=True)
    assert graph.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert graph.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    dtype_counts = Counter(onnx.TensorProto.DataType.Name(t.data_type)
                          for t in graph.graph.initializer)
    onnx.save(graph, str(target))
    manifest = json.loads((folder / 'manifest.json').read_text())
    info = {'model_id':model_id, 'source_fp32':str(source), 'export_fp16':str(target),
            'source_checkpoint_sha256':manifest['checkpoint_sha256'],
            'fp16_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'fp32_bytes':source.stat().st_size, 'fp16_bytes':target.stat().st_size,
            'initializer_types':dict(dtype_counts), 'input_dtype':'float16',
            'output_dtype':'float16', 'input_shape':manifest['input_shape'],
            'labels':manifest['labels'], 'preprocessing':manifest['preprocessing'],
            'sampling':manifest['sampling'],
            'conversion':{'tool':'onnxconverter_common.float16', 'keep_io_types':False,
                          'min_positive_val':0.0, 'max_finite_val':65504.0,
                          'cast_target_type_repairs':cast_repairs,
                          'op_block_list':float16.DEFAULT_OP_BLOCK_LIST},
            'scope':'Inference logits only; no retraining, calibration or XAI export.'}
    dump(folder / 'fp16_manifest.json', info)
    print('CONVERTED '+json.dumps(info), flush=True)
    del graph
    gc.collect()
    return info


def run_model(model_id, args, api, registry, samples, gpu_inputs, out_dir):
    export = convert(model_id)
    spec = registry.MODELS[model_id]
    original = api.load_model(spec, torch.device('cuda'))
    original.vit.config.output_attentions = False
    half = api.load_model(spec, torch.device('cuda')).half()
    half.vit.config.output_attentions = False
    inputs_half = [value.half() for value in gpu_inputs]
    models = {'pytorch_gpu_fp32':original, 'pytorch_gpu_fp16':half}
    cuda_options = {'device_id':'0', 'use_tf32':'0', 'cudnn_conv_algo_search':'HEURISTIC',
                    'user_compute_stream':str(torch.cuda.current_stream().cuda_stream)}
    sessions, outputs, bindings = {}, {}, {}
    for precision, tensors, dtype, numpy_type in [
        ('fp32',gpu_inputs,torch.float32,np.float32),
        ('fp16',inputs_half,torch.float16,np.float16)]:
        name = 'onnx_gpu_'+precision
        options = session_options(args.threads)
        # Session creation must fail rather than silently placing nodes on CPU.
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        session = ort.InferenceSession(export['source_fp32'] if precision=='fp32'
                                       else export['export_fp16'], options,
                                       providers=[('CUDAExecutionProvider',cuda_options)])
        session.disable_fallback()
        sessions[name] = session
        outputs[name] = [torch.empty((1,spec.num_classes),device='cuda',dtype=dtype)
                         for _ in tensors]
        bindings[name] = []
        for tensor, output in zip(tensors,outputs[name]):
            binding = session.io_binding()
            binding.bind_input('frames','cuda',0,numpy_type,tuple(tensor.shape),tensor.data_ptr())
            binding.bind_output('logits','cuda',0,numpy_type,tuple(output.shape),output.data_ptr())
            bindings[name].append(binding)
    names = ['pytorch_gpu_fp32','onnx_gpu_fp32','pytorch_gpu_fp16','onnx_gpu_fp16']

    def forward(name,index):
        if name in models:
            return models[name](inputs_half[index] if name.endswith('fp16') else gpu_inputs[index])
        sessions[name].run_with_iobinding(bindings[name][index])
        return outputs[name][index]

    comparisons = []
    timings = {name:[] for name in names}
    observations = []
    with torch.inference_mode():
        for name in names:
            for index in range(args.warmups):
                forward(name,index % len(samples))
        torch.cuda.synchronize()
        for index,(video,_,indices) in enumerate(samples):
            reference = forward('pytorch_gpu_fp32',index).float().cpu().numpy().copy()
            entry = {'video':video, 'frame_indices':indices, 'backends':{}}
            for name in names:
                actual = forward(name,index).float().cpu().numpy().copy()
                assert actual.shape==reference.shape and np.isfinite(actual).all(), (model_id,name)
                error = float(np.abs(actual-reference).max())
                prob_ref = torch.from_numpy(reference).softmax(-1).numpy()
                prob = torch.from_numpy(actual).softmax(-1).numpy()
                entry['backends'][name] = {'max_abs_logit_error':error,
                    'max_abs_probability_error':float(np.abs(prob-prob_ref).max()),
                    'top1_agreement':bool(actual.argmax()==reference.argmax()),
                    'fp32_class':int(reference.argmax()), 'actual_class':int(actual.argmax())}
                if name=='onnx_gpu_fp32':
                    assert error<0.002, (model_id,name,error)
            comparisons.append(entry)
        for _ in range(args.gpu_warmups):
            for name in names:
                forward(name,0)
        torch.cuda.synchronize()
        rng = random.Random(args.seed)
        for repeat in range(args.repeats):
            order_inputs = list(range(len(samples)))
            rng.shuffle(order_inputs)
            for index in order_inputs:
                order = list(names)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    result = forward(name,index)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    timings[name].append(elapsed)
                    observations.append({'backend':name, 'repeat':repeat, 'sample_index':index,
                                         'video':samples[index][0], 'latency_ms':elapsed})
                    del result
            print(json.dumps({'model':model_id,'timed_repeat_completed':repeat+1}),flush=True)
    record = {'model_id':model_id, 'dataset':spec.dataset, 'export':export,
              'stats':{name:stats(values) for name,values in timings.items()},
              'parity_checks':comparisons, 'timing_observations':observations}
    baseline = record['stats']['pytorch_gpu_fp32']['mean_ms']
    record['speedups_vs_pytorch_fp32'] = {name:baseline/value['mean_ms']
                                        for name,value in record['stats'].items()}
    record['onnx_fp16_speedup_vs_onnx_fp32'] = (
        record['stats']['onnx_gpu_fp32']['mean_ms']/record['stats']['onnx_gpu_fp16']['mean_ms'])
    record['agreement'] = {name:{'top1_matches':sum(v['backends'][name]['top1_agreement']
                                                     for v in comparisons),
                               'total':len(comparisons),
                               'max_abs_logit_error':max(v['backends'][name]['max_abs_logit_error']
                                                        for v in comparisons),
                               'max_abs_probability_error':max(v['backends'][name]['max_abs_probability_error']
                                                              for v in comparisons)} for name in names}
    dump(out_dir/f'{model_id}.json',record)
    # No profiling in timed sessions; independently verify node placement after.
    models.clear()
    del original,half,sessions,outputs,bindings,session,binding,tensor,output
    gc.collect()
    torch.cuda.empty_cache()
    placement = {}
    for precision,tensors in [('fp32',gpu_inputs),('fp16',inputs_half)]:
        options = session_options(args.threads,out_dir/f'{model_id}_{precision}_placement')
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        profile_session = ort.InferenceSession(export['source_fp32'] if precision=='fp32'
                                              else export['export_fp16'],options,
                                              providers=[('CUDAExecutionProvider',cuda_options)])
        profile_session.run(None,{'frames':tensors[0].cpu().numpy()})
        torch.cuda.synchronize()
        profile_path = Path(profile_session.end_profiling())
        nodes = [e for e in json.loads(profile_path.read_text())
                 if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
        counts = Counter(e['args']['provider'] for e in nodes)
        operations = Counter(e['args'].get('op_name') for e in nodes)
        assert counts.get('CUDAExecutionProvider',0)>0 and counts.get('CPUExecutionProvider',0)==0
        placement[precision] = {'executed_node_counts':dict(counts),
                                'operations':dict(operations),'profile_path':str(profile_path)}
        del profile_session
        gc.collect()
        torch.cuda.empty_cache()
    record['gpu_placement'] = placement
    dump(out_dir/f'{model_id}.json',record)
    print('RESULT '+json.dumps({'model':model_id,'stats':record['stats'],
                               'agreement':record['agreement'],'placement':placement}),flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--showcase',type=Path,default=Path(r'C:\Users\Xerox\Desktop\MeitY_Research_Showcase'))
    parser.add_argument('--models',nargs='+',default=['kth','uav','kard','hon4d','ucf45'])
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--warmups',type=int,default=5)
    parser.add_argument('--gpu-warmups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261009)
    parser.add_argument('--resume',action='store_true',help='Reuse completed records with placement verification.')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    ort.preload_dlls(directory=str(Path(torch.__file__).parent/'lib'))
    import sys
    sys.path.insert(0,str(args.showcase))
    from backend import model as api, registry
    samples = get_samples(args.showcase)
    gpu_inputs = [api.preprocess_bgr(frames).unsqueeze(0).to('cuda') for _,frames,_ in samples]
    out_dir = HERE/'latency_benchmark_fp16'
    out_dir.mkdir(exist_ok=True)
    previous = json.loads((HERE/'latency_benchmark/summary.json').read_text())
    summary = {key:previous[key] for key in ['hardware','gpu','gpu_memory_bytes','platform',
                 'python','torch','torch_cuda','cudnn','onnxruntime']}
    summary['protocol'] = {**previous['protocol'],
        'order':'GPU-only block; four backends randomized per input and repeat',
        'precision':'Fresh paired FP32 (TF32 disabled) and FP16 GPU measurements; pure FP16 PyTorch weights/input, FP16 ONNX input/output.',
        'warmup_calls_per_backend':args.warmups,
        'additional_warmup_calls_per_gpu_backend_before_gpu_block':args.gpu_warmups,
        'repeats_per_window':args.repeats,'timed_calls_per_backend':len(samples)*args.repeats,
        'seed':args.seed,'gpu_runtime':'Official Microsoft onnxruntime-gpu; CUDA EP; CPU fallback prohibited.',
        'validation':'Numerical/top-1 agreement on supplied demonstrations; not held-out accuracy.'}
    summary['models'] = []
    for model_id in args.models:
        print('START '+model_id,flush=True)
        saved = out_dir/f'{model_id}.json'
        existing = json.loads(saved.read_text()) if args.resume and saved.exists() else {}
        if existing.get('gpu_placement'):
            summary['models'].append(existing)
            print('REUSED_COMPLETE '+model_id,flush=True)
        else:
            summary['models'].append(run_model(model_id,args,api,registry,samples,gpu_inputs,out_dir))
        dump(out_dir/'summary.json',summary)


if __name__=='__main__':
    main()
