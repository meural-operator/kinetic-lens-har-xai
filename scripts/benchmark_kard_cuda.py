"""Paired TensorFlow CPU / ONNX CUDA FP32 / FP16 checks on KARD video windows.

No webcam is opened. Numerical agreement concerns these windows, not full-clip
benchmark accuracy or recognition accuracy on continuous camera footage.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class WindowClassifier:
    def __init__(self, experiment):
        import tensorflow as tf
        self.labels=json.loads((experiment/'labels.json').read_text())
        self.loaded=tf.saved_model.load(str(experiment/'saved_model'))
        self.predict=self.loaded.signatures['serving_default']


import cv2
import numpy as np
import torch
import onnxruntime as ort


def stats(values):
    return {'n':len(values),'mean_ms':float(np.mean(values)),
            'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
            'min_ms':min(values),'max_ms':max(values)}


def read_window(path,fps,count):
    capture=cv2.VideoCapture(str(path))
    if not capture.isOpened(): raise ValueError(f'Cannot read {path}')
    source_fps=capture.get(cv2.CAP_PROP_FPS)
    frames=[];indices=[];index=0
    while len(frames)<count:
        ok,frame=capture.read()
        if not ok:break
        wanted=int(np.ceil(len(frames)*source_fps/fps-1e-8))
        if index>=wanted:
            frames.append(frame);indices.append(index)
        index+=1
    capture.release()
    return frames,indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--export-dir',type=Path,default=ROOT/'models/kard/onnx')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--samples-per-class',type=int,default=1)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/cuda_benchmark")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is unavailable')
    print('CUDA_AVAILABLE',torch.cuda.get_device_name(),flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ort.preload_dlls(directory=str(Path(torch.__file__).parent/'lib'))
    manifest=json.loads((args.export_dir/'manifest.json').read_text())
    shape=manifest['input_shape'];count=shape[1]
    reference=WindowClassifier(args.experiment_dir)
    if manifest['source_saved_model_pb_sha256']!=hashlib.sha256((args.experiment_dir/'saved_model/saved_model.pb').read_bytes()).hexdigest():
        raise RuntimeError('Export/source checkpoint mismatch')
    if manifest['source_variables_sha256']!=hashlib.sha256((args.experiment_dir/'saved_model/variables/variables.data-00000-of-00001').read_bytes()).hexdigest():
        raise RuntimeError('Export/source variables mismatch')
    if reference.labels!=manifest['labels']:raise RuntimeError('Label order mismatch')
    samples=[];skipped=[];coverage=Counter()
    rows=list(csv.DictReader((args.experiment_dir/'split_manifest.csv').open(newline='')))
    for row in rows:
        if row['split']!='validation' or coverage[row['class_name']]>=args.samples_per_class:continue
        path=args.dataset/Path(row['relative_path'].replace(chr(92),'/'))
        frames,indices=read_window(path,manifest['fps'],count)
        if len(frames)!=count:
            skipped.append({'video':row['relative_path'],'samples':len(frames),'reason':'shorter than fixed window; no duplicate padding'})
            continue
        rgb=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames])
        image=tf.cast(tf.image.resize_with_pad(tf.convert_to_tensor(rgb),shape[2],shape[3])[tf.newaxis],tf.float32)/255.
        samples.append({'video':row['relative_path'],'class':row['class_name'],
                        'frame_indices':indices,'frames':frames,'array':image.numpy()})
        coverage[row['class_name']]+=1
    if not samples:raise RuntimeError('No complete validation windows available')
    print('SAMPLES_READY',len(samples),flush=True)
    sessions={};cold={}
    for precision in ('fp32','fp16'):
        if hashlib.sha256((args.export_dir/manifest[precision]['file']).read_bytes()).hexdigest()!=manifest[precision]['sha256']:
            raise RuntimeError(f'Artifact checksum mismatch: {precision}')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        # Reject fallback for computational operators rather than claiming GPU
        # from the provider's presence alone. Fixed-shape constants are folded.
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        start=time.perf_counter()
        session=ort.InferenceSession(str(args.export_dir/manifest[precision]['file']),options,
            providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':'1' if precision=='fp16' else '0',
                       'cudnn_conv_algo_search':'HEURISTIC'})])
        session.disable_fallback()
        if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA provider did not load')
        sessions['onnx_cuda_'+precision]=session
        cold[precision]=1000*(time.perf_counter()-start)
        print('SESSION_READY',precision,session.get_providers(),flush=True)
    names=['tensorflow_cpu_fp32',*sessions]
    def run(name,array):
        if name=='tensorflow_cpu_fp32':return reference.predict(image=tf.convert_to_tensor(array))['logits'].numpy()[0]
        session=sessions[name]
        return session.run(None,{session.get_inputs()[0].name:array})[0][0]
    for name in names:
        for _ in range(3):run(name,samples[0]['array'])
        print('WARMED',name,flush=True)
    comparisons=[]
    for sample in samples:
        base=run(names[0],sample['array'])
        record={k:sample[k] for k in ('video','class','frame_indices')}
        record['reference_label']=reference.labels[int(base.argmax())];record['backends']={}
        p=np.exp(base-base.max());p/=p.sum()
        for name in sessions:
            actual=run(name,sample['array'])
            if not np.isfinite(actual).all():raise RuntimeError(f'Nonfinite output: {name}')
            q=np.exp(actual-actual.max());q/=q.sum()
            record['backends'][name]={'max_abs_logit_error':float(np.abs(base-actual).max()),
                'max_abs_probability_error':float(np.abs(p-q).max()),
                'top1_agreement':bool(base.argmax()==actual.argmax()),
                'label':reference.labels[int(actual.argmax())]}
        comparisons.append(record)
        print('COMPARED',json.dumps(record),flush=True)
    rng=random.Random(42);timings={name:[] for name in names}
    for _ in range(args.repeats):
        order_samples=list(range(len(samples)));rng.shuffle(order_samples)
        for i in order_samples:
            order=names.copy();rng.shuffle(order)
            for name in order:
                start=time.perf_counter();run(name,samples[i]['array'])
                timings[name].append(1000*(time.perf_counter()-start))
    # Profile one independent run per ONNX precision to prove actual placement.
    placements={}
    for precision in ('fp32','fp16'):
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.enable_profiling=True;options.profile_file_prefix=str(args.output_dir/f'placement_{precision}')
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        session=ort.InferenceSession(str(args.export_dir/manifest[precision]['file']),options,
            providers=[('CUDAExecutionProvider',{'use_tf32':'1' if precision=='fp16' else '0','cudnn_conv_algo_search':'HEURISTIC'})])
        session.disable_fallback()
        session.run(None,{session.get_inputs()[0].name:samples[0]['array']})
        events=json.loads(Path(session.end_profiling()).read_text())
        nodes=[e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
        providers=Counter(e['args']['provider'] for e in nodes)
        ops={}
        for e in nodes:ops.setdefault(e['args']['provider'],Counter())[e['args'].get('op_name')]+=1
        if not providers.get('CUDAExecutionProvider'):raise RuntimeError('No CUDA execution observed')
        placements[precision]={'executed_node_providers':dict(providers),'operations_by_provider':{k:dict(v) for k,v in ops.items()}}
    result={'scope':'Warm model invocation with identical preprocessed CPU tensors; ONNX includes host/device transfers and synchronous CPU output. Excludes video decoding, RGB conversion, resizing and display.',
        'live_camera_tested':False,'gpu':torch.cuda.get_device_name(),'onnxruntime':ort.__version__,
        'tensorflow':tf.__version__,'window_frames':count,'window_seconds':manifest['window_seconds'],
        'artifact_manifest':manifest,
        'cuda_provider_options_requested':{'fp32':{'use_tf32':'0'},'fp16':{'use_tf32':'1'},'both':{'cudnn_conv_algo_search':'HEURISTIC','cpu_ep_fallback':False}},
        'sample_count':len(samples),'class_coverage':dict(coverage),'skipped_short_videos':skipped,
        'timings':{k:stats(v) for k,v in timings.items()},'session_initialization_ms':cold,
        'comparisons':comparisons,'placement':placements,
        'limitation':'Numerical agreement on validation windows is not a new blind evaluation or live-camera accuracy measurement.'}
    for name in sessions:
        records=[r['backends'][name] for r in comparisons]
        result.setdefault('agreement',{})[name]={'top1_agreement_count':sum(r['top1_agreement'] for r in records),
            'n':len(records),'max_abs_logit_error':max(r['max_abs_logit_error'] for r in records),
            'max_abs_probability_error':max(r['max_abs_probability_error'] for r in records)}
    (args.output_dir/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('BENCHMARK_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='comparisons'}),flush=True)


if __name__=='__main__':main()
