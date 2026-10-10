"""Export the selected trained KARD SavedModel for a fixed live inference window.

Input is the existing preprocessed RGB [1,T,172,172,3] float32 tensor in [0,1].
Every call resets MoViNet state, as the source SavedModel does. No retraining.
"""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import onnx
from onnxruntime.transformers import float16
from onnxconverter_common.float16 import sort_topology


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/onnx_export')
    parser.add_argument('--reuse-fp32', action='store_true',help='Reuse the FP32 file from this same export attempt')
    parser.add_argument('--cuda-fp16-ops',action='store_true',help='Allow CUDA-supported Min/Max/CumSum to remain FP16; must revalidate numerics and placement')
    args = parser.parse_args()
    protocol = json.loads((args.experiment_dir/'protocol.json').read_text())
    if not 2 <= args.frames <= protocol['max_frames']:
        parser.error('frames must be 2..max_frames')
    resolution = protocol['resolution']
    shape = [1,args.frames,resolution,resolution,3]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fp32 = args.output_dir/f'movinet-kard-{args.frames}f-fp32.onnx'
    fp16 = args.output_dir/f'movinet-kard-{args.frames}f-fp16.onnx'
    print('EXPORT_START',json.dumps({'shape':shape,'tensorflow':version('tensorflow'),
          'tf2onnx':version('tf2onnx')}),flush=True)
    if args.reuse_fp32:
        previous=json.loads((args.output_dir/'manifest.json').read_text())
        source_checks={
            'source_saved_model_pb_sha256':digest(args.experiment_dir/'saved_model/saved_model.pb'),
            'source_variables_sha256':digest(args.experiment_dir/'saved_model/variables/variables.data-00000-of-00001')}
        if any(previous.get(key)!=value for key,value in source_checks.items()):
            raise RuntimeError('Cached FP32 belongs to different source weights; export again without --reuse-fp32')
        if previous['fp32']['file']!=fp32.name or previous['fp32']['sha256']!=digest(fp32):
            raise RuntimeError('Cached FP32 checksum/provenance mismatch')
        if previous['labels']!=json.loads((args.experiment_dir/'labels.json').read_text()):
            raise RuntimeError('Cached FP32 label order mismatch')
        graph=onnx.load(str(fp32))
        actual=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
        if actual!=shape:raise RuntimeError('Cached FP32 input shape does not match this export')
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        import tf2onnx
        loaded = tf.saved_model.load(str(args.experiment_dir/'saved_model'))
        serving = loaded.signatures['serving_default']
        @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32,name='image')])
        def predict(image):
            return tf.identity(serving(image=image)['logits'],name='logits')
        graph,_ = tf2onnx.convert.from_function(predict,input_signature=predict.input_signature,
            opset=18,output_path=str(fp32))
    # tf2onnx can retain unused SavedModel constant-capture placeholders even
    # after folding their values. Remove only inputs unreferenced anywhere,
    # including nested graphs and graph outputs; never guess their values.
    used=set();queue=[graph.graph]
    while queue:
        current=queue.pop();used.update(v.name for v in current.output)
        for node in current.node:
            used.update(node.input)
            for attr in node.attribute:
                if attr.type==onnx.AttributeProto.GRAPH:queue.append(attr.g)
                elif attr.type==onnx.AttributeProto.GRAPHS:queue.extend(attr.graphs)
    dropped_inputs=[v.name for v in graph.graph.input if v.name not in used]
    inputs=[v for v in graph.graph.input if v.name in used]
    del graph.graph.input[:];graph.graph.input.extend(inputs)
    if len(inputs)!=1 or inputs[0].name!='image':raise RuntimeError('Export did not produce a single image input')
    onnx.checker.check_model(graph,full_check=True)
    onnx.save(graph,str(fp32))
    blocked=[op for op in float16.DEFAULT_OP_BLOCK_LIST if not(args.cuda_fp16_ops and op in {'Min','Max','CumSum'})]
    half = float16.convert_float_to_float16(graph,keep_io_types=True,
        min_positive_val=5.96e-8,max_finite_val=65504.0,op_block_list=blocked)
    # Shared Clip bounds can cause ORT's converter to emit the same Cast
    # producer repeatedly. Remove only provably identical Cast operations;
    # differing duplicate producers remain an error, never a guessed repair.
    producers={};kept=[];deduplicated=0
    for node in half.graph.node:
        signature=(node.op_type,node.domain,tuple(node.input),
                   tuple(a.SerializeToString() for a in node.attribute))
        outputs=tuple(node.output)
        if outputs in producers:
            if node.op_type!='Cast' or producers[outputs]!=signature:
                raise RuntimeError(f'Non-equivalent duplicate ONNX producer: {outputs}')
            deduplicated+=1
            continue
        producers[outputs]=signature;kept.append(node)
    del half.graph.node[:];half.graph.node.extend(kept)
    sort_topology(half.graph)
    onnx.checker.check_model(half,full_check=True)
    onnx.save(half,str(fp16))
    manifest = {
        'source_saved_model_pb_sha256':digest(args.experiment_dir/'saved_model/saved_model.pb'),
        'source_variables_sha256':digest(args.experiment_dir/'saved_model/variables/variables.data-00000-of-00001'),
        'source_experiment':str(args.experiment_dir.resolve()),
        'labels':json.loads((args.experiment_dir/'labels.json').read_text()),
        'input_shape':shape,'input_name':graph.graph.input[0].name,
        'output_name':graph.graph.output[0].name,'input_dtype':'float32',
        'preprocessing':f'RGB float32 [0,1]; bilinear resize with padding to {resolution}x{resolution}; fixed {args.frames}-sample window at {protocol["target_fps"]:g} fps.',
        'source_clip_protocol_preprocessing':protocol['preprocessing'],
        'window_seconds':args.frames/protocol['target_fps'],
        'fps':protocol['target_fps'],'states':'fresh initial states on every window',
        'removed_unused_capture_inputs':dropped_inputs,
        'fp32':{'file':fp32.name,'sha256':digest(fp32),'bytes':fp32.stat().st_size},
        'fp16':{'file':fp16.name,'sha256':digest(fp16),'bytes':fp16.stat().st_size,
                'keep_io_types':True,'converter':'official onnxruntime.transformers.float16',
                'min_positive_val':5.96e-8,'max_finite_val':65504.0,
                'deduplicated_equivalent_casts':deduplicated,
                'op_block_list':blocked},
        'versions':{'tensorflow':version('tensorflow'),'tf2onnx':version('tf2onnx'),'onnx':onnx.__version__},
        'scope':'Inference export only; not gradient/XAI hooks. Runtime and equivalence validation required.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('EXPORT_COMPLETE',json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
