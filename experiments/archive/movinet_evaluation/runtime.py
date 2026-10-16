"""Local CPU inference using unmodified official Google MoViNet SavedModels."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tf_deps'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
import numpy as np
import cv2
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
LABELS = [line.strip() for line in (ROOT/'kinetics_600_labels.txt').read_text().splitlines()]


def read_video(path,target_fps=5.0,max_frames=50):
    path = Path(path)
    if path.suffix.lower()=='.gif':
        # Preserve the exact reference example's frame sequence.
        frames = tf.io.decode_gif(tf.io.read_file(str(path))).numpy()
        frames = frames[:max_frames]
        return frames,{'path':str(path),'frame_indices':list(range(len(frames))),
                       'sampling':'all GIF frames, matching the official tutorial'}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('Cannot open video: '+str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        capture.release()
        raise ValueError('Video has no valid frame rate: '+str(path))
    frames,indices = [],[]
    index = 0
    next_time = 0.0
    while len(frames)<max_frames:
        ok,frame = capture.read()
        if not ok:
            break
        if index/fps+1e-9>=next_time:
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            indices.append(index)
            next_time += 1/target_fps
        index += 1
    capture.release()
    if not frames:
        raise ValueError('No decoded frames: '+str(path))
    return np.stack(frames),{'path':str(path),'source_fps':float(fps),
                             'target_fps':target_fps,'frame_indices':indices,
                             'sampling':'first decoded frame at or after each 5-fps timestamp; max 50 frames'}


def preprocess(rgb_frames,resolution):
    # The published Hub contract is RGB [B,T,H,W,C] float32 in [0,1].
    resized = tf.image.resize(tf.convert_to_tensor(rgb_frames),[resolution,resolution])
    return tf.cast(resized[tf.newaxis],tf.float32)/255.0


def top5(logits):
    probs = tf.nn.softmax(logits,axis=-1).numpy().reshape(-1)
    assert probs.shape==(600,) and np.isfinite(probs).all()
    return [{'index':int(index),'label':LABELS[index],'probability':float(probs[index])}
            for index in np.argsort(probs)[-5:][::-1]]


class MovinetRunner:
    def __init__(self,name='a2_stream'):
        if name not in ['a0_stream','a2_stream','a2_base']:
            raise ValueError('Unsupported downloaded model: '+name)
        self.name = name
        self.streaming = name.endswith('stream')
        self.resolution = 172 if name.startswith('a0') else 224
        self.model = tf.saved_model.load(str(ROOT/'models'/name))
        self.states = None

    def reset(self,image_shape):
        self.states = self.model.init_states(tf.convert_to_tensor(image_shape,dtype=tf.int32)) if self.streaming else None

    def infer(self,images):
        if self.streaming:
            if self.states is None:
                self.reset(images.shape)
            logits,self.states = self.model({**self.states,'image':images})
        else:
            logits = self.model.signatures['serving_default'](image=images)['classifier_head']
        values = logits.numpy()
        assert values.shape==(1,600) and np.isfinite(values).all()
        return values

    def state_info(self):
        if self.states is None:
            return None
        assert all(np.isfinite(value.numpy()).all() for value in self.states.values())
        return {'tensor_count':len(self.states),
                'bytes':sum(value.numpy().nbytes for value in self.states.values()),
                'head_frame_count':int(self.states['state/head/pool_frame_count'].numpy().reshape(-1)[0])}
