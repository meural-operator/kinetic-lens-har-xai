"""Real pooled backbone features and the replacement KARD classifier."""
import hashlib
import json
import tempfile
from pathlib import Path
import cv2
from kard_config import OUT,PROTOCOL
# Keras uses NamedTemporaryFile during .keras serialization. Keep that temporary
# storage inside the writable experiment workspace in this restricted runtime.
WORK_TMP=OUT/'tmp'
WORK_TMP.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(WORK_TMP)
from runtime import tf, np, ROOT

def sample_frame_indices(fps,count,policy):
    assert np.isfinite(fps) and fps>0 and count>0
    window_start=max(0.0,(count/fps-policy['max_frames']/policy['target_fps'])/2)
    targets=np.arange(window_start,count/fps,1/policy['target_fps'])[:policy['max_frames']]
    wanted=np.ceil(targets*fps-1e-8).astype(int)
    # A nominal timestamp before count/fps may still round to index count.
    # Omit that unavailable terminal target; do not repeat or pad real frames.
    dropped=int(np.sum(wanted>=count))
    wanted=wanted[wanted<count]
    assert len(wanted)>0 and np.all(np.diff(wanted)>0)
    return window_start,wanted,dropped

def read_clip(path,protocol=None):
    policy=PROTOCOL if protocol is None else protocol
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot open {path}')
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps>0 and count>0
    window_start,wanted,dropped=sample_frame_indices(fps,count,policy)
    frames=[];indices=[];index=0;target=0
    while target<len(wanted):
        ok,frame=cap.read()
        if not ok:break
        if index>=wanted[target]:
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));indices.append(index);target+=1
        index+=1
    cap.release()
    assert len(frames)==len(wanted)>0
    images=tf.image.resize_with_pad(tf.convert_to_tensor(np.stack(frames)),policy['resolution'],policy['resolution'])
    images=tf.cast(images[tf.newaxis],tf.float32)/255.0
    return images,{'frame_indices':indices,'window_start_seconds':window_start,
                   'sampled_frames':len(frames),'source_fps':fps,'target_fps':policy['target_fps'],
                   'max_frames':policy['max_frames'],'resolution':policy['resolution'],
                   'unavailable_terminal_targets_skipped':dropped}

def variable_digest(variables):
    digest=hashlib.sha256()
    for variable in variables:
        digest.update(variable.name.encode());digest.update(variable.numpy().tobytes())
    return digest.hexdigest()

class FrozenBackbone:
    def __init__(self):
        self.model=tf.saved_model.load(str(ROOT/'models'/'a0_stream'))
        self.backbone=self.model._backbone
        self.initial_hash=variable_digest(self.backbone.variables)
        states=self.model.init_states(tf.constant([1,1,172,172,3]))
        self.state_specs={k:tf.TensorSpec(v.shape,v.dtype,name=k.replace('/','_')) for k,v in states.items()}
        self.step=tf.function(self._step,input_signature=[tf.TensorSpec([1,None,172,172,3],tf.float32),self.state_specs])

    def _step(self,images,states):
        endpoints,next_states=self.backbone({**states,'image':images})
        return tf.reshape(endpoints['head'],[1,480]),next_states

    def extract(self,images,chunk=8):
        states=self.model.init_states(tf.constant([1,1,172,172,3]))
        for offset in range(0,int(images.shape[1]),chunk):
            features,states=self.step(images[:,offset:offset+chunk],states)
        assert int(states['state/head/pool_frame_count'].numpy()[0])==int(images.shape[1])
        values=features.numpy()[0]
        assert values.shape==(480,) and np.isfinite(values).all()
        return values,states

def build_head(mean,scale,l2):
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(480,)),
        tf.keras.layers.Normalization(mean=mean,variance=np.square(scale),name='train_only_standardization'),
        tf.keras.layers.Dropout(0.25,name='head_dropout'),
        tf.keras.layers.Dense(18,kernel_regularizer=tf.keras.regularizers.L2(l2),name='kard_classifier')
    ],name='kard_head')
    assert sum(int(np.prod(v.shape)) for v in model.trainable_variables)==8658
    return model

def metrics(logits,labels):
    logits=np.asarray(logits);labels=np.asarray(labels,dtype=int)
    prediction=logits.argmax(axis=1)
    matrix=np.zeros((18,18),dtype=int)
    np.add.at(matrix,(labels,prediction),1)
    tp=np.diag(matrix);actual=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros(18,float),where=predicted>0)
    recall=np.divide(tp,actual,out=np.zeros(18,float),where=actual>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(18,float),where=precision+recall>0)
    x=logits-logits.max(axis=1,keepdims=True)
    probabilities=np.exp(x)/np.exp(x).sum(axis=1,keepdims=True)
    return {'n':len(labels),'accuracy':float((prediction==labels).mean()),
        'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),
        'cross_entropy':float(-np.log(np.maximum(probabilities[np.arange(len(labels)),labels],1e-12)).mean()),
        'confusion_matrix':matrix.tolist(),'precision_per_class':precision.tolist(),
        'recall_per_class':recall.tolist(),'f1_per_class':f1.tolist(),'support_per_class':actual.tolist()}

class KardInference(tf.Module):
    """End-to-end clip inference; each call resets streaming state."""
    def __init__(self,features,head):
        super().__init__()
        self.backbone=features.backbone
        self.init_states=features.model.init_states
        self.head=head

    @tf.function(input_signature=[tf.TensorSpec([1,None,172,172,3],tf.float32,name='image')])
    def classify_clip(self,image):
        tf.debugging.assert_positive(tf.shape(image)[1])
        tf.debugging.assert_all_finite(image,'Input contains NaN/Inf')
        tf.debugging.assert_greater_equal(image,0.0)
        tf.debugging.assert_less_equal(image,1.0)
        states=self.init_states(tf.constant([1,1,172,172,3]))
        endpoints,states=self.backbone({**states,'image':image})
        features=tf.reshape(endpoints['head'],[1,480])
        logits=self.head(features,training=False)
        return {'logits':logits,'probabilities':tf.nn.softmax(logits),
                'class_id':tf.argmax(logits,axis=-1,output_type=tf.int32)}
