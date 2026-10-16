"""Partial-backbone adaptation of the original, restored official A0 model."""
import hashlib
import json
import tempfile
from pathlib import Path
from kard_model import FrozenBackbone,read_clip,variable_digest,metrics,tf,np
from kard_config import ROOT,LABELS

OUT=ROOT/'kard_a0_upper_finetune'
BASE=ROOT/'kard_a0_head_100pct'
OUT.mkdir(exist_ok=True)
(OUT/'tmp').mkdir(exist_ok=True)
tempfile.tempdir=str(OUT/'tmp')
PREFIXES=('b4/l2/','b4/l3/','head/')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class UpperTuner:
    def __init__(self):
        self.features=FrozenBackbone()
        self.backbone=self.features.backbone
        self.model=self.features.model
        self.head=tf.keras.models.load_model(BASE/'kard_head.keras')
        self.layers=[getattr(self.backbone,f'layer_with_weights-{i}') for i in [14,15,16]]
        assert [layer.variables[0].name.split('/')[0:2] for layer in self.layers]==[['b4','l2'],['b4','l3'],['head','project']]
        self.upper=[v for v in self.backbone.variables if v.trainable and v.name.startswith(PREFIXES) and '/bn/' not in v.name]
        selected={v.ref() for v in self.upper}
        self.frozen=[v for v in self.backbone.variables if v.ref() not in selected]
        self.initial_upper=[v.numpy().copy() for v in self.upper]
        self.frozen_hash=variable_digest(self.frozen)
        self.bn=[v for v in self.backbone.variables if '/bn/' in v.name]
        self.bn_hash=variable_digest(self.bn)
        self.input_spec=tf.TensorSpec([None,None,6,6,104],tf.float32)
        self.pooled=tf.function(self._pooled,input_signature=[self.input_spec])
        self.predict=tf.function(lambda x:self.head(self.pooled(x),training=False),input_signature=[self.input_spec])

    def _pooled(self,x):
        states=self.model.init_states(tf.stack([tf.shape(x)[0],1,172,172,3]))
        for layer in self.layers:x,states=layer(x,states=states,training=False)
        return tf.reshape(x,[tf.shape(x)[0],480])

    def full_endpoints(self,images):
        states=self.model.init_states(tf.constant([1,1,172,172,3]))
        endpoints,_=self.backbone({**states,'image':images})
        return endpoints

    def save_weights(self,path):
        arrays={f'upper_{i}':v.numpy() for i,v in enumerate(self.upper)}
        arrays.update({f'head_{i}':v for i,v in enumerate(self.head.get_weights())})
        arrays['upper_names']=np.asarray([v.name for v in self.upper])
        np.savez_compressed(path,**arrays)

    def restore_weights(self,path):
        with np.load(path) as saved:
            assert list(saved['upper_names'])==[v.name for v in self.upper]
            for i,v in enumerate(self.upper):v.assign(saved[f'upper_{i}'])
            self.head.set_weights([saved[f'head_{i}'] for i in range(len(self.head.get_weights()))])

    def verify_frozen(self):
        assert variable_digest(self.frozen)==self.frozen_hash,'A frozen variable changed'
        assert variable_digest(self.bn)==self.bn_hash,'Batch normalization changed'

def cache_stem(record):return record['split']+'_'+record['relative_path'].replace('\\','_').replace('/','_').removesuffix('.mp4')

def predict_records(tuner,records):
    logits=[]
    for r in records:
        with np.load(OUT/'activation_cache'/(cache_stem(r)+'.npz')) as data:
            logits.append(tuner.predict(tf.convert_to_tensor(data['activation'][None])).numpy()[0])
    return np.stack(logits)
