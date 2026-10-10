"""Stateless single-input video ONNX adapter; streaming graphs need a custom adapter."""
from dataclasses import asdict
import hashlib
import json
import cv2
import numpy as np
from ..contracts import InputSpec, validate_clip, validate_logits
from ..registry import artifact_path


class ONNXModel:
    capabilities = frozenset({"predict"})

    def __init__(self, config):
        import onnxruntime as ort
        self.input_spec = InputSpec(**config["input"])
        self.name = config.get("name", "ONNX HAR")
        self.labels = json.loads(artifact_path(config,"labels").read_text(encoding="utf-8-sig"))
        self.path = artifact_path(config,"weights")
        providers = config.get("providers",["CPUExecutionProvider"])
        unavailable = set(providers)-set(ort.get_available_providers())
        if unavailable: raise ValueError(f"Unavailable ONNX providers: {sorted(unavailable)}")
        self.session = ort.InferenceSession(str(self.path),providers=providers)
        if len(self.session.get_inputs()) != 1:
            raise ValueError("Multi-input/stateful ONNX graphs require a custom model adapter")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name; self.input_dtype = {"tensor(float)":np.float32,"tensor(float16)":np.float16}.get(inp.type)
        if self.input_dtype is None: raise ValueError("This adapter expects float32/float16 image input")
        self.output = config.get("output",self.session.get_outputs()[0].name)
        self.layout = config.get("layout","NCTHW")
        if self.layout not in ("NCTHW","NTHWC","NTCHW"): raise ValueError("Unsupported tensor layout")
        self.resize = config.get("resize","stretch")
        if self.resize not in ("stretch","pad"): raise ValueError("resize must be stretch or pad")
        self.mean = np.asarray(config.get("mean",[0,0,0]),np.float32).reshape(1,1,1,3)
        self.std = np.asarray(config.get("std",[1,1,1]),np.float32).reshape(1,1,1,3)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0): raise ValueError("Invalid normalization")
        self.frames_required = config.get("fixed_frames")
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if config.get("sha256") and self.hash != config["sha256"]: raise ValueError("ONNX hash mismatch")

    def predict(self, frames):
        x = validate_clip(frames,self.input_spec)
        if self.frames_required and len(x) != self.frames_required:
            raise ValueError(f"Model requires exactly {self.frames_required} sampled frames")
        size = self.input_spec.resolution; images = []
        for frame in x:
            if self.resize == "stretch": image = cv2.resize(frame,(size,size),interpolation=cv2.INTER_LINEAR)
            else:
                h,w=frame.shape[:2]; ratio=size/max(h,w); rh,rw=max(1,int(h*ratio)),max(1,int(w*ratio))
                image=np.zeros((size,size,3),np.uint8); y,z=(size-rh)//2,(size-rw)//2
                image[y:y+rh,z:z+rw]=cv2.resize(frame,(rw,rh),interpolation=cv2.INTER_LINEAR)
            images.append(image)
        tensor=(np.stack(images).astype(np.float32)/255-self.mean)/self.std
        if self.layout=="NCTHW": tensor=tensor.transpose(3,0,1,2)
        elif self.layout=="NTCHW": tensor=tensor.transpose(0,3,1,2)
        values=self.session.run([self.output],{self.input_name:np.ascontiguousarray(tensor[None],dtype=self.input_dtype)})[0]
        if values.shape != (1,len(self.labels)): raise ValueError("Selected ONNX output must be [1,num_classes] raw logits")
        return validate_logits(values[0],self.labels)

    def provenance(self):
        return {"name":self.name,"runtime":"ONNX Runtime","providers":self.session.get_providers(),
                "sha256":self.hash,"input":asdict(self.input_spec),"layout":self.layout,"resize":self.resize,
                "mean":self.mean.ravel().tolist(),"std":self.std.ravel().tolist(),"state_policy":"stateless graph",
                "provider_note":"Provider list is not a per-operation GPU placement audit"}
