"""TorchScript video logits adapter, with explicit CPU/CUDA selection."""
from dataclasses import asdict
import hashlib
import json
import cv2
import numpy as np
from ..contracts import InputSpec, validate_clip, validate_logits
from ..registry import artifact_path


class TorchScriptModel:
    capabilities=frozenset({"predict"})

    def __init__(self, config):
        import torch
        self.torch=torch; self.input_spec=InputSpec(**config["input"])
        self.name=config.get("name","TorchScript HAR")
        self.labels=json.loads(artifact_path(config,"labels").read_text(encoding="utf-8-sig"))
        self.path=artifact_path(config,"weights")
        self.device=config.get("device","cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
        self.model=torch.jit.load(str(self.path),map_location=self.device).eval()
        self.layout=config.get("layout","NCTHW")
        if self.layout not in ("NCTHW","NTCHW","NTHWC"): raise ValueError("Unknown input layout")
        self.mean=np.array(config.get("mean",[0,0,0]),np.float32).reshape(1,1,1,3)
        self.std=np.array(config.get("std",[1,1,1]),np.float32).reshape(1,1,1,3)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0): raise ValueError("Invalid normalization")
        self.fixed_frames=config.get("fixed_frames")
        self.hash=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if config.get("sha256") and config["sha256"]!=self.hash: raise ValueError("TorchScript hash mismatch")

    def predict(self,frames):
        frames=validate_clip(frames,self.input_spec)
        if self.fixed_frames and len(frames)!=self.fixed_frames: raise ValueError(f"Expected {self.fixed_frames} frames")
        size=self.input_spec.resolution
        x=np.stack([cv2.resize(f,(size,size),interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.float32)/255
        x=(x-self.mean)/self.std
        if self.layout=="NCTHW": x=x.transpose(3,0,1,2)
        elif self.layout=="NTCHW": x=x.transpose(0,3,1,2)
        tensor=self.torch.from_numpy(np.ascontiguousarray(x[None])).to(self.device)
        with self.torch.inference_mode(): values=self.model(tensor)
        if not isinstance(values,self.torch.Tensor) or tuple(values.shape)!=(1,len(self.labels)):
            raise ValueError("Export a logits wrapper returning [1,num_classes]")
        return validate_logits(values.detach().cpu().numpy()[0],self.labels)

    def provenance(self):
        return {"name":self.name,"runtime":"PyTorch TorchScript","device":self.device,"sha256":self.hash,
                "input":asdict(self.input_spec),"layout":self.layout,"resize":"OpenCV bilinear stretch",
                "state_policy":"adapter accepts stateless exported graphs only"}
