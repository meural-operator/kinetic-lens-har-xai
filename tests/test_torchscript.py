import json
import numpy as np
import pytest
from kinetic_lens.registry import load_model


def test_torchscript_adapter_actual_runtime_and_layout(tmp_path):
    torch=pytest.importorskip("torch")
    class RGBMean(torch.nn.Module):
        def forward(self,x): return x.mean(dim=(2,3,4))
    path=tmp_path/"fixture.ts"
    torch.jit.trace(RGBMean(),torch.zeros(1,3,4,8,8)).save(str(path))
    (tmp_path/"labels.json").write_text(json.dumps(["r","g","b"]))
    cfg={"adapter":"torchscript","weights":str(path),"labels":str(tmp_path/"labels.json"),
         "input":{"fps":5,"max_frames":4,"resolution":8},"fixed_frames":4,"layout":"NCTHW"}
    model=load_model(cfg);frames=np.zeros((4,8,8,3),np.uint8);frames[...,0]=255
    assert np.allclose(model.predict(frames),[1,0,0])
    with pytest.raises(ValueError,match="frames"):model.predict(frames[:2])
