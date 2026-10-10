import json
import numpy as np
import pytest
from kinetic_lens.registry import load_model


def test_onnx_actual_runtime_tensor_layout_and_logits(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx=pytest.importorskip("onnx")
    from onnx import helper,TensorProto
    graph=helper.make_graph([helper.make_node("ReduceMean",["image"],["logits"],axes=[2,3,4],keepdims=0)],"RGB mean",
        [helper.make_tensor_value_info("image",TensorProto.FLOAT,[1,3,4,8,8])],
        [helper.make_tensor_value_info("logits",TensorProto.FLOAT,[1,3])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",13)]);model.ir_version=10
    path=tmp_path/"fixture.onnx";onnx.save(model,str(path))
    (tmp_path/"labels.json").write_text(json.dumps(["r","g","b"]))
    cfg={"adapter":"onnx","weights":str(path),"labels":str(tmp_path/"labels.json"),
         "input":{"fps":5,"max_frames":4,"resolution":8},"fixed_frames":4,"layout":"NCTHW"}
    adapter=load_model(cfg);frames=np.zeros((4,8,8,3),np.uint8);frames[...,1]=255
    assert np.allclose(adapter.predict(frames),[0,1,0])
    with pytest.raises(ValueError,match="frames"):adapter.predict(frames[:2])
    cfg["providers"]=["NonexistentExecutionProvider"]
    with pytest.raises(ValueError,match="Unavailable"):load_model(cfg)
