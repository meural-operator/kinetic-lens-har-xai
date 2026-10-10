import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from kinetic_lens.contracts import InputSpec, prediction, validate_clip
from kinetic_lens.registry import load_model, make_method, methods_for
from kinetic_lens.video import RollingWindow, sample_indices, read_video
from kinetic_lens.server import create_app
from kinetic_lens.xai.concept import fit_probe


class ToyLinearModel:
    name="Independent toy oracle"
    labels=["rising","falling","constant"]
    input_spec=InputSpec(fps=5,max_frames=20,resolution=64)
    capabilities=frozenset({"predict","linear_head"})
    def linear_evidence(self,frames):
        z=np.array([float(frames[-1].mean()-frames[0].mean())/255,float(frames.mean())/255])
        w=np.array([[4.,-4.,0.],[0.,0.,1.]])
        b=np.array([0.,0.,.1]); return dict(features=z,weights=w,bias=b,logits=z@w+b)
    def predict(self,frames): return self.linear_evidence(frames)["logits"]
    def provenance(self): return {"name":self.name,"trained":False,"input":asdict(self.input_spec)}


def clip(): return np.stack([np.full((16,24,3),i*20,np.uint8) for i in range(10)])


def jpeg(frame):
    b=io.BytesIO();Image.fromarray(frame).save(b,format="JPEG");return base64.b64encode(b.getvalue()).decode()


def test_exact_contrast_and_reversal_have_independent_oracles():
    model=ToyLinearModel();x=clip()
    result=make_method("contrastive").explain(model,x,rival=1).data
    assert result["margin"]==pytest.approx(8*180/255)
    assert sum(result["contributions"])+result["bias_difference"]==pytest.approx(result["margin"])
    temporal=make_method("temporal").explain(model,x,bins=5).data
    reversal=next(r for r in temporal["measurements"] if r["intervention"]=="reverse")
    assert reversal["predicted_class"]==1
    assert reversal["target_logit_drop"]==pytest.approx(8*180/255)
    assert len(temporal["measurements"])==18


def test_target_not_winner_contrast_defaults_to_actual_winner():
    result=make_method("contrastive").explain(ToyLinearModel(),clip(),target=1).data
    assert result["rival"]==0 and result["margin"]<0


def test_prediction_rejects_nonfinite_and_uses_stable_softmax():
    r=prediction([10000,9999],["a","b"]);assert r["top3"][0]["score"]==pytest.approx(1/(1+np.exp(-1)))
    with pytest.raises(ValueError):prediction([np.nan,0],["a","b"])


@pytest.mark.parametrize("kwargs",[{"fps":0},{"fps":float("nan")},{"max_frames":0},{"resolution":0}])
def test_invalid_specs_fail(kwargs):
    with pytest.raises(ValueError):InputSpec(**kwargs)


def test_capabilities_are_not_silently_faked():
    model=load_model("configs/demo.json")
    assert [m["id"] for m in methods_for(model) if m["available"]]==["temporal"]
    with pytest.raises(ValueError,match="lacks"):make_method("spatial").explain(model,clip())


def test_custom_plugins_work_without_core_changes():
    model=load_model({"adapter":"examples.custom_model:ExampleModel","input":{"resolution":64,"max_frames":20}})
    result=make_method("examples.custom_xai:ReverseSensitivity").explain(model,clip())
    assert result.method=="custom_reverse"


def test_sampling_matches_the_audited_centered_timestamp_policy():
    spec=InputSpec();ids=sample_indices(30,600,spec)
    assert np.array_equal(ids,np.arange(150,450,6))
    assert np.array_equal(sample_indices(30,60,spec),np.arange(0,60,6))
    with pytest.raises(ValueError):sample_indices(2,30,spec)


def test_rolling_window_is_bounded_and_resets_after_gaps_or_resolution_change():
    buf=RollingWindow(InputSpec(),1.)
    frame=clip()[0]
    for i in range(100):buf.add(i/5,frame)
    assert len(buf.items)==5 and buf.snapshot()[1][0]==pytest.approx(19)
    with pytest.raises(ValueError):buf.add(19.8,frame)
    buf.add(21,frame);assert len(buf.items)==1 and buf.snapshot() is None
    buf.add(21.2,np.zeros((8,8,3),np.uint8));assert len(buf.items)==1


def test_clip_contract_rejects_float_wrong_axis_empty_or_long():
    for x in [clip().astype(float),clip().transpose(0,3,1,2),clip()[:0],np.repeat(clip(),6,axis=0)]:
        with pytest.raises(ValueError):validate_clip(x,InputSpec())


def test_probe_fits_available_information_without_touching_a_model():
    x=np.array([[1.,0.],[2.,1.],[-1.,0.],[-2.,1.]])
    p=fit_probe(x,[1,1,0,0],alpha=.1)
    assert np.all((x@p["weights"]+p["bias"]>.5)==[1,1,0,0])
    assert np.linalg.norm(p["direction"])==pytest.approx(1)
    with pytest.raises(ValueError):fit_probe(x,[1,1,1,1])


def test_rest_predict_and_malformed_input_and_hosts():
    app=create_app(ToyLinearModel(),window_seconds=1)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"]=="ready"
        assert "KineticLens" in client.get("/").text
        assert client.get("/api/model").json()["fps"]==5
        result=client.post("/api/predict",json={"frames":[jpeg(f) for f in clip()]})
        assert result.status_code==200 and result.json()["class_id"]==0
        assert client.post("/api/predict",json={"frames":["not-base64"]}).status_code==400
        assert client.post("/api/predict",json={"frames":[]}).status_code==400
        assert client.get("/",headers={"host":"evil.example"}).status_code==400


def test_live_websocket_produces_real_adapter_logits_and_clears_context():
    with TestClient(create_app(ToyLinearModel(),window_seconds=1,update_seconds=.1)) as client:
        with client.websocket_connect("/ws/live") as ws:
            assert ws.receive_json()["type"]=="ready"
            for i in range(5):
                ws.send_json({"type":"frame","timestamp_ms":i*200,"jpeg":jpeg(clip()[i])})
                message=ws.receive_json()
                if i<4:assert message["type"]=="buffer"
            if message["type"]!="prediction":message=ws.receive_json()
            assert message["type"]=="prediction" and message["class_id"]==0
            assert message["effective_fps"]==pytest.approx(5)
            ws.send_json({"type":"reset"});assert ws.receive_json()["type"]=="ready"
            ws.send_json({"type":"explain","method":"temporal"})
            assert "fresh prediction" in ws.receive_json()["message"]


def test_malformed_websocket_is_closed_and_cross_origin_rejected():
    with TestClient(create_app(ToyLinearModel(),window_seconds=1)) as client:
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json();ws.send_text("not JSON");assert ws.receive_json()["type"]=="error"
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/live",headers={"origin":"https://other.example"}):pass


def test_offline_decode_failure_is_explicit(tmp_path):
    with pytest.raises(ValueError):read_video(tmp_path/"missing.mp4",InputSpec())
