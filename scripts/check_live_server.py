"""Exercise HTTP and a real TCP camera protocol using in-memory synthetic JPEGs.

Requires an already running server. Does not access the physical camera or record.
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path
import urllib.request
import numpy as np
from PIL import Image
from websockets.sync.client import connect

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--port",type=int,default=8765);args=parser.parse_args()
    address=f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(address+"/api/model",timeout=10) as response:info=json.load(response)
    count=round(info["fps"]*info["window_seconds"])
    with connect(f"ws://127.0.0.1:{args.port}/ws/live",origin=address,max_size=2_000_000,open_timeout=10) as ws:
        assert json.loads(ws.recv(timeout=10))["type"]=="ready"
        for i in range(count):
            frame=np.zeros((64,96,3),np.uint8);frame[...,i%3]=100+i%100
            b=io.BytesIO();Image.fromarray(frame).save(b,format="JPEG")
            ws.send(json.dumps({"type":"frame","timestamp_ms":i*1000/info["fps"],"jpeg":base64.b64encode(b.getvalue()).decode()}))
        prediction=None
        for _ in range(count+5):
            message=json.loads(ws.recv(timeout=60))
            if message["type"]=="error":raise RuntimeError(message["message"])
            if message["type"]=="prediction":prediction=message;break
        assert prediction is not None and len(prediction["logits"])==len(info["labels"])
        ws.send(json.dumps({"type":"explain","method":"contrastive"}))
        explanation=json.loads(ws.recv(timeout=60))
        assert explanation["type"]=="explanation",explanation
        assert explanation["result"]["data"]["reconstruction_error"]<1e-4
    result={"model":info["name"],"tcp_websocket_prediction":True,"tcp_websocket_contrastive_explanation":True,
            "sampled_frames":count,"processing_ms":prediction["processing_ms"],"physical_camera_tested":False,
            "frames_recorded":False,"note":"Synthetic JPEG protocol check against the real model, not HAR accuracy evidence"}
    (ROOT/"docs/live_protocol_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
