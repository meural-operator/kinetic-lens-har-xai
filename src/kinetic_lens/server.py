"""Loopback dashboard: per-connection buffers, bounded work, no camera recording."""
import asyncio
import base64
import io
import json
import time
from urllib.parse import urlparse
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from .contracts import prediction
from .registry import METHODS, make_method, methods_for
from .video import RollingWindow

MAX_MESSAGE=700_000
WEB=Path(__file__).resolve().parent/"web"


def decode_frame(encoded):
    if not isinstance(encoded,str) or len(encoded)>600_000: raise ValueError("Frame exceeds size limit")
    raw=base64.b64decode(encoded,validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "JPEG" or max(image.size)>1024 or min(image.size)<8:
            raise ValueError("Expected an 8..1024 pixel JPEG frame")
        return np.asarray(image.convert("RGB"),dtype=np.uint8).copy()


def options_for(method,payload):
    allowed={"target","rival","bins","seed","size","reference"}
    options={k:v for k,v in payload.items() if k in allowed}
    for key in ("target","rival","bins","seed","size"):
        if key in options and (isinstance(options[key],bool) or not isinstance(options[key],int)):
            raise ValueError(f"{key} must be an integer")
    if method not in METHODS: raise ValueError("Dashboard accepts registered built-in methods only")
    return options


def create_app(model,window_seconds=4.,update_seconds=1.):
    if update_seconds<.1 or update_seconds>60: raise ValueError("Invalid update interval")
    RollingWindow(model.input_spec,window_seconds)
    app=FastAPI(title="KineticLens HAR & XAI Studio")
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","[::1]","testserver"])
    work_lock=asyncio.Lock()

    @app.get("/api/model")
    def info():
        return {"name":model.name,"labels":model.labels,"fps":model.input_spec.fps,
                "max_frames":model.input_spec.max_frames,"window_seconds":window_seconds,
                "update_seconds":update_seconds,"methods":methods_for(model),"provenance":model.provenance()}

    @app.get("/api/health")
    def health(): return {"status":"ready","model":model.name}

    @app.get("/")
    def index(): return FileResponse(WEB/"index.html")

    @app.get("/assets/{filename}")
    def asset(filename:str):
        if filename not in ("app.js","style.css"): raise HTTPException(404)
        return FileResponse(WEB/filename)

    @app.post("/api/predict")
    async def predict_api(request:Request):
        chunks=[]; size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>min(32_000_000,model.input_spec.max_frames*600_000): raise HTTPException(413,"Clip payload too large")
            chunks.append(chunk)
        try:
            payload=json.loads(b"".join(chunks)); encoded=payload["frames"]
            if not isinstance(encoded,list) or not 1<=len(encoded)<=model.input_spec.max_frames: raise ValueError("Invalid frame count")
            frames=np.stack([decode_frame(x) for x in encoded])
            if work_lock.locked(): raise HTTPException(429,"Model busy; retry with the latest window")
            async with work_lock:
                start=time.perf_counter(); z=await run_in_threadpool(model.predict,frames)
            return {**prediction(z,model.labels),"processing_ms":1000*(time.perf_counter()-start),"sampled_frames":len(frames)}
        except HTTPException: raise
        except (ValueError,TypeError,KeyError,UnidentifiedImageError) as error: raise HTTPException(400,str(error)) from error

    @app.websocket("/ws/live")
    async def camera(ws:WebSocket):
        origin=ws.headers.get("origin")
        if origin and urlparse(origin).netloc!=ws.headers.get("host"):
            await ws.close(code=1008); return
        await ws.accept()
        buffer=RollingWindow(model.input_spec,window_seconds)
        send_lock=asyncio.Lock(); task=None; next_update=0.; captured=0; sampled=0; dropped=0
        latest=None; last_completed=None

        async def send(message):
            async with send_lock: await ws.send_json(message)

        async def infer(snapshot):
            nonlocal latest,last_completed
            frames,timestamps=snapshot
            try:
                async with work_lock:
                    start=time.perf_counter(); z=await run_in_threadpool(model.predict,frames)
                result={"type":"prediction",**prediction(z,model.labels),"processing_ms":1000*(time.perf_counter()-start),
                        "sampled_frames":len(frames),"window_end_ms":timestamps[-1]*1000,
                        "effective_fps":(len(timestamps)-1)/(timestamps[-1]-timestamps[0]),"dropped_windows":dropped}
                latest=(frames,result)
                last_completed=time.perf_counter()
                await send(result)
            except Exception as error: await send({"type":"error","message":str(error)})

        async def explain(snapshot,method,options):
            frames,result=snapshot
            try:
                async with work_lock:
                    start=time.perf_counter()
                    explanation=await run_in_threadpool(make_method(method).explain,model,frames,**options)
                data=explanation.to_dict()
                if method=="spatial":
                    thumbs=[]
                    for frame in frames:
                        image=Image.fromarray(frame); image.thumbnail((192,192)); stream=io.BytesIO()
                        image.save(stream,format="JPEG",quality=70)
                        thumbs.append(base64.b64encode(stream.getvalue()).decode())
                    data["thumbnails"]=thumbs
                await send({"type":"explanation","result":data,"processing_ms":1000*(time.perf_counter()-start),
                            "window_end_ms":result["window_end_ms"]})
            except Exception as error: await send({"type":"error","message":str(error)})

        try:
            await send({"type":"ready","message":"Camera frames are processed in memory; no recording."})
            while True:
                raw=await ws.receive_text()
                if len(raw)>MAX_MESSAGE: raise ValueError("Message exceeds size limit")
                message=json.loads(raw)
                if message.get("type")=="frame":
                    stamp=float(message["timestamp_ms"])/1000
                    frame=decode_frame(message["jpeg"]); captured+=1
                    accepted=buffer.add(stamp,frame); sampled+=int(accepted)
                    snapshot=buffer.snapshot() if accepted else None; now=time.perf_counter()
                    if snapshot is not None and now>=next_update:
                        if (task and not task.done()) or work_lock.locked(): dropped+=1
                        else:
                            task=asyncio.create_task(infer(snapshot)); next_update=now+update_seconds
                    if captured%5==0 or snapshot is None:
                        await send({"type":"buffer","count":len(buffer.items),"required":buffer.capacity,"sampled":sampled,
                                    "busy":bool(task and not task.done()),"dropped_windows":dropped})
                elif message.get("type")=="explain":
                    if latest is None or last_completed is None or time.perf_counter()-last_completed>max(3,update_seconds*3):
                        await send({"type":"error","message":"Collect a fresh prediction before explaining."}); continue
                    if (task and not task.done()) or work_lock.locked():
                        await send({"type":"error","message":"Inference is busy; try again shortly."}); continue
                    method=message["method"]; opts=options_for(method,message.get("options",{}))
                    task=asyncio.create_task(explain(latest,method,opts))
                elif message.get("type")=="reset":
                    if task and not task.done(): await task
                    buffer.clear(); latest=None; last_completed=None
                    await send({"type":"ready","message":"Temporal context reset."})
                else: raise ValueError("Unknown message type")
        except WebSocketDisconnect: pass
        except (ValueError,TypeError,KeyError,UnidentifiedImageError) as error:
            await send({"type":"error","message":str(error)}); await ws.close(code=1008)
        finally:
            # Do not release the model lock while its executor thread is still running.
            if task and not task.done():
                try: await task
                except (RuntimeError,WebSocketDisconnect): pass
            buffer.clear(); latest=None

    return app
