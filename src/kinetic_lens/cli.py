import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from . import __version__
from .contracts import prediction
from .registry import load_model, make_method, methods_for
from .video import read_video


def main(argv=None):
    parser=argparse.ArgumentParser(description="KineticLens — HAR & XAI Studio")
    parser.add_argument("--config",default="configs/movinet_kard.json",help="Trusted local model/plugin configuration")
    sub=parser.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve"); serve.add_argument("--port",type=int,default=8765)
    serve.add_argument("--window-seconds",type=float,default=4.)
    serve.add_argument("--update-seconds",type=float,default=1.)
    sub.add_parser("methods")
    for name in ("predict","explain","benchmark"):
        cmd=sub.add_parser(name); cmd.add_argument("--video",required=True); cmd.add_argument("--output")
        if name=="explain":
            cmd.add_argument("--method",required=True); cmd.add_argument("--target",type=int); cmd.add_argument("--rival",type=int)
            cmd.add_argument("--options",default="{}",help="Method options as JSON; local CLI only")
        if name=="benchmark": cmd.add_argument("--repeats",type=int,default=10)
    args=parser.parse_args(argv); model=load_model(args.config)
    if args.command=="serve":
        from .server import create_app
        import uvicorn
        uvicorn.run(create_app(model,args.window_seconds,args.update_seconds),host="127.0.0.1",port=args.port,
                    ws_max_size=700_000,ws_max_queue=2,limit_concurrency=16)
        return
    if args.command=="methods": result={"model":model.provenance(),"methods":methods_for(model)}
    else:
        frames,meta=read_video(args.video,model.input_spec)
        start=time.perf_counter(); logits=model.predict(frames)
        result={"model":model.provenance(),"version":__version__,"input":meta,
                "video_sha256":hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
                "prediction":prediction(logits,model.labels),"cold_processing_ms":1000*(time.perf_counter()-start)}
        if args.command=="explain":
            options=json.loads(args.options)
            if args.target is not None: options["target"]=args.target
            if args.rival is not None: options["rival"]=args.rival
            result["explanation"]=make_method(args.method).explain(model,frames,**options).to_dict()
        if args.command=="benchmark":
            if not 2<=args.repeats<=1000: parser.error("repeats must be 2..1000")
            times=[]; error=0.
            for _ in range(args.repeats):
                start=time.perf_counter(); replay=model.predict(frames); times.append(1000*(time.perf_counter()-start))
                error=max(error,float(np.max(np.abs(replay-logits))))
            result["benchmark"]={"n":len(times),"median_ms":float(np.median(times)),"p95_ms":float(np.percentile(times,95)),
                                  "repeat_max_logit_error":error,"scope":"adapter preprocessing and model; excludes decoding, loading, browser and network"}
    text=json.dumps(result,indent=2,allow_nan=False)
    if getattr(args,"output",None):
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")
    print(text)


if __name__=="__main__": main()
