"""Predict a video with the separately trained compact temporal KARD model."""
import argparse
from pathlib import Path
import json
import torch
from run_kard_compact_pipeline import FrozenSequences, MANIFEST, OUT, pack, sha
from kard_model import read_clip
from kard_compact_model import load_compact_head


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    selection=json.loads((OUT/'selection.json').read_text())
    assert sha(OUT/'compact_kard_head.pt')==selection['checkpoint_sha256']
    model,labels=load_compact_head(OUT/'compact_kard_head.pt',args.device)
    frozen=FrozenSequences()
    images,metadata=read_clip(args.video,MANIFEST['protocol'])
    sequence=frozen.full(images).numpy()[0]
    x,lengths=pack([sequence],args.device)
    with torch.no_grad():
        probabilities=model(x,lengths).softmax(-1)[0].cpu().numpy()
    frozen.verify_unchanged()
    print(json.dumps({'video':str(args.video),'metadata':metadata,'top3':[
        {'class':labels[i],'probability':float(probabilities[i])} for i in probabilities.argsort()[-3:][::-1]]},indent=2))


if __name__=='__main__':
    main()
