"""Retrieve private KARD artifacts and the hash-pinned official backbone."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from kinetic_lens.weights import ensure_artifacts


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-workspace",type=Path,help="Existing workspace containing the verified backbone and private-artifact backup")
    parser.add_argument("--group", action="append", choices=("default", "cuda", "research"), help="Repeat to fetch multiple private groups; default is default")
    parser.add_argument("--all", action="store_true", help="Fetch all three private artifact groups")
    parser.add_argument("--skip-backbone", action="store_true", help="Fetch private artifacts only, for example CUDA/research checkpoints")
    parser.add_argument("--verify-only", action="store_true", help="Check local hashes without authentication or network access")
    args=parser.parse_args()
    groups=("default","cuda","research") if args.all else tuple(args.group or ["default"])
    result=ensure_artifacts(ROOT,groups,from_workspace=args.from_workspace,verify_only=args.verify_only)
    print(json.dumps({"private_artifacts":result}))
    if args.skip_backbone:return
    provenance=json.loads((ROOT/"models/kard/provenance.json").read_text())
    target=ROOT/"models/backbones/a0_stream"; var=target/"variables/variables.data-00000-of-00001"
    expected=provenance["backbone_data_sha256"]
    if var.is_file():
        if sha(var)!=expected: raise ValueError("Existing backbone does not match pinned weights; move it aside explicitly")
        if not (target/"saved_model.pb").is_file(): raise ValueError("Incomplete backbone")
        print("Verified existing backbone"); return
    if args.verify_only:raise FileNotFoundError("Official backbone is absent; omit --verify-only to install it.")
    if args.from_workspace:
        source=args.from_workspace/"movinet_evaluation/models/a0_stream"
        if sha(source/"variables/variables.data-00000-of-00001")!=expected: raise ValueError("Source backbone hash mismatch")
        shutil.copytree(source,target,dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        archive=target.parent/"a0_stream.tar.gz"
        partial=archive.with_suffix(".partial")
        if not archive.exists():
            print("Downloading the official Google/TensorFlow A0-Stream backbone…",flush=True)
            with urllib.request.urlopen(provenance["backbone_handle"]+"?tf-hub-format=compressed",timeout=120) as response,partial.open("wb") as output:
                while chunk:=response.read(1024*1024): output.write(chunk)
            partial.replace(archive)
        if sha(archive)!=provenance["backbone_archive_sha256"]:
            raise ValueError("Downloaded archive differs from the audited model. No extraction performed.")
        target.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,"r:gz") as bundle: bundle.extractall(target,filter="data")
    if not (target/"saved_model.pb").is_file() or sha(var)!=expected: raise ValueError("Backbone installation verification failed")
    print("Official backbone installed and verified. Private trained artifacts are ready.")


if __name__=="__main__": main()
