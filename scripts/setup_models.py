"""Install the hash-pinned official backbone; no dataset downloads or training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-workspace",type=Path,help="Existing directory containing movinet_evaluation/models/a0_stream")
    args=parser.parse_args()
    provenance=json.loads((ROOT/"models/kard/provenance.json").read_text())
    target=ROOT/"models/backbones/a0_stream"; var=target/"variables/variables.data-00000-of-00001"
    expected=provenance["backbone_data_sha256"]
    if var.is_file():
        if sha(var)!=expected: raise ValueError("Existing backbone does not match pinned weights; move it aside explicitly")
        if not (target/"saved_model.pb").is_file(): raise ValueError("Incomplete backbone")
        print("Verified existing backbone"); return
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
    print("Official backbone installed and verified. Trained head is already included in models/kard/.")


if __name__=="__main__": main()
