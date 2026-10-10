"""Copy interpreted results, not dataset frames/caches, from the previous studies."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,required=True);args=p.parse_args()
    base=args.workspace/"movinet_evaluation";target=ROOT/"docs/research";target.mkdir(parents=True,exist_ok=True)
    def clean(s):
        for prefix in [str(args.workspace),str(args.workspace).replace("\\","/")]:s=s.replace(prefix,"<local-workspace>")
        s=re.sub(r"C:[/\\]Users[/\\]Xerox[/\\]miniconda ?3[/\\]envs[/\\]xai[/\\]python\.exe","python",s)
        return s
    suite=base/"kard_xai_suite_extended"
    for name in ("INTERPRETATION.md","SEMANTIC_VALIDATION_NEXT.md","interpretation_article.tex","parent_numerical_audit.json"):
        text=clean((suite/name).read_text(encoding="utf-8"))
        for method in ("temporal","concept","contrastive","mechanistic"):
            text=text.replace(f"../kard_{method}_xai_extended/",f"{method}/")
        (target/name).write_text(text,encoding="utf-8")
    for method in ("temporal","concept","contrastive","mechanistic"):
        dest=target/method;dest.mkdir(exist_ok=True)
        for name in ("REPORT.md","INTERPRETATION.md"):
            (dest/name).write_text(clean((base/f"kard_{method}_xai_extended"/name).read_text(encoding="utf-8")),encoding="utf-8")
    spatial=target/"spatial";spatial.mkdir(exist_ok=True)
    (spatial/"REPORT.md").write_text(clean((base/"kard_spatial_xai/REPORT.md").read_text(encoding="utf-8")),encoding="utf-8")
    figures=target/"figures";figures.mkdir(exist_ok=True)
    numeric_figures=[suite/"xai_evidence_map.svg",base/"kard_temporal_xai_extended/fig01_temporal_controls.svg",
        base/"kard_concept_xai_extended/matched_deletion_controls.svg",base/"kard_contrastive_xai_extended/boundary_and_reference.svg",
        base/"kard_mechanistic_xai_extended/retention_reference_curves.svg"]
    for path in numeric_figures:shutil.copy2(path,figures/path.name)
    print("Imported interpreted reports and numeric vector figures; no raw videos, frame images or caches copied.")


if __name__=="__main__":main()
