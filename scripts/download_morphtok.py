"""Put MorphTok hindi.csv into data/morphtok/ and record its provenance.

    python scripts/download_morphtok.py                       # use the local copy (default: ../hindi.csv)
    python scripts/download_morphtok.py --local PATH
    python scripts/download_morphtok.py --hf [--marathi]      # download from karthika95/morphTok

Phase 1 uses the user's local copy (user decision, 2026-11-02): no Hugging Face revision hash is
available for it, so its sha256 is recorded instead. Provenance -> results/phase1/morphtok_source.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "morphtok"
REPO = "karthika95/morphTok"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=str(ROOT.parent / "hindi.csv"))
    ap.add_argument("--hf", action="store_true", help="download from Hugging Face instead of the local copy")
    ap.add_argument("--marathi", action="store_true", help="also fetch marathi.csv (later phases; needs --hf)")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    info = {}
    if a.hf:
        from huggingface_hub import HfApi, hf_hub_download
        rev = HfApi().dataset_info(REPO).sha
        for fn in ["hindi.csv"] + (["marathi.csv"] if a.marathi else []):
            p = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=fn, revision=rev))
            shutil.copy2(p, DEST / fn)
            info[fn] = dict(source=f"hf:{REPO}", revision=rev, sha256=sha256(DEST / fn))
    else:
        src = Path(a.local)
        if not src.exists():
            raise SystemExit(f"{src} not found; pass --local PATH or --hf")
        if not (DEST / "hindi.csv").exists() or sha256(DEST / "hindi.csv") != sha256(src):
            shutil.copy2(src, DEST / "hindi.csv")
        info["hindi.csv"] = dict(source="local copy supplied by the user (originally MorphTok, karthika95/morphTok)",
                                 revision=None, sha256=sha256(DEST / "hindi.csv"))
    out = ROOT / "results" / "phase1" / "morphtok_source.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
