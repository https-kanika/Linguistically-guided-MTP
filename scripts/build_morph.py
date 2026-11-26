"""Morphology (local, CPU): inspect MorphTok, build the gold lexicon, train + evaluate the CRF.

    python scripts/build_morph.py

Writes results/phase1/morphtok_info.json, morph_crf.pkl, morph_crf_eval.json.
Steps 5-6 (coverage on target outputs, prefix model from dev) need the traces and run in step S4.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgmtp.morph import GoldLexicon, load_morphtok, train_and_eval_crf  # noqa: E402

RES = ROOT / "results" / "phase1"


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    df, info = load_morphtok(ROOT / "data" / "morphtok" / "hindi.csv")
    print("columns:", list(df.columns[:3]), "| dtypes:", dict(df.dtypes.astype(str).iloc[:3]))
    print("rows:", info["rows"], "| kept:", info["kept_rows"], "| dropped (sandhi):", info["dropped_rows"])
    print(df.head(10).to_string())
    print("dropped examples:", *info["dropped_examples"], sep="\n  ")

    lex = GoldLexicon.from_dataframe(df)
    info["lexicon"] = lex.stats
    src = RES / "morphtok_source.json"
    if src.exists():
        info["source"] = json.loads(src.read_text(encoding="utf-8"))
    (RES / "morphtok_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print("lexicon:", lex.stats)

    seg, ev = train_and_eval_crf(lex, seed=0)
    with open(RES / "morph_crf.pkl", "wb") as fh:
        pickle.dump(seg, fh, protocol=4)
    (RES / "morph_crf_eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(json.dumps(ev, indent=1))


if __name__ == "__main__":
    main()
