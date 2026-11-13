"""Add apostrophes to the training data.

build_data.py dropped every sentence containing " '" and WikiText writes
"Butler 's" and "didn 't", so the first training set had no apostrophes at all
and the model misread them. This script rejoins those forms, converts the
sentences, and writes a fine-tuning set:

  <out>/train.jsonl        apostrophe sentences + a random sample of the original rows
  <out>/apostrophe_test.jsonl  500 apostrophe sentences held out from the above
  (validation and pilot files are copied so train.py can score them)

Only the WikiText-103 TRAIN split is read; sentences equal to any held-out row
are dropped, as in build_data.py.
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

from datasets import load_dataset

from braillelm.braille_g1 import to_braille, rule_decode, safe_for_synthesis
from braillelm.pipelines.build_data import ALLOWED, SEED, clean

JOIN = re.compile(r"(?<=[A-Za-z]) (n?'(?:s|t|re|ve|ll|d|m))\b|(?<=s) '(?= )")
PREFIX = re.compile(r"\b([ODdLl]) '(?=[A-Z])")


def rejoin(line):
    line = JOIN.sub(lambda m: m.group(1) or "'", line)
    return PREFIX.sub(r"\1'", line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="directory written by build_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_apostrophe", type=int, default=60000)
    ap.add_argument("--n_original", type=int, default=120000)
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blocked = set()
    for name in ["validation", "pilot_plain", "pilot_extended", "final_test", "digits_test"]:
        for l in open(base / f"{name}.jsonl", encoding="utf-8"):
            blocked.add(json.loads(l)["print"].lower())

    seen, cand = set(), []
    for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]:
        text = row["text"]
        if " '" not in text or text.lstrip().startswith("="):
            continue
        for piece in re.split(r"(?<=[.!?])\s+", rejoin(clean(text))):
            if ("'" in piece and " '" not in piece and "' " not in piece.replace("s' ", "")
                    and 20 <= len(piece) <= 140 and ALLOWED.fullmatch(piece)
                    and len(piece.split()) >= 4 and safe_for_synthesis(piece)
                    and piece not in seen and piece.lower() not in blocked):
                seen.add(piece)
                cand.append(piece)
    rng = random.Random(SEED + 1)
    rng.shuffle(cand)
    cand = cand[:args.n_apostrophe + 500]
    rows = []
    for c in cand:
        b = to_braille(c)
        assert rule_decode(b) == c, c
        rows.append({"print": c, "braille": b, "source": "wikitext-103-raw-v1/train (apostrophes rejoined)"})
    dev, apos = rows[:500], rows[500:]

    original = [l for l in open(base / "train.jsonl", encoding="utf-8")]
    rng.shuffle(original)
    mixed = [json.dumps(r, ensure_ascii=False) + "\n" for r in apos] + original[:args.n_original]
    rng.shuffle(mixed)
    (out / "train.jsonl").write_text("".join(mixed), encoding="utf-8")
    with open(out / "apostrophe_test.jsonl", "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ["validation", "pilot_plain", "pilot_extended"]:
        shutil.copy(base / f"{name}.jsonl", out / f"{name}.jsonl")
    print(json.dumps({"apostrophe_rows": len(apos), "apostrophe_test": len(dev),
                      "original_rows": min(len(original), args.n_original), "examples": [r["print"] for r in apos[:6]]},
                     indent=1))


if __name__ == "__main__":
    main()
