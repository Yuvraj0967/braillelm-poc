"""Build the v2 training set.

Sources
  * legacy/neural_mac/data/train.jsonl   (15,886 liblouis-generated pairs, unchanged)
  * WikiText-103 raw, TRAIN split only, converted with braille_g1.to_braille
    (verified identical to liblouis on every published row by check_converter.py)

Leakage controls
  * WikiText-103 validation/test articles are the same articles as WikiText-2
    validation/test, and neither is touched here.
  * Any sentence equal (case-insensitively) to a row of validation, either
    pilot, or the locked final set is dropped anyway, and the count is reported.

Also writes digits_test.jsonl: sentences with whole numbers from the WikiText
TEST split, disjoint from every other file, to measure numeric mode (which the
original evaluation sets do not contain).
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

from braillelm.braille_g1 import to_braille, rule_decode, safe_for_synthesis

from braillelm.paths import HELDOUT_DATA

SEED = 20261112
ALLOWED = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.;:?!'\-]*[.!?]\Z")


def clean(line):
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r"\s+([,.;:?!])", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def candidates(dataset, lo=20, hi=140):
    seen, out = set(), []
    for row in dataset:
        text = row["text"]
        if not text.strip() or text.lstrip().startswith("="):
            continue
        for piece in re.split(r"(?<=[.!?])\s+", clean(text)):
            if (lo <= len(piece) <= hi and ALLOWED.fullmatch(piece)
                    and len(piece.split()) >= 4 and " '" not in piece
                    and safe_for_synthesis(piece) and piece not in seen):
                seen.add(piece)
                out.append(piece)
    return out


def read(name):
    return [json.loads(l) for l in open(HELDOUT_DATA / f"{name}.jsonl", encoding="utf-8")]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_wiki103", type=int, default=400000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    held = {}
    for name in ["validation", "pilot_plain", "pilot_extended", "final_test"]:
        held[name] = read(name)
    blocked = {r["print"].lower() for rows in held.values() for r in rows}

    base = read("train")
    assert not any(r["print"].lower() in blocked for r in base)

    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    cand = candidates(wiki["train"])
    have = {r["print"] for r in base}
    leaked = sum(c.lower() in blocked for c in cand)
    cand = [c for c in cand if c.lower() not in blocked and c not in have]
    random.Random(SEED).shuffle(cand)
    cand = cand[:args.n_wiki103]

    rows = [{"print": r["print"], "braille": r["braille"], "source": "neural_mac/train"} for r in base]
    for c in cand:
        b = to_braille(c)
        assert rule_decode(b) == c, c
        rows.append({"print": c, "braille": b, "source": "wikitext-103-raw-v1/train"})
    meta = {"seed": SEED, "base_rows": len(base), "wikitext103_rows": len(cand),
            "train_rows": len(rows), "dropped_because_in_heldout": leaked,
            "with_digits": sum(any(ch.isdigit() for ch in r["print"]) for r in rows),
            "train_sha256": write(out / "train.jsonl", rows)}

    # Numeric-mode test: WikiText test split, sentences containing digits.
    train_set = {r["print"].lower() for r in rows}
    digits = [c for c in candidates(wiki["test"], 30, 120)
              if any(ch.isdigit() for ch in c) and c.lower() not in blocked
              and c.lower() not in train_set]
    random.Random(SEED).shuffle(digits)
    digits = [{"print": c, "braille": to_braille(c), "source": "wikitext-103-raw-v1/test"} for c in digits[:400]]
    meta["digits_test_rows"] = len(digits)
    meta["digits_test_sha256"] = write(out / "digits_test.jsonl", digits)

    for name, rs in held.items():
        assert not any(r["print"].lower() in train_set for r in rs), name
        write(out / f"{name}.jsonl", rs)
    meta["heldout_overlap_with_train"] = 0
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
