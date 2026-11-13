"""Score a v2 checkpoint.

Reports, for each test file:
  * strict exact match (character-for-character), CER, sacreBLEU on clean
    braille, read in faithful mode (the headline) and in repair mode
  * the same on braille with single-dot errors, read in repair mode, next to
    the deterministic rule decoder on the identical corrupted input
  * exact match on sentences containing a word absent from the training set

Predictions for every row are written next to the metrics so each number can
be recomputed.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import sacrebleu

from braillelm.braille_g1 import rule_decode
from braillelm.model import load
from braillelm.noise import corrupt_braille


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wilson(k, n, z=1.96):
    p = k / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    d = 1 + z * z / n
    return round((c - h) / d, 4), round((c + h) / d, 4)


def score(hyps, refs):
    k = sum(h == r for h, r in zip(hyps, refs))
    return {"exact": k, "n": len(refs), "exact_rate": round(k / len(refs), 4),
            "wilson95": wilson(k, len(refs)),
            "cer": round(sum(lev(r, h) / len(r) for h, r in zip(hyps, refs)) / len(refs), 5),
            "sacrebleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)}


def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", nargs="*", type=float, default=[0.02, 0.05])
    ap.add_argument("--train", default="", help="train.jsonl, for the unseen-word subgroup")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, extra = load(args.ckpt)
    vocab = set()
    if args.train:
        for line in open(args.train, encoding="utf-8"):
            vocab |= words(json.loads(line)["print"])

    report = {"checkpoint": {k: extra.get(k) for k in ("step", "params", "dev_exact_clean", "dev_exact_noise3",
                                                       "dev_exact_clean_repair_mode")}}
    for name in args.files:
        if name.endswith(".jsonl"):               # a path to a file outside --data
            rows, name = read(name), Path(name).stem
        else:
            rows = read(Path(args.data) / f"{name}.jsonl")
        refs = [r["print"] for r in rows]
        t = time.time()
        pred = model.translate([r["braille"] for r in rows], "faithful")
        sec = time.time() - t
        hyps = [h for h, _ in pred]
        rep_clean = [h for h, _ in model.translate([r["braille"] for r in rows], "repair")]
        res = {"clean": {"neural": score(hyps, refs),
                         "neural_repair_mode": score(rep_clean, refs),
                         "rule_decoder": score([rule_decode(r["braille"]) for r in rows], refs)},
               "ms_per_sentence": round(1000 * sec / len(rows), 2)}
        if vocab:
            unseen = [i for i, r in enumerate(rows) if words(r["print"]) - vocab]
            seen = [i for i in range(len(rows)) if i not in set(unseen)]
            for label, sub in (("has_unseen_word", unseen), ("all_words_seen", seen)):
                if sub:
                    res["clean"][label] = score([hyps[i] for i in sub], [refs[i] for i in sub])
        preds = [{"print": r["print"], "braille": r["braille"], "prediction": h, "confidence": round(c, 4)}
                 for r, (h, c) in zip(rows, pred)]
        for rate in args.noise:
            noisy = [corrupt_braille(r["braille"], rate, 7000 + i) for i, r in enumerate(rows)]
            nh = [h for h, _ in model.translate(noisy, "repair")]
            res[f"dot_errors_{rate}"] = {
                "neural": score(nh, refs),
                "neural_faithful_mode": score([h for h, _ in model.translate(noisy, "faithful")], refs),
                "rule_decoder": score([rule_decode(b) for b in noisy], refs),
                "inputs_actually_changed": sum(b != r["braille"] for b, r in zip(noisy, rows))}
            for p, b, h in zip(preds, noisy, nh):
                p[f"noisy_{rate}"] = {"braille": b, "prediction": h}
        report[name] = res
        with open(out / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
