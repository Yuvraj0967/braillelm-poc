"""Score a checkpoint.

Reports, for each test file and each grade the model reads ("g1", and "g2" when
the rows carry braille_g2):
  * strict exact match (character-for-character), CER, sacreBLEU on clean
    braille, read in faithful mode (the headline) and in repair mode
  * the same on braille with single-dot errors, read in repair mode, next to
    the rule-based baseline on the identical corrupted input. The baseline is
    the lookup-table decoder for Grade 1 and liblouis back-translation for
    Grade 2 (precomputed as louis_g2_back for clean input; computed here for
    damaged input when liblouis is installed).
  * exact match on sentences containing a word absent from the training set

Predictions for every row are written next to the metrics so each number can
be recomputed.
"""
import argparse
import json
import re
import time
from pathlib import Path

import sacrebleu
import torch

from braillelm import louis
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


KEYS = {"g1": "braille", "g2": "braille_g2"}


def baseline(grade, rows, inputs=None):
    """Rule-based reading. inputs=None means the clean braille of each row."""
    if grade == "g1":
        return [rule_decode(b) for b in (inputs or [r["braille"] for r in rows])]
    if inputs is None and all("louis_g2_back" in r for r in rows):
        return [r["louis_g2_back"] for r in rows]
    if not louis.available():
        return None
    return [t.replace("…", "...") for t in louis.to_print(inputs or [r["braille_g2"] for r in rows], "g2")]


def evaluate(model, rows, grade, noise, vocab):
    key = KEYS[grade]
    refs = [r["print"] for r in rows]
    brl = [r[key] for r in rows]
    t = time.time()
    pred = model.translate(brl, "faithful", grade)
    sec = time.time() - t
    hyps = [h for h, _ in pred]
    res = {"clean": {"neural": score(hyps, refs),
                     "neural_repair_mode": score([h for h, _ in model.translate(brl, "repair", grade)], refs)},
           "baseline": "lookup table" if grade == "g1" else "liblouis back-translation",
           "ms_per_sentence": round(1000 * sec / len(rows), 2),
           "cells": sum(map(len, brl))}
    base = baseline(grade, rows)
    if base:
        res["clean"]["rule_decoder"] = score(base, refs)
    if vocab:
        unseen = {i for i, r in enumerate(rows) if words(r["print"]) - vocab}
        for label, sub in (("has_unseen_word", sorted(unseen)),
                           ("all_words_seen", [i for i in range(len(rows)) if i not in unseen])):
            if sub:
                res["clean"][label] = score([hyps[i] for i in sub], [refs[i] for i in sub])
    preds = [{"print": r["print"], "braille": b, "prediction": h, "confidence": round(c, 4)}
             for r, b, (h, c) in zip(rows, brl, pred)]
    if base:
        for p, b in zip(preds, base):
            p["baseline"] = b
    for rate in noise:
        noisy = [corrupt_braille(b, rate, 7000 + i) for i, b in enumerate(brl)]
        nh = [h for h, _ in model.translate(noisy, "repair", grade)]
        block = {"neural": score(nh, refs),
                 "neural_faithful_mode": score([h for h, _ in model.translate(noisy, "faithful", grade)], refs),
                 "inputs_actually_changed": sum(a != b for a, b in zip(noisy, brl))}
        nb = baseline(grade, rows, noisy)
        if nb:
            block["rule_decoder"] = score(nb, refs)
        res[f"dot_errors_{rate}"] = block
        for p, b, h in zip(preds, noisy, nh):
            p[f"noisy_{rate}"] = {"braille": b, "prediction": h}
    return res, preds


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
    if torch.cuda.is_available():
        model.cuda()
    vocab = set()
    if args.train:
        for line in open(args.train, encoding="utf-8"):
            vocab |= words(json.loads(line)["print"])

    report = {"checkpoint": {k: v for k, v in extra.items() if k in ("step", "params") or k.startswith("dev_")},
              "model": model.cfg}
    for name in args.files:
        if name.endswith(".jsonl"):               # a path to a file outside --data
            rows, name = read(name), Path(name).stem
        else:
            rows = read(Path(args.data) / f"{name}.jsonl")
        report[name] = {}
        for grade in ("g1", "g2"):
            if grade == "g2" and (model.grades < 2 or "braille_g2" not in rows[0]):
                continue
            res, preds = evaluate(model, rows, grade, args.noise, vocab)
            report[name][grade] = res
            with open(out / f"{name}_{grade}_predictions.jsonl", "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    brief = {n: {g: {"clean": f"{r['clean']['neural']['exact']}/{r['clean']['neural']['n']}",
                     "baseline": r["clean"].get("rule_decoder", {}).get("exact")} for g, r in v.items()}
             for n, v in report.items() if n not in ("checkpoint", "model")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
