#!/usr/bin/env python3
"""Score any model's braille->print predictions on the BrailleLM test set.

Usage
  python score_llm.py my_predictions.tsv --name "GPT-5"
  python score_llm.py my_predictions.jsonl --name "Claude"
  python score_llm.py my_predictions.txt --name "local-7B" --order

Prediction formats
  .tsv / .csv    id<TAB>text  (or id,text); a single text column is also read
                 in file order when --order is given or no id column is detected
  .jsonl         {"id": 1, "prediction": "..."} or one JSON string per line
  .txt           one predicted sentence per line, in test-set order

The test set, references and BrailleLM reference scores are read from the
directory that contains this script. Only the Python standard library is
required; sacreBLEU is used if installed.
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_test():
    rows = read_jsonl(os.path.join(HERE, "test_set.jsonl"))
    rows.sort(key=lambda r: r["id"])
    return rows


def clean(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = re.sub(r"^(English|Translation|Output)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def load_predictions(path, rows):
    """Return {id: text} or a list in order."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        data = read_jsonl(path)
        if data and isinstance(data[0], dict):
            return {int(d.get("id", i + 1)): clean(str(d.get("prediction", d.get("text", ""))))
                    for i, d in enumerate(data)}
        return [clean(str(d)) for d in data]
    if ext in (".tsv", ".csv"):
        delim = "\t" if ext == ".tsv" else ","
        out = {}
        ordered = []
        with open(path, encoding="utf-8", newline="") as f:
            for i, parts in enumerate(csv.reader(f, delimiter=delim)):
                if not parts or all(not p.strip() for p in parts):
                    continue
                if i == 0 and parts[0].strip().lower() in ("id", "#", "index"):
                    continue
                if len(parts) >= 2 and re.fullmatch(r"\s*\d+\s*", parts[0]):
                    out[int(parts[0])] = clean(delim.join(parts[1:]))
                else:
                    ordered.append(clean(delim.join(parts)))
        return out if out else ordered
    return [clean(l) for l in open(path, encoding="utf-8") if l.strip()]


def align(preds, rows):
    ids = [r["id"] for r in rows]
    if isinstance(preds, dict):
        missing = [i for i in ids if i not in preds]
        extra = [i for i in preds if i not in ids]
        if missing:
            sys.exit(f"missing predictions for {len(missing)} ids, e.g. {missing[:5]}")
        if extra:
            print(f"[warn] ignoring {len(extra)} unknown ids", file=sys.stderr)
        return [preds[i] for i in ids]
    if len(preds) != len(rows):
        sys.exit(f"expected {len(rows)} predictions, got {len(preds)}")
    return preds


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[len(hyp)] / len(ref)


def norm_words(s):
    return [w.strip(".,").lower() for w in s.split()]


def score(refs, hyps):
    cers = [cer(r, h) for r, h in zip(refs, hyps)]
    freq = collections.Counter(hyps)
    entropy = -sum(c / len(hyps) * math.log2(c / len(hyps)) for c in freq.values())
    refv = set(w for r in refs for w in norm_words(r))
    outv = set(w for h in hyps for w in norm_words(h))
    pos_ok, pos_n = [0] * 10, [0] * 10
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        if len(rw) != 9:
            continue
        for i, (a, b) in enumerate(zip(rw, hw)):
            pos_n[i] += 1
            pos_ok[i] += (a == b)
    result = {
        "n": len(hyps),
        "exact": sum(r == h for r, h in zip(refs, hyps)),
        "exact_relaxed": sum(r.lower() == h.lower() for r, h in zip(refs, hyps)),
        "cer_mean": round(statistics.mean(cers), 4),
        "cer_median": round(statistics.median(cers), 4),
        "cer_min": round(min(cers), 4),
        "cer_max": round(max(cers), 4),
        "distinct_outputs": len(freq),
        "entropy_bits": round(entropy, 3),
        "largest_cluster": freq.most_common(1)[0][1],
        "vocab_emitted": len(outv),
        "vocab_reference": len(refv),
        "words_missing": len(refv - outv),
        "starts_the": sum(h.startswith("The") for h in hyps),
        "slot_accuracy": [round(pos_ok[i] / pos_n[i], 4) if pos_n[i] else None for i in range(9)],
        "slot_n": pos_n[0],
    }
    try:
        import sacrebleu
        result["sacrebleu"] = round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)
    except Exception:
        result["sacrebleu"] = None
    return result, cers


def row(label, r, extra=""):
    bleu = r.get("sacrebleu")
    bleu = "—" if bleu is None else f"{bleu:.1f}"
    print(f"{label:<22}{r['exact']:>3}/{r['n']:<4}"
          f"{r['cer_mean']*100:>7.1f}%{bleu:>7}"
          f"{r['distinct_outputs']:>7}{r['vocab_emitted']:>6}/{r['vocab_reference']:<4}"
          f"{r['slot_accuracy'][1]*100:>7.1f}%{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--name", default="your model")
    ap.add_argument("--out", default=None, help="write per-row CSV and JSON here")
    ap.add_argument("--order", action="store_true", help="force file-order alignment")
    args = ap.parse_args()

    rows = load_test()
    refs = [r["print"] for r in rows]
    preds = load_predictions(args.predictions, rows)
    if args.order and isinstance(preds, dict):
        preds = [preds[i] for i in sorted(preds)]
    hyps = align(preds, rows)

    result, cers = score(refs, hyps)
    result["model"] = args.name
    result["predictions_file"] = os.path.basename(args.predictions)

    ref_scores = json.load(open(os.path.join(HERE, "reference_scores.json"), encoding="utf-8"))
    print(f"\n{'model':<22}{'exact':>8}{'CER':>8}{'BLEU':>7}{'distinct':>9}{'vocab':>9}{'actor':>8}")
    print("-" * 73)
    row("BrailleLM native", ref_scores["native"])
    row("BrailleLM baseline", ref_scores["baseline"])
    row("Baseline 256 control", ref_scores["control"])
    row(args.name, result)
    print("\nactor = word accuracy in slot 2 of the 112 nine-word sentences")
    if result["sacrebleu"] is None:
        print("BLEU not computed (install sacrebleu for corpus BLEU).")

    slug = re.sub(r"[^a-z0-9]+", "_", args.name.lower()).strip("_") or "model"
    outdir = args.out or HERE
    os.makedirs(outdir, exist_ok=True)
    json.dump(result, open(os.path.join(outdir, f"scores_{slug}.json"), "w", encoding="utf-8"),
              indent=2)
    with open(os.path.join(outdir, f"per_row_{slug}.csv"), "w", encoding="utf-8") as f:
        f.write("id,cer,reference,prediction\n")
        for r, p, c in zip(rows, hyps, cers):
            f.write(f'{r["id"]},{c:.4f},"{r["print"]}","{p.replace(chr(34), chr(34)*2)}"\n')
    print(f"wrote {outdir}/scores_{slug}.json and per_row_{slug}.csv")


if __name__ == "__main__":
    main()
