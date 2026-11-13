"""Independently audit saved final predictions without running model inference.

Usage: python code/audit_final.py --root .
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import sacrebleu


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def wilson_interval(successes, count, z=1.959963984540054):
    if not count:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [round(center - radius, 4), round(center + radius, 4)]


def scores(rows):
    refs = [row["print"] for row in rows]
    hyps = [row["prediction"] for row in rows]
    count = len(rows)
    exact = sum(ref == hyp for ref, hyp in zip(refs, hyps))
    normalized = sum(ref.strip().lower() == hyp.strip().lower()
                     for ref, hyp in zip(refs, hyps))
    return {
        "n": count,
        "strict_exact_count": exact,
        "strict_exact_rate": round(exact / count, 4) if count else None,
        "strict_exact_wilson_95": wilson_interval(exact, count),
        "normalized_exact_count": normalized,
        "mean_cer": round(sum(edit_distance(ref, hyp) / len(ref) if ref else
                              float(bool(hyp)) for ref, hyp in zip(refs, hyps)) / count, 4)
        if count else None,
        "sacrebleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)
        if count else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    test_path = root / "data/final_test.jsonl"
    predictions_path = root / "results/eval_qwen_final_locked_beam4_predictions.jsonl"
    metrics_path = root / "results/eval_qwen_final_locked_beam4.json"
    train_path = root / "data/train.jsonl"

    assert sha256(test_path) == "74726e62a8e31ca396fd2b02a61230368c0f15d5963360a0deb91c85a3b48c91"
    test = read_jsonl(test_path)
    predictions = read_jsonl(predictions_path)
    assert len(test) == len(predictions) == 399
    for index, (reference, hypothesis) in enumerate(zip(test, predictions), 1):
        assert reference["print"] == hypothesis["print"], index
        assert reference["braille"] == hypothesis["braille"], index
        assert isinstance(hypothesis["prediction"], str), index

    train_words = {word.casefold() for row in read_jsonl(train_path)
                   for word in WORD.findall(row["print"])}
    unseen_flags = [any(word.casefold() not in train_words
                        for word in WORD.findall(row["print"]))
                    for row in predictions]
    unseen = [row for row, flag in zip(predictions, unseen_flags) if flag]
    seen = [row for row, flag in zip(predictions, unseen_flags) if not flag]

    whole = scores(predictions)
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    for field, audited in (("n", whole["n"]),
                           ("exact_match", whole["strict_exact_rate"]),
                           ("normalized_exact_match", round(whole["normalized_exact_count"] / 399, 4)),
                           ("cer", whole["mean_cer"]),
                           ("sacrebleu", whole["sacrebleu"])):
        assert reported[field] == audited, (field, reported[field], audited)
    assert reported["num_beams"] == 4

    audit = {
        "test_sha256": sha256(test_path),
        "predictions_sha256": sha256(predictions_path),
        "checkpoint": reported["ckpt"],
        "beam_width": reported["num_beams"],
        "all_399": whole,
        "plain_first_200": scores(predictions[:200]),
        "extended_last_199": scores(predictions[200:]),
        "contains_training_unseen_word": scores(unseen),
        "all_words_in_training_vocabulary": scores(seen),
        "labels_match_test_file_in_order": True,
        "saved_metrics_recomputed": True,
    }
    out = root / "results/final_independent_audit.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
