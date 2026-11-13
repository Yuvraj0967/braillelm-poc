"""Build results/summary.json and demo/samples.json from saved evaluation files.

Every number is read from files on disk: v2 from results/metrics.json and the
v1 LLM from neural_mac/results (the committed Qwen predictions on the same
locked final set). Nothing is typed in by hand.
"""
import json
import re
from pathlib import Path

from braillelm.paths import HELDOUT_DATA, REPORTS, SERVING, V1_PREDICTIONS

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")      # same definition as neural_mac/code/audit_final.py


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    m = json.loads((REPORTS / "metrics.json").read_text())
    f = m["final_test"]
    v1_rows = read(V1_PREDICTIONS)
    v2_rows = read(REPORTS / "final_test_predictions.jsonl")
    assert [r["print"] for r in v1_rows] == [r["print"] for r in v2_rows]
    v1_exact = sum(r["prediction"] == r["print"] for r in v1_rows)
    n = f["clean"]["neural"]["n"]

    # The subgroup v1 reported: sentences with a word absent from the v1 training file.
    v1_vocab = {w.lower() for r in read(HELDOUT_DATA / "train.jsonl") for w in WORD.findall(r["print"])}
    sub = [i for i, r in enumerate(v1_rows) if {w.lower() for w in WORD.findall(r["print"])} - v1_vocab]
    v1_sub = sum(v1_rows[i]["prediction"] == v1_rows[i]["print"] for i in sub)
    v2_sub = sum(v2_rows[i]["prediction"] == v2_rows[i]["print"] for i in sub)

    noise = [{"label": "Clean braille", "mode": "faithful", "neural": f["clean"]["neural"]["exact_rate"],
              "rule": f["clean"]["rule_decoder"]["exact_rate"],
              "neural_cer": f["clean"]["neural"]["cer"], "rule_cer": f["clean"]["rule_decoder"]["cer"]},
             {"label": "Clean braille", "mode": "repair", "neural": f["clean"]["neural_repair_mode"]["exact_rate"],
              "rule": f["clean"]["rule_decoder"]["exact_rate"],
              "neural_cer": f["clean"]["neural_repair_mode"]["cer"], "rule_cer": f["clean"]["rule_decoder"]["cer"]}]
    for key in sorted(k for k in f if k.startswith("dot_errors_")):
        rate = float(key.split("_")[-1])
        noise.append({"label": f"{rate:.0%} of cells have one wrong dot", "mode": "repair",
                      "neural": f[key]["neural"]["exact_rate"], "rule": f[key]["rule_decoder"]["exact_rate"],
                      "neural_cer": f[key]["neural"]["cer"], "rule_cer": f[key]["rule_decoder"]["cer"],
                      "sentences_damaged": f[key]["inputs_actually_changed"]})
    summary = {
        "final": {"n": n, "v2_exact": f["clean"]["neural"]["exact"], "v2_rate": f["clean"]["neural"]["exact"] / n,
                  "v2_wilson95": f["clean"]["neural"]["wilson95"],
                  "v1_exact": v1_exact, "v1_rate": round(v1_exact / n, 4),
                  "unseen": {"n": len(sub), "exact": v2_sub, "rate": round(v2_sub / len(sub), 4),
                             "v1_exact": v1_sub, "v1_rate": round(v1_sub / len(sub), 4),
                             "definition": "sentences with a word absent from the v1 training file"}},
        "noise": noise,
        "params": m["checkpoint"]["params"],
        "ms_per_sentence": f["ms_per_sentence"],
        "scope": ("Converter-generated Grade 1 braille from WikiText test sentences that were never used for "
                  "training or model selection. Not yet validated on Grade 2 (contracted) braille or on "
                  "human-entered braille."),
    }
    if "has_unseen_word" in f["clean"]:
        u = f["clean"]["has_unseen_word"]
        summary["final"]["unseen_vs_v2_training"] = {"n": u["n"], "exact": u["exact"], "rate": u["exact_rate"]}
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # Demo samples: chosen by the v1 outcome and file order only, never by the v2 outcome.
    wrong = [i for i, r in enumerate(v1_rows) if r["prediction"] != r["print"] and len(r["print"]) <= 95]
    right = [i for i, r in enumerate(v1_rows) if r["prediction"] == r["print"] and len(r["print"]) <= 95]
    samples = []
    for i in wrong[:6] + right[:2]:
        r = v1_rows[i]
        samples.append({"label": " ".join(r["print"].split()[:3]) + " …", "braille": r["braille"],
                        "print": r["print"], "v1": r["prediction"], "row": i})
    (SERVING / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
