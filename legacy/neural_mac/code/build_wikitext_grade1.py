"""Build a locked, independently sourced Grade 1 evaluation from WikiText-2.

English is human-written WikiText. Braille is synthesized by liblouis 3.39.0;
this is not a corpus of human-entered braille. The test split is never used
for training or model selection.
"""

import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from datasets import load_dataset


SEED = 20261104
OUT = Path("work/wikitext_grade1")
ALLOWED = re.compile(r"[A-Za-z][A-Za-z ,.;:?!'\-]*[.!?]\Z")
PLAIN = re.compile(r"[A-Za-z][A-Za-z ]*\.\Z")


def clean(line):
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r"\s+([,.;:?!])", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def candidates(dataset):
    seen = set()
    out = []
    for line_no, row in enumerate(dataset):
        for piece in re.split(r"(?<=[.!?])\s+", clean(row["text"])):
            if (30 <= len(piece) <= 120 and ALLOWED.fullmatch(piece)
                    and len(piece.split()) >= 6 and " '" not in piece
                    and not re.search(r"\b[A-Z]\.$", piece)
                    and piece not in seen):
                out.append({"print": piece, "line": line_no})
                seen.add(piece)
    return out


def translate(lines, backward=False):
    cmd = ["lou_translate", "-d", "unicode.dis"]
    if backward:
        cmd.append("-b")
    cmd.append("en-ueb-g1.ctb")
    result = subprocess.run(cmd, input="\n".join(lines) + "\n", text=True,
                            capture_output=True, check=True)
    translated = result.stdout.splitlines()
    if len(translated) != len(lines):
        raise RuntimeError(f"liblouis returned {len(translated)} rows for {len(lines)} inputs")
    return translated


def write(name, rows, source_split):
    braille = translate([r["print"] for r in rows])
    roundtrip = translate(braille, backward=True)
    pairs = []
    for row, brl, back in zip(rows, braille, roundtrip):
        if row["print"].lower() == back.lower():
            pairs.append({"print": row["print"], "braille": brl,
                          "source": "Salesforce/wikitext:wikitext-2-raw-v1",
                          "wikitext_split": source_split, "source_line": row["line"]})
    dropped = len(rows) - len(pairs)
    path = OUT / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs), dropped, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    rng = random.Random(SEED)
    train = candidates(data["train"])
    validation = candidates(data["validation"])
    test = candidates(data["test"])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    train = train[:2000]
    validation = validation[:100]
    plain_test = [r for r in test if PLAIN.fullmatch(r["print"])][:200]
    plain_text = {r["print"] for r in plain_test}
    extended_test = [r for r in test if r["print"] not in plain_text][:200]
    pilot_text = plain_text | {r["print"] for r in extended_test}
    final_plain_test = [r for r in test if PLAIN.fullmatch(r["print"])
                        and r["print"] not in pilot_text][:200]
    final_extended_test = [r for r in test if not PLAIN.fullmatch(r["print"])
                           and r["print"] not in pilot_text][:200]
    all_text = [r["print"] for group in (train, validation, plain_test, extended_test,
                                         final_plain_test, final_extended_test)
                for r in group]
    if len(all_text) != len(set(all_text)):
        raise RuntimeError("duplicate sentences across splits")
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows, source in (("train", train, "train"), ("validation", validation, "validation"),
                               ("plain_test", plain_test, "test"),
                               ("extended_test", extended_test, "test"),
                               ("final_plain_test", final_plain_test, "test"),
                               ("final_extended_test", final_extended_test, "test")):
        n, dropped, digest = write(name, rows, source)
        stats[name] = {"n": n, "dropped_by_roundtrip": dropped, "sha256": digest}
    meta = {
        "dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
        "license": "CC BY-SA 4.0 per dataset card", "seed": SEED,
        "braille_engine": "liblouis en-ueb-g1.ctb + unicode.dis",
        "grade": 1, "rows": stats,
        "limits": "Human-written English, machine-transcribed braille; not real braille entry",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
