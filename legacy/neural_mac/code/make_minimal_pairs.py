"""Generate train-only word substitutions to force exact braille conditioning.

Uses words from the WikiText train split; does not read validation or test data.
Braille labels are generated and round-trip checked with pinned liblouis tables.
"""
import collections
import json
from pathlib import Path
import random
import re
import subprocess

SEED = 20261104
WORD = re.compile(r"\b[A-Za-z]{4,10}\b")
STOP = set("that this with from they were have been into about there their would could which where while after before during through these those some many much most each more than then when what here only also them over very ever even your ours mine ours itself itself".split())


def translate(lines, backward=False):
    cmd = ["lou_translate", "-d", "unicode.dis"]
    if backward:
        cmd.append("-b")
    cmd.append("en-ueb-g1.ctb")
    result = subprocess.run(cmd, input="\n".join(lines) + "\n", text=True,
                            capture_output=True, check=True)
    out = result.stdout.splitlines()
    if len(out) != len(lines):
        raise RuntimeError(f"liblouis output count {len(out)} != {len(lines)}")
    return out


def main():
    source = [json.loads(x) for x in Path("data_all/train.jsonl").open()]
    counts = collections.Counter(w.lower() for r in source for w in WORD.findall(r["print"]))
    pool = collections.defaultdict(list)
    for w, n in counts.items():
        if n >= 3 and w not in STOP:
            pool[len(w)].append(w)
    rng = random.Random(SEED)
    seen = {r["print"] for r in source}
    variants = []
    for row in source:
        text = row["print"]
        positions = [m for m in WORD.finditer(text) if m.group().lower() not in STOP]
        if not positions:
            continue
        for _ in range(2):
            match = rng.choice(positions)
            old = match.group()
            choices = pool[len(old)]
            if not choices:
                continue
            replacement = rng.choice(choices)
            if replacement == old.lower():
                continue
            if old[0].isupper():
                replacement = replacement.capitalize()
            changed = text[:match.start()] + replacement + text[match.end():]
            if changed not in seen:
                variants.append(changed)
                seen.add(changed)
    braille = translate(variants)
    roundtrip = translate(braille, backward=True)
    out = Path("data_all/minimal_pairs.jsonl")
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for text, b, back in zip(variants, braille, roundtrip):
            if text.lower() != back.lower():
                continue
            f.write(json.dumps({"print": text, "braille": b,
                                "source": "train-only word substitution + liblouis UEB G1"},
                               ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"source_rows": len(source), "variants": len(variants),
                      "roundtrip_kept": kept, "file": str(out)}, indent=2))


if __name__ == "__main__":
    main()
