"""Build the Grade 1 + Grade 2 training and test data (BrailleLM v3).

Every sentence is stored once with both renderings:

    {"print": ..., "braille": <Grade 1>, "braille_g2": <Grade 2>}

Grade 2 comes from liblouis (en-ueb-g2.ctb), the reference open-source
translator. Grade 1 comes from braille_g1.to_braille and is cross-checked
against liblouis (en-ueb-g1.ctb) on every row; rows that differ are dropped
and counted.

Training text is the WikiText-103 TRAIN split (apostrophes rejoined) plus the
print side of legacy/neural_mac/data/train.jsonl.

Test files written next to train.jsonl, each with a `louis_g2_back` field
holding the liblouis rule-based reading of the Grade 2 braille (the baseline):

    validation, pilot_plain, pilot_extended   development (used for model selection)
    final_test                                locked set, same sentences as v1/v2
    digits_test, apostrophe_test              WikiText-103 TEST split

Leakage controls: no training sentence equals a held-out sentence
(case-insensitive); the build asserts this before writing.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from braillelm import augment, louis
from braillelm.braille_g1 import SUPPORTED, rule_decode, safe_for_synthesis, to_braille
from braillelm.paths import HELDOUT_DATA, REPO
from braillelm.pipelines.build_apostrophe_data import rejoin
from braillelm.pipelines.build_data import ALLOWED, SEED, clean

MAX_FRAMES = 8          # CTC frames per cell the Grade 2 model is trained with


def sentences(dataset, lo=20, hi=140):
    seen, out = set(), []
    for row in dataset:
        text = row["text"]
        if not text.strip() or text.lstrip().startswith("="):
            continue
        for piece in re.split(r"(?<=[.!?])\s+", rejoin(clean(text))):
            if (lo <= len(piece) <= hi and ALLOWED.fullmatch(piece) and len(piece.split()) >= 4
                    and " '" not in piece and "' " not in piece.replace("s' ", "")
                    and safe_for_synthesis(piece) and piece not in seen):
                seen.add(piece)
                out.append(piece)
    return out


def six_dot(b):
    return bool(b) and all(0x2800 <= ord(c) <= 0x283F for c in b)


def ctc_fits(text, cells, frames=MAX_FRAMES):
    """CTC needs one frame per character plus one per doubled letter."""
    need = len(text) + sum(a == b for a, b in zip(text, text[1:]))
    return need <= frames * cells


def render(prints, g1="strict", tag=None):
    """-> (rows, stats). Rows that cannot be used are dropped and counted.

    g1 decides what happens when this repository's Grade 1 differs from liblouis's
    (in practice: runs of capitalised words, where liblouis uses a passage indicator):
      "strict"  drop the sentence
      "mine"    keep this repository's form (what the demo produces for typed English)
      "both"    keep one row for each form, so the model learns to read either
    """
    g1_ref = louis.to_braille(prints, "g1")
    g2 = louis.to_braille(prints, "g2")
    rows, stats = [], {"g1_differs_from_liblouis": 0, "g1_not_reversible": 0, "g2_not_six_dot": 0,
                       "g2_too_dense_for_ctc": 0}
    for p, a, b in zip(prints, g1_ref, g2):
        mine = to_braille(p)
        differs = mine != a
        stats["g1_differs_from_liblouis"] += differs
        if rule_decode(mine) != p:
            stats["g1_not_reversible"] += 1
        elif differs and g1 == "strict":
            pass
        elif not six_dot(b):
            stats["g2_not_six_dot"] += 1
        elif not ctc_fits(p, len(b)):
            stats["g2_too_dense_for_ctc"] += 1
        else:
            forms = [mine] + ([a] if differs and g1 == "both" and six_dot(a) else [])
            for form in forms:
                row = {"print": p, "braille": form, "braille_g2": b}
                if tag:
                    row["kind"] = tag
                rows.append(row)
    return rows, stats


def vary(sents, kind, rng, blocked):
    """Apply one augment.KINDS transform; drop results that are unusable or held out."""
    out, seen = [], set()
    for s in sents:
        v = augment.KINDS[kind](s, rng)
        # safe_for_synthesis rejects runs of capitals because the two Grade 1 forms differ there;
        # render() handles that case explicitly, so all-capital text only needs the character check.
        usable = SUPPORTED.match(v) if kind == "all_caps" else safe_for_synthesis(v)
        if v and v != s and v not in seen and v.lower() not in blocked and usable:
            seen.add(v)
            out.append(v)
    return out


def with_baseline(rows):
    back = louis.to_print([r["braille_g2"] for r in rows], "g2")
    for r, b in zip(rows, back):
        r["louis_g2_back"] = b.replace("…", "...")
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800000, help="WikiText-103 training sentences to use")
    ap.add_argument("--n_aug", type=int, default=60000,
                    help="extra training sentences per kind of unusual text (see augment.py); 0 disables")
    ap.add_argument("--limit_articles", type=int, default=0, help="read only this many WikiText rows (smoke tests)")
    args = ap.parse_args()
    from datasets import load_dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "liblouis": louis.version(), "frames_per_cell": MAX_FRAMES}

    held = {name: read(HELDOUT_DATA / f"{name}.jsonl")
            for name in ["validation", "pilot_plain", "pilot_extended", "final_test"]}
    blocked = {r["print"].lower() for rows in held.values() for r in rows}

    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    test_split = wiki["test"]
    train_split = wiki["train"]
    if args.limit_articles:
        train_split = train_split.select(range(min(args.limit_articles, len(train_split))))

    # Extra test sets from the WikiText TEST split.
    pool = [s for s in sentences(test_split, 30, 120) if s.lower() not in blocked]
    random.Random(SEED).shuffle(pool)
    digits = [s for s in pool if any(c.isdigit() for c in s)][:400]
    taken = set(digits)
    apos = [s for s in pool if "'" in s and s not in taken][:500]
    extra = {"digits_test": digits, "apostrophe_test": apos}
    taken |= set(apos)
    blocked |= {s.lower() for v in extra.values() for s in v}

    # Unusual text: 80 test sentences per kind from the TEST split, plus hand-written everyday sentences.
    rest = [s for s in pool if s not in taken]
    rng = random.Random(SEED + 5)
    unusual = []
    for k, kind in enumerate(augment.KINDS):
        for v in vary(rest[k * 150:(k + 1) * 150], kind, rng, blocked)[:80]:
            unusual.append((kind, v))
    everyday = [line.strip() for line in open(REPO / "data" / "everyday_sentences.txt", encoding="utf-8")
                if line.strip()]
    blocked |= {v.lower() for _, v in unusual} | {s.lower() for s in everyday}

    # Training sentences.
    base = [r["print"] for r in read(HELDOUT_DATA / "train.jsonl")]
    cand = sentences(train_split)
    leaked = sum(c.lower() in blocked for c in cand)
    have = set(base)
    cand = [c for c in cand if c.lower() not in blocked and c not in have]
    random.Random(SEED).shuffle(cand)
    prints = [p for p in base if p.lower() not in blocked and ALLOWED.fullmatch(p) and safe_for_synthesis(p)]
    prints += cand[:args.n]
    rows, stats = render(prints)
    meta["augmented"] = {}
    rng = random.Random(SEED + 6)
    spare = cand[args.n:]
    for k, kind in enumerate(augment.KINDS if args.n_aug else []):
        source = spare[k * args.n_aug:(k + 1) * args.n_aug] or rng.sample(prints, min(len(prints), args.n_aug))
        extra_rows, st = render(vary(source, kind, rng, blocked), g1="both", tag=kind)
        rows += extra_rows
        meta["augmented"][kind] = {"rows": len(extra_rows), "dropped": st}
    random.Random(SEED + 2).shuffle(rows)
    train_set = {r["print"].lower() for r in rows}
    assert not (train_set & blocked), "training data overlaps a held-out set"
    meta.update({"train_rows": len(rows), "train_dropped": stats, "dropped_because_in_heldout": leaked,
                 "with_apostrophe": sum("'" in r["print"] for r in rows),
                 "with_digits": sum(any(c.isdigit() for c in r["print"]) for r in rows),
                 "g2_cells_per_g1_cell": round(sum(len(r["braille_g2"]) for r in rows)
                                               / sum(len(r["braille"]) for r in rows), 4),
                 "train_sha256": write(out / "train.jsonl", rows)})

    # Test files: legacy sets keep their published Grade 1 braille.
    meta["test_sets"] = {}
    for name, legacy in held.items():
        g2 = louis.to_braille([r["print"] for r in legacy], "g2")
        rs = [{"print": r["print"], "braille": r["braille"], "braille_g2": b}
              for r, b in zip(legacy, g2) if six_dot(b)]
        rs = with_baseline(rs)
        meta["test_sets"][name] = {"rows": len(rs), "dropped": len(legacy) - len(rs),
                                   "sha256": write(out / f"{name}.jsonl", rs)}
    for name, ps in extra.items():
        rs, st = render(ps)
        rs = with_baseline(rs)
        meta["test_sets"][name] = {"rows": len(rs), "dropped": st, "sha256": write(out / f"{name}.jsonl", rs)}
    rs = []
    for kind in augment.KINDS:
        part, _ = render([v for k, v in unusual if k == kind], g1="mine", tag=kind)
        rs += part
    meta["test_sets"]["unusual_test"] = {"rows": len(rs), "sha256": write(out / "unusual_test.jsonl", with_baseline(rs))}
    rs, st = render(everyday, g1="mine")
    meta["test_sets"]["everyday_test"] = {"rows": len(rs), "dropped": st,
                                          "sha256": write(out / "everyday_test.jsonl", with_baseline(rs))}
    for name, info in meta["test_sets"].items():
        rs = read(out / f"{name}.jsonl")
        info["liblouis_g2_baseline_exact"] = sum(r["louis_g2_back"] == r["print"] for r in rs)

    meta["heldout_overlap_with_train"] = 0
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
