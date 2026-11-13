"""
braille_tokenizer.py -- train the braille-native vocabulary.

Starts from the 64 six-dot braille cells and learns BPE merges over the braille
side of the training data, so frequent cell-sequences (contractions, common
words) collapse into single tokens. Saves the list of tokens to ADD to the base
model's tokenizer, each with its cell decomposition (used later for BKFT init).

Usage:
  python braille_tokenizer.py --data data --num_merges 150 --out braille_tokens.json
"""
import argparse, json, collections, sys

SPACE = chr(0x2800)
BLOCK = [chr(0x2800 + i) for i in range(64)]  # all 64 six-dot cells

def load_braille(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln)["braille"])
    return out

def train_bpe(texts, num_merges):
    seqs = [list(t) for t in texts]
    merges = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for s in seqs:
            for i in range(len(s) - 1):
                a, b = s[i], s[i+1]
                if a in (' ', '\n', SPACE) or b in (' ', '\n', SPACE):
                    continue  # don't merge across word boundaries
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        merges.append((a, b, cnt))
        merged = a + b
        new = []
        for s in seqs:
            ns, i = [], 0
            while i < len(s):
                if i < len(s)-1 and s[i] == a and s[i+1] == b:
                    ns.append(merged); i += 2
                else:
                    ns.append(s[i]); i += 1
            new.append(ns)
        seqs = new
    return merges

def main():
    # Braille output is Unicode; a cp1252 console on Windows would otherwise
    # raise UnicodeEncodeError after the token file has already been written,
    # which makes the runner treat a successful step as a failure.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--num_merges", type=int, default=150)
    ap.add_argument("--out", default="braille_tokens.json")
    args = ap.parse_args()

    texts = load_braille(f"{args.data}/train.jsonl")
    merges = train_bpe(texts, args.num_merges)

    # tokens to add = every base cell that appears + the indicators + merged tokens
    seen_cells = set()
    for t in texts:
        for ch in t:
            if ch != '\n':
                seen_cells.add(ch)
    base = [c for c in BLOCK if c in seen_cells]
    # ensure capital / number / space indicators are present even if BLOCK order missed
    for ind in [chr(0x2820), chr(0x283C), SPACE]:
        if ind in seen_cells and ind not in base:
            base.append(ind)

    merged_tokens = [a + b for (a, b, _) in merges]
    all_tokens = base + merged_tokens

    out = {
        "base_cells": base,
        "merges": [{"pair": a + b, "count": c} for (a, b, c) in merges],
        "tokens": all_tokens,
        "num_base": len(base),
        "num_merges": len(merges),
        "num_tokens": len(all_tokens),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[tok] base cells: {len(base)}, merges: {len(merges)}, total added tokens: {len(all_tokens)}")
    if merges:
        import braille_ueb as B
        print("[tok] first merges (cells -> print):")
        for m in merges[:6]:
            tok = m[0] + m[1]
            print(f"       {tok}  = \"{B.cell_to_print(tok)}\"  (x{m[2]})")
    print(f"[tok] wrote {args.out}")

if __name__ == "__main__":
    main()
