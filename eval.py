"""
eval.py -- evaluate a trained model on braille -> print.

Computes, on the held-out test set:
  * sacreBLEU  (higher better)
  * character error rate / CER (lower better)
  * tokens-per-cell: how many model tokens the braille input costs per braille
    cell under THIS model's tokenizer (the efficiency metric)

Run it for both the native and baseline checkpoints, then compare.

Usage:
  python eval.py --ckpt out_native   --data data --cpu
  python eval.py --ckpt out_baseline --data data --cpu
"""
import argparse, json, os
import torch

def read_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

PROMPT = "Translate the following braille to English text.\nBraille: {brl}\nEnglish:"

def cer(ref, hyp):
    # Levenshtein distance / len(ref)
    r, h = ref, hyp
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (r[i-1] != h[j-1]))
            prev = cur
    return dp[len(h)] / len(r)

def count_cells(brl):
    return sum(1 for ch in brl if 0x2800 <= ord(ch) <= 0x28FF)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="eval only first N (0 = all)")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    meta = json.load(open(f"{args.ckpt}/train_meta.json"))
    base_model = meta["base_model"]
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, args.ckpt)
    model.to(device).eval()

    rows = read_jsonl(f"{args.data}/test.jsonl")
    if args.limit:
        rows = rows[:args.limit]

    refs, hyps = [], []
    tok_per_cell = []
    for r in rows:
        brl, ref = r["braille"], r["print"]
        prompt = PROMPT.format(brl=brl)
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        out = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        hyp = out.split("\n")[0].strip()
        refs.append(ref); hyps.append(hyp)
        n_cells = count_cells(brl)
        n_tok = len(tok(brl, add_special_tokens=False)["input_ids"])
        if n_cells:
            tok_per_cell.append(n_tok / n_cells)

    # sacreBLEU
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    except Exception as e:
        bleu = None
        print(f"[eval] sacrebleu unavailable ({type(e).__name__}); skipping BLEU")

    cers = [cer(r, h) for r, h in zip(refs, hyps)]
    exact = sum(1 for r, h in zip(refs, hyps) if r.strip().lower() == h.strip().lower())
    result = {
        "ckpt": args.ckpt,
        "arm": meta.get("arm", meta.get("mode")),
        "seed": meta.get("seed"),
        "n": len(rows),
        "sacrebleu": round(bleu, 2) if bleu is not None else None,
        "cer": round(sum(cers)/len(cers), 4),
        "exact_match": round(exact/len(rows), 4),
        "tokens_per_cell": round(sum(tok_per_cell)/len(tok_per_cell), 3),
    }
    json.dump(result, open(f"{args.ckpt}/eval.json", "w"), indent=2)
    print("[eval]", json.dumps(result))
    print("[eval] sample:")
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"   ref: {r}")
        print(f"   hyp: {h}\n")

if __name__ == "__main__":
    main()
