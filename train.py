"""
train.py -- fine-tune a small model for braille -> print.

Arms (set with --mode / --init):
  --mode baseline              : braille fed as raw Unicode through the model's
                                 ORIGINAL tokenizer (the control for "tokenizer").
  --mode native --init bkft    : add braille tokens; new embeddings initialised
                                 from the print each token maps to (BKFT).
  --mode native --init random  : SAME tokens, but new embeddings random-init.
                                 This is the control that isolates BKFT: compare
                                 native-bkft vs native-random to credit BKFT, and
                                 native vs baseline to credit the tokenizer.

Embeddings are FROZEN by default (only LoRA trains), which makes the quality of
the initialisation directly visible. Pass --train_embeddings to let the new rows
train too.

A validation split is held out from train to record a loss curve (loss_curve.json).

CPU: pass --cpu (float32). GPU: drop --cpu, optionally --bf16.
"""
import argparse, json, os, random
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

def build_example(tok, brl, eng, max_len):
    prompt = PROMPT.format(brl=brl)
    target = " " + eng + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    # Keep the answer; trim the PROMPT from the left if too long (truncating the
    # answer would leave an all-masked label row -> NaN loss).
    if len(t_ids) > max_len - 1:
        t_ids = t_ids[:max_len - 1]
    avail = max_len - len(t_ids)
    if len(p_ids) > avail:
        p_ids = p_ids[-avail:]
    input_ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return {"input_ids": input_ids, "labels": labels}

def bkft_init(model, tok, added_tokens, base_vocab_size):
    """Each new braille token embedding = mean of the base model's embeddings of
    the print string it maps to."""
    import braille_ueb as B
    emb = model.get_input_embeddings().weight.data
    n_mapped, n_fallback = 0, 0
    global_mean = emb[:base_vocab_size].mean(dim=0)
    with torch.no_grad():
        for t in added_tokens:
            tid = tok.convert_tokens_to_ids(t)
            if tid is None or tid < base_vocab_size:
                continue
            frag = B.cell_to_print(t)
            ids = tok(frag, add_special_tokens=False)["input_ids"] if frag else []
            ids = [i for i in ids if i < base_vocab_size]
            if ids:
                emb[tid] = emb[ids].mean(dim=0); n_mapped += 1
            else:
                emb[tid] = global_mean; n_fallback += 1
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != emb.data_ptr():
        out.weight.data[base_vocab_size:] = emb[base_vocab_size:]
    return n_mapped, n_fallback

def random_init(model, tok, added_tokens, base_vocab_size, std=0.02):
    """Control: random-init the new rows (no print information)."""
    emb = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for t in added_tokens:
            tid = tok.convert_tokens_to_ids(t)
            if tid is not None and tid >= base_vocab_size:
                emb[tid].normal_(0.0, std)
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != emb.data_ptr():
        out.weight.data[base_vocab_size:].normal_(0.0, std)

def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"]); pad = m - n
        input_ids.append(b["input_ids"] + [pad_id]*pad)
        attn.append([1]*n + [0]*pad)
        labels.append(b["labels"] + [-100]*pad)
    return {"input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels)}

def val_loss(model, val, pad_id, device, batch):
    if not val:
        return None
    model.eval(); tot, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), batch):
            b = collate(val[i:i+batch], pad_id)
            b = {k: v.to(device) for k, v in b.items()}
            tot += model(**b).loss.item(); nb += 1
    model.train()
    return tot / max(1, nb)

def load_model(model_name, dtype):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)       # transformers >=5
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)  # transformers 4.x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "baseline"], required=True)
    ap.add_argument("--init", choices=["bkft", "random"], default="bkft",
                    help="native only: how new braille embeddings are initialised")
    ap.add_argument("--train_embeddings", action="store_true",
                    help="let new embedding rows train (default: frozen)")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokens", default="braille_tokens.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arm = args.mode if args.mode == "baseline" else f"native-{args.init}"
    out = args.out or f"out_{arm}_seed{args.seed}"

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 2)
    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    print(f"[train] arm={arm} device={device} dtype={dtype} seed={args.seed} cuda={torch.cuda.is_available()}")

    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, dtype)
    base_vocab = len(tok)

    init_info = {}
    modules_to_save = None
    if args.mode == "native":
        added = json.load(open(args.tokens, encoding="utf-8"))["tokens"]
        n_added = tok.add_tokens(added)
        model.resize_token_embeddings(len(tok))
        if args.init == "bkft":
            nm, nf = bkft_init(model, tok, added, base_vocab)
            init_info = {"bkft_mapped": nm, "bkft_fallback": nf}
        else:
            random_init(model, tok, added, base_vocab)
        if args.train_embeddings:
            modules_to_save = ["embed_tokens", "lm_head"]
        print(f"[train] added {n_added} tokens ({args.init} init, "
              f"{'trainable' if args.train_embeddings else 'frozen'} embeddings); "
              f"vocab {base_vocab} -> {len(tok)}")

    lora = LoraConfig(r=args.lora_rank, lora_alpha=2*args.lora_rank, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
                      modules_to_save=modules_to_save)
    model = get_peft_model(model, lora)
    model.to(device)
    model.print_trainable_parameters()

    if device == "cuda":
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    rows = read_jsonl(f"{args.data}/train.jsonl")
    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    data = [build_example(tok, r["braille"], r["print"], args.max_len) for r in train_rows]
    val = [build_example(tok, r["braille"], r["print"], args.max_len) for r in val_rows]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    curve = []
    n_epochs = max(1, int(round(args.epochs)))
    steps = 0
    for ep in range(n_epochs):
        random.Random(args.seed + ep).shuffle(data)
        for i in range(0, len(data), args.batch):
            batch = collate(data[i:i+args.batch], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if ((i // args.batch) + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            steps += 1
            tl = loss.item() * args.grad_accum
            if steps % args.eval_every == 0:
                vl = val_loss(model, val, tok.pad_token_id, device, args.batch)
                curve.append({"step": steps, "train_loss": round(tl, 4),
                              "val_loss": round(vl, 4) if vl is not None else None})
                print(f"[train] step {steps} train {tl:.3f} val {vl:.3f}" if vl is not None
                      else f"[train] step {steps} train {tl:.3f}")
            elif steps % 10 == 0:
                print(f"[train] epoch {ep+1}/{n_epochs} step {steps} loss {tl:.3f}")
    opt.step(); opt.zero_grad()
    final_val = val_loss(model, val, tok.pad_token_id, device, args.batch)

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    json.dump({"step": steps, "train_loss": None,
               "val_loss": round(final_val, 4) if final_val is not None else None,
               "final": True}, open(f"{out}/_f", "w"))  # touch
    json.dump(curve + [{"step": steps, "val_loss": round(final_val, 4) if final_val else None, "final": True}],
              open(f"{out}/loss_curve.json", "w"), indent=2)
    meta = {"arm": arm, "mode": args.mode, "init": args.init if args.mode == "native" else None,
            "train_embeddings": bool(args.train_embeddings), "base_model": args.model,
            "device": device, "seed": args.seed, "steps": steps,
            "final_val_loss": round(final_val, 4) if final_val is not None else None,
            "n_train": len(train_rows), "n_val": len(val_rows), **init_info}
    json.dump(meta, open(f"{out}/train_meta.json", "w"), indent=2)
    os.remove(f"{out}/_f")
    print(f"[train] saved to {out}/  (final val loss {final_val})" if final_val else f"[train] saved to {out}/")

if __name__ == "__main__":
    main()
