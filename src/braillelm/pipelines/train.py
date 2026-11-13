"""Train the BrailleLM v2 cell-level reader with CTC.

Model selection uses only the development rows (validation + both pilots).
The locked final set is never read by this script.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from braillelm.model import (BrailleReader, PAD_IN, FRAMES_PER_CELL, encode_cells, encode_print, load, save)
from braillelm import tracking
from braillelm.noise import corrupt_ids, corrupt_braille


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def batches(lengths, cell_budget, rng):
    order = list(range(len(lengths)))
    rng.shuffle(order)
    pool = 4096
    out = []
    for s in range(0, len(order), pool):
        chunk = sorted(order[s:s + pool], key=lambda i: lengths[i])
        cur, longest = [], 0
        for i in chunk:
            longest = max(longest, lengths[i])
            if cur and longest * (len(cur) + 1) > cell_budget:
                out.append(cur)
                cur, longest = [], lengths[i]
            cur.append(i)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def dev_scores(model, dev, noise_rate):
    """(faithful on clean, repair on noisy, repair on clean) strict exact."""
    brl = [r["braille"] for r in dev]
    noisy = [corrupt_braille(b, noise_rate, 1000 + i) for i, b in enumerate(brl)]
    n = len(dev)

    def exact(inputs, mode):
        return sum(h == r["print"] for (h, _), r in zip(model.translate(inputs, mode), dev)) / n
    return exact(brl, "faithful"), exact(noisy, "repair"), exact(brl, "repair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML file of defaults for the options below (see configs/)")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--cell_budget", type=int, default=8000, help="max padded cells per batch")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--p_noisy", type=float, default=0.6, help="share of training rows shown in repair mode")
    ap.add_argument("--max_noise", type=float, default=0.08)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--mlflow", metavar="EXPERIMENT", help="log this run to MLflow under this experiment name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="checkpoint to continue from (weights only; the optimiser restarts)")
    ap.add_argument("--start_step", type=int, default=0, help="step the resumed checkpoint was saved at")
    args = ap.parse_args()
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        unknown = set(cfg) - set(vars(args))
        if unknown:
            ap.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        ap.set_defaults(**cfg)
        args = ap.parse_args()          # command-line flags still override the file
    if not args.data or not args.out:
        ap.error("--data and --out are required (on the command line or in --config)")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read(data / "train.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    X = [np.asarray(encode_cells(r["braille"]), dtype=np.uint8) for r in rows]
    Y = [np.asarray(encode_print(r["print"]), dtype=np.uint8) for r in rows]
    lengths = [len(x) for x in X]
    del rows
    dev = read(data / "validation.jsonl") + read(data / "pilot_plain.jsonl") + read(data / "pilot_extended.jsonl")

    resumed = None
    if args.resume:
        model, resumed = load(args.resume)
    else:
        model = BrailleReader(args.d, args.layers, args.heads, args.ff, args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train rows {len(X)}  dev rows {len(dev)}  params {n_params/1e6:.2f}M  threads {torch.get_num_threads()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        ramp = min(1.0, (step - args.start_step + 1) / 50) if args.resume else 1.0   # let Adam statistics rebuild
        return ramp * args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    log, best, step, epoch, t0, run_loss = [], -1.0, args.start_step, 0, time.time(), None
    if resumed:
        best = resumed.get('dev_exact_clean', 0) + resumed.get('dev_exact_noise3', 0)
    with tracking.run(args.mlflow, run_name=out.name) as tracker:
        tracker.params(vars(args))
        while step < args.steps:
            for idx in batches(lengths, args.cell_budget, rng):
                if step >= args.steps:
                    break
                xs, modes = [], []
                for i in idx:
                    ids = X[i].tolist()
                    if rng.random() < args.p_noisy:       # repair mode: input may be damaged
                        if rng.random() < 0.85:
                            ids = corrupt_ids(ids, rng.uniform(0.005, args.max_noise), rng)
                        modes.append(1)
                    else:                                 # faithful mode: input is always clean
                        modes.append(0)
                    xs.append(ids)
                T = max(map(len, xs))
                x = torch.full((len(xs), T), PAD_IN, dtype=torch.long)
                for r, q in enumerate(xs):
                    x[r, :len(q)] = torch.tensor(q)
                tgt = torch.cat([torch.from_numpy(Y[i].astype(np.int64)) for i in idx])
                in_len = torch.tensor([len(q) * FRAMES_PER_CELL for q in xs])
                tgt_len = torch.tensor([len(Y[i]) for i in idx])

                model.train()
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                logp = model(x, torch.tensor(modes)).log_softmax(-1).transpose(0, 1)
                loss = F.ctc_loss(logp, tgt, in_len, tgt_len, blank=0, zero_infinity=True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()

                if step % 50 == 0:
                    el = time.time() - t0
                    print(f"step {step:6d}/{args.steps} epoch {epoch} loss {run_loss:.4f} lr {lr_at(step):.2e} "
                          f"{el/(step-args.start_step):.2f}s/step eta {(args.steps-step)*el/(step-args.start_step)/60:.0f}m", flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    clean, noisy, repair_clean = dev_scores(model, dev, 0.03)
                    rec = {"step": step, "epoch": epoch, "train_loss": round(run_loss, 5),
                           "dev_exact_clean": round(clean, 4), "dev_exact_noise3": round(noisy, 4),
                           "dev_exact_clean_repair_mode": round(repair_clean, 4),
                           "minutes": round((time.time() - t0) / 60, 1)}
                    log.append(rec)
                    print("EVAL", json.dumps(rec), flush=True)
                    tracker.metrics(rec, step=step)
                    extra = {"args": vars(args), "params": n_params, **rec}
                    save(model, out / "last.pt", extra)
                    if clean + noisy > best:
                        best = clean + noisy
                        save(model, out / "best.pt", extra)
                    (out / "train_log.json").write_text(json.dumps(log, indent=1))
            epoch += 1
        for name in ("best.pt", "train_log.json"):
            if (out / name).exists():
                tracker.artifact(out / name)
    print("done", flush=True)


if __name__ == "__main__":
    main()
