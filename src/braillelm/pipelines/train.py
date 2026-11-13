"""Train the BrailleLM cell-level reader with CTC.

Works for Grade 1 only (rows with "braille") or for both grades (rows that
also have "braille_g2", see build_data_g2.py). With both, each row is shown as
Grade 2 with probability --p_g2 and the model learns a grade switch.

Model selection uses only the development rows (validation + both pilots).
The locked final set is never read by this script.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from braillelm import tracking
from braillelm.model import GRADES, PAD_IN, BrailleReader, encode_cells, encode_print, load, save
from braillelm.noise import corrupt_braille, corrupt_ids

KEYS = ("braille", "braille_g2")          # row field for grade 0 and grade 1


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def batches(items, lengths, cell_budget, rng):
    """items: list of (row, grade); lengths: matching input lengths."""
    order = list(range(len(items)))
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
    return [[items[i] for i in b] for b in out]


def dev_scores(model, dev, noise_rate, grade):
    """(faithful on clean, repair on noisy, repair on clean) strict exact."""
    brl = [r[KEYS[grade]] for r in dev]
    noisy = [corrupt_braille(b, noise_rate, 1000 + i) for i, b in enumerate(brl)]
    n = len(dev)

    def exact(inputs, mode):
        got = model.translate(inputs, mode, GRADES[grade])
        return sum(h == r["print"] for (h, _), r in zip(got, dev)) / n
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
    ap.add_argument("--frames", type=int, default=2, help="CTC frames per cell (2 for Grade 1, 8 for Grade 2)")
    ap.add_argument("--p_g2", type=float, default=0.0, help="share of rows shown as Grade 2 (needs braille_g2)")
    ap.add_argument("--p_noisy", type=float, default=0.6, help="share of training rows shown in repair mode")
    ap.add_argument("--max_noise", type=float, default=0.08)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--mlflow", metavar="EXPERIMENT", help="log this run to MLflow under this experiment name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="checkpoint to continue from (weights only; the optimiser restarts)")
    ap.add_argument("--start_step", type=int, default=0, help="step the resumed checkpoint was saved at")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
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
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    amp = device.type == "cuda"

    rows = read(data / "train.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    two_grades = args.p_g2 > 0
    if two_grades and "braille_g2" not in rows[0]:
        ap.error("--p_g2 needs training rows with a braille_g2 field (see build_data_g2.py)")
    X = [[np.asarray(encode_cells(r[KEYS[g]]), dtype=np.uint8) for r in rows] for g in range(2 if two_grades else 1)]
    Y = [np.asarray(encode_print(r["print"]), dtype=np.uint8) for r in rows]
    n_rows = len(rows)
    del rows
    dev = read(data / "validation.jsonl") + read(data / "pilot_plain.jsonl") + read(data / "pilot_extended.jsonl")

    resumed = None
    if args.resume:
        model, resumed = load(args.resume)
        if model.frames != args.frames or model.grades != (2 if two_grades else 1):
            ap.error("--resume checkpoint has a different frames/grades shape than this run")
    else:
        model = BrailleReader(args.d, args.layers, args.heads, args.ff, args.dropout,
                              frames=args.frames, grades=2 if two_grades else 1)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train rows {n_rows}  dev rows {len(dev)}  params {n_params/1e6:.2f}M  device {device}  "
          f"grades {model.grades}  frames {model.frames}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        ramp = min(1.0, (step - args.start_step + 1) / 50) if args.resume else 1.0   # let Adam statistics rebuild
        return ramp * args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    log, best, step, epoch, t0, run_loss = [], -1.0, args.start_step, 0, time.time(), None
    if resumed:
        best = sum(v for k, v in resumed.items() if k in ("dev_exact_clean", "dev_exact_noise3",
                                                          "dev_g2_exact_clean", "dev_g2_exact_noise3"))
    with tracking.run(args.mlflow, run_name=out.name) as tracker:
        tracker.params(vars(args))
        while step < args.steps:
            items = [(i, 1 if two_grades and rng.random() < args.p_g2 else 0) for i in range(n_rows)]
            lengths = [len(X[g][i]) for i, g in items]
            for batch in batches(items, lengths, args.cell_budget, rng):
                if step >= args.steps:
                    break
                xs, modes, grades = [], [], []
                for i, g in batch:
                    ids = X[g][i].tolist()
                    if rng.random() < args.p_noisy:       # repair mode: input may be damaged
                        if rng.random() < 0.85:
                            ids = corrupt_ids(ids, rng.uniform(0.005, args.max_noise), rng)
                        modes.append(1)
                    else:                                 # faithful mode: input is always clean
                        modes.append(0)
                    grades.append(g)
                    xs.append(ids)
                T = max(map(len, xs))
                x = torch.full((len(xs), T), PAD_IN, dtype=torch.long)
                for r, q in enumerate(xs):
                    x[r, :len(q)] = torch.tensor(q)
                tgt = torch.cat([torch.from_numpy(Y[i].astype(np.int64)) for i, _ in batch])
                in_len = torch.tensor([len(q) * model.frames for q in xs])
                tgt_len = torch.tensor([len(Y[i]) for i, _ in batch])

                model.train()
                for group in opt.param_groups:
                    group["lr"] = lr_at(step)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x.to(device), torch.tensor(modes, device=device),
                                   torch.tensor(grades, device=device))
                logp = logits.float().log_softmax(-1).transpose(0, 1)
                loss = F.ctc_loss(logp, tgt.to(device), in_len, tgt_len, blank=0, zero_infinity=True)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                step += 1
                run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()

                if step % 50 == 0:
                    el, done = time.time() - t0, step - args.start_step
                    print(f"step {step:6d}/{args.steps} epoch {epoch} loss {run_loss:.4f} lr {lr_at(step):.2e} "
                          f"{el/done:.2f}s/step eta {(args.steps-step)*el/done/60:.0f}m", flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    clean, noisy, repair_clean = dev_scores(model, dev, 0.03, 0)
                    rec = {"step": step, "epoch": epoch, "train_loss": round(run_loss, 5),
                           "dev_exact_clean": round(clean, 4), "dev_exact_noise3": round(noisy, 4),
                           "dev_exact_clean_repair_mode": round(repair_clean, 4)}
                    total = clean + noisy
                    if two_grades:
                        c2, n2, r2 = dev_scores(model, dev, 0.03, 1)
                        rec.update({"dev_g2_exact_clean": round(c2, 4), "dev_g2_exact_noise3": round(n2, 4),
                                    "dev_g2_exact_clean_repair_mode": round(r2, 4)})
                        total += c2 + n2
                    rec["minutes"] = round((time.time() - t0) / 60, 1)
                    log.append(rec)
                    print("EVAL", json.dumps(rec), flush=True)
                    tracker.metrics(rec, step=step)
                    extra = {"args": vars(args), "params": n_params, **rec}
                    save(model, out / "last.pt", extra)
                    if total > best:
                        best = total
                        save(model, out / "best.pt", extra)
                    (out / "train_log.json").write_text(json.dumps(log, indent=1))
            epoch += 1
        for name in ("best.pt", "train_log.json"):
            if (out / name).exists():
                tracker.artifact(out / name)
    print("done", flush=True)


if __name__ == "__main__":
    main()
