"""
run_experiment.py -- the full controlled experiment (cross-platform).

Runs three arms across several seeds and reports mean +/- spread, so the result
has the controls the smoke test lacked:

  baseline        : braille through the original tokenizer
  native-random   : braille-native tokenizer, new embeddings RANDOM-init
  native-bkft     : braille-native tokenizer, new embeddings BKFT-init

Reading the comparison:
  native-bkft vs baseline        -> value of the braille-native tokenizer
  native-bkft vs native-random   -> value of BKFT specifically (the key control)

Standard configuration (what the reported results use):
  python run_experiment.py --seeds 0 1 2 --input sentences.txt \
      --split word_disjoint --holdout_frac 0.15 \
      --gpu --bf16 --train_new_tokens_only --lr_end 5e-5 --beams 4

Cheap CPU control run on the small model:
  python run_experiment.py --seeds 0 1 2 --model HuggingFaceTB/SmolLM2-135M --cpu

Notes:
  * --train_new_tokens_only trains ONLY the 177 new braille embedding rows.
    Preferred over --train_embeddings: same benefit, a fraction of the
    trainable parameters, and it cannot degrade the base model's English.
  * The baseline arm automatically gets 2x the max_len, because braille costs
    ~3 tokens/cell through a standard tokenizer and ~0.57 native. A single
    shared window truncates the baseline's input and understates the control.
"""
import argparse, subprocess, sys, json, os, statistics

def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if r.returncode != 0:
        print(f"[experiment] step failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(r.returncode)

# Per-arm max_len. Braille costs ~3 tokens/cell through a STANDARD tokenizer
# and ~0.57 native, so one shared window silently truncates the baseline's
# input and makes the control look worse than it is. Give baseline 2x.
ARMS = [
    ("baseline",      ["--mode", "baseline"],                   2.0),
    ("native-random", ["--mode", "native", "--init", "random"], 1.0),
    ("native-bkft",   ["--mode", "native", "--init", "bkft"],   1.0),
]
METRICS = ["sacrebleu", "cer", "exact_match", "tokens_per_cell"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="standard model for the headline experiment; pass "
                         "HuggingFaceTB/SmolLM2-135M for a cheap CPU control run")
    ap.add_argument("--grade", type=int, default=1, choices=[1, 2])
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--merges", type=int, default=150)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--input", default=None)
    ap.add_argument("--split", choices=["random", "word_disjoint"], default="random")
    ap.add_argument("--holdout_frac", type=float, default=0.15)
    ap.add_argument("--train_embeddings", action="store_true")
    ap.add_argument("--train_new_tokens_only", action="store_true",
                    help="preferred over --train_embeddings (see train.py)")
    ap.add_argument("--lr_end", type=float, default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    dev = [] if args.gpu else ["--cpu"]
    if args.bf16 and args.gpu:
        dev += ["--bf16"]
    te = ["--train_embeddings"] if args.train_embeddings else []
    if args.train_new_tokens_only:
        te = ["--train_new_tokens_only"]
    if args.lr_end is not None:
        te += ["--lr_end", str(args.lr_end)]
    ev_args = (["--beams", str(args.beams)] +
               (["--limit", str(args.eval_limit)] if args.eval_limit else []))

    # 1-2: data + tokenizer (shared by all arms/seeds)
    prep = ["prep_data.py", "--grade", args.grade, "--out", "data",
            "--split", args.split, "--holdout_frac", args.holdout_frac]
    if args.input:
        prep += ["--input", args.input]
    run(prep)
    run(["braille_tokenizer.py", "--data", "data", "--num_merges", args.merges, "--out", "braille_tokens.json"])

    # 3: train + eval every (arm, seed)
    results = {}  # arm -> metric -> [values across seeds]
    for arm, arm_args, len_mult in ARMS:
        results[arm] = {m: [] for m in METRICS}
        arm_max_len = int(args.max_len * len_mult)
        for seed in args.seeds:
            out = f"out_{arm}_seed{seed}"
            run(["train.py"] + arm_args + ["--model", args.model, "--data", "data",
                 "--epochs", args.epochs, "--max_len", arm_max_len, "--seed", seed,
                 "--out", out] + dev + te)
            run(["eval.py", "--ckpt", out, "--data", "data"] + dev + ev_args)
            ev = json.load(open(f"{out}/eval.json"))
            for m in METRICS:
                if ev.get(m) is not None:
                    results[arm][m].append(ev[m])

    # 4: aggregate
    def ms(xs):
        if not xs:
            return (None, None)
        return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    print("\n" + "=" * 70)
    print(f"RESULTS  ({len(args.seeds)} seeds: {args.seeds}, grade {args.grade}, {args.model})")
    print("=" * 70)
    hdr = f"{'metric':<16}" + "".join(f"{a:>18}" for a, _, _ in ARMS)
    print(hdr)
    agg = {a: {} for a, _, _ in ARMS}
    for m in METRICS:
        row = f"{m:<16}"
        for arm, _, _ in ARMS:
            mean, sd = ms(results[arm][m])
            agg[arm][m] = {"mean": mean, "std": sd}
            row += f"{(f'{mean:.3f}±{sd:.3f}' if mean is not None else 'NA'):>18}"
        print(row)

    print("\nControls:")
    def better(a, b, m, lower=False):
        x, y = agg[a][m]["mean"], agg[b][m]["mean"]
        if x is None or y is None:
            return "NA"
        win = (x < y) if lower else (x > y)
        return f"{a} {'wins' if win else 'does NOT win'} ({x:.3f} vs {y:.3f})"
    print("  tokenizer value  (native-bkft vs baseline), sacreBLEU:",
          better("native-bkft", "baseline", "sacrebleu"))
    print("  BKFT value       (native-bkft vs native-random), sacreBLEU:",
          better("native-bkft", "native-random", "sacrebleu"))
    print("  BKFT value       (native-bkft vs native-random), CER:",
          better("native-bkft", "native-random", "cer", lower=True))

    json.dump({"seeds": args.seeds, "grade": args.grade, "model": args.model,
               "arms": agg, "raw": results},
              open("experiment_results.json", "w"), indent=2)
    print("\n[experiment] wrote experiment_results.json")

if __name__ == "__main__":
    main()
