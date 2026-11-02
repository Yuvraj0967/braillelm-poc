"""
run_all.py -- cross-platform end-to-end runner (Windows / Mac / Linux).
Same pipeline as run_all.sh, but works in plain PowerShell/cmd with no bash.

Usage:
  python run_all.py                       # CPU, SmolLM2-135M, Grade 1
  python run_all.py --model Qwen/Qwen2.5-0.5B-Instruct --grade 2 --epochs 3
  python run_all.py --gpu                 # on a CUDA machine (e.g. RTX 4090)
"""
import argparse, subprocess, sys, json, os

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        print(f"\n[run_all] step failed: {' '.join(cmd)}")
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--grade", type=int, default=1, choices=[1, 2])
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--merges", type=int, default=150)
    ap.add_argument("--gpu", action="store_true", help="use CUDA (omit --cpu flag)")
    ap.add_argument("--input", default=None, help="optional textfile, one sentence per line")
    args = ap.parse_args()
    cpu = [] if args.gpu else ["--cpu"]

    print("== 1. Data prep (Grade %d) ==" % args.grade)
    prep = ["prep_data.py", "--grade", str(args.grade), "--out", "data"]
    if args.input:
        prep += ["--input", args.input]
    run(prep)

    print("== 2. Braille-native tokenizer ==")
    run(["braille_tokenizer.py", "--data", "data", "--num_merges", str(args.merges),
         "--out", "braille_tokens.json"])

    print("== 3. Baseline (standard tokenizer) ==")
    run(["train.py", "--mode", "baseline", "--model", args.model, "--data", "data",
         "--epochs", str(args.epochs)] + cpu)
    run(["eval.py", "--ckpt", "out_baseline", "--data", "data"] + cpu)

    print("== 4. Native (braille tokenizer + BKFT) ==")
    run(["train.py", "--mode", "native", "--model", args.model, "--data", "data",
         "--epochs", str(args.epochs)] + cpu)
    run(["eval.py", "--ckpt", "out_native", "--data", "data"] + cpu)

    print("\n== 5. Comparison ==")
    b = json.load(open("out_baseline/eval.json"))
    n = json.load(open("out_native/eval.json"))
    print(f"{'metric':<18}{'baseline':>12}{'native':>12}")
    for k in ["sacrebleu", "cer", "tokens_per_cell"]:
        print(f"{k:<18}{str(b[k]):>12}{str(n[k]):>12}")
    better_tokens = (n["tokens_per_cell"] or 9) < (b["tokens_per_cell"] or 9)
    better_acc = (n["sacrebleu"] or 0) >= (b["sacrebleu"] or 0) or (n["cer"] <= b["cer"])
    print("\nVERDICT:", "PASS - native wins" if (better_tokens and better_acc)
          else "MIXED - inspect the numbers above")

if __name__ == "__main__":
    main()
