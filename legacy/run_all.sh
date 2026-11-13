#!/usr/bin/env bash
# End-to-end BrailleLM proof-of-concept (CPU smoke test).
# Runs: data prep -> braille tokenizer -> baseline train/eval -> native train/eval -> compare.
set -e

MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M}"
GRADE="${GRADE:-1}"
CPU_FLAG="${CPU_FLAG:---cpu}"     # set CPU_FLAG="" on a CUDA machine
EPOCHS="${EPOCHS:-3}"

echo "== 1. Data prep (Grade $GRADE) =="
python prep_data.py --grade "$GRADE" --out data

echo "== 2. Braille-native tokenizer =="
python braille_tokenizer.py --data data --num_merges 150 --out braille_tokens.json

echo "== 3. Baseline (standard tokenizer) =="
python train.py --mode baseline --model "$MODEL" --data data $CPU_FLAG --epochs "$EPOCHS"
python eval.py  --ckpt out_baseline --data data $CPU_FLAG

echo "== 4. Native (braille tokenizer + BKFT) =="
python train.py --mode native --model "$MODEL" --data data $CPU_FLAG --epochs "$EPOCHS"
python eval.py  --ckpt out_native --data data $CPU_FLAG

echo "== 5. Comparison =="
python - <<'PY'
import json
b = json.load(open("out_baseline/eval.json"))
n = json.load(open("out_native/eval.json"))
print(f"{'metric':<18}{'baseline':>12}{'native':>12}")
for k in ["sacrebleu", "cer", "tokens_per_cell"]:
    print(f"{k:<18}{str(b[k]):>12}{str(n[k]):>12}")
better_tokens = (n["tokens_per_cell"] or 9) < (b["tokens_per_cell"] or 9)
better_acc = (n["sacrebleu"] or 0) >= (b["sacrebleu"] or 0) or (n["cer"] <= b["cer"])
print("\nVERDICT:", "PASS - native wins" if (better_tokens and better_acc)
      else "MIXED - inspect numbers")
PY
