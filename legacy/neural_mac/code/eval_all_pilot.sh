#!/bin/zsh
set -e
PY=../braillelm_venv/bin/python
for arm in baseline random bkft; do
  case "$arm" in
    baseline) base=HuggingFaceTB/SmolLM2-135M ;;
    random|bkft) base=HuggingFaceTB/SmolLM2-135M ;;
  esac
  case "$arm" in
    baseline|random) ckpt="${arm}_3ep/epoch1" ;;
    bkft) ckpt="${arm}_3ep/epoch3" ;;
  esac
  $PY -u eval.py --ckpt "$ckpt" --base_model "$base" --test_file data/test.jsonl --max_new 96 --out "eval_${arm}_pilot.json" --predictions "eval_${arm}_pilot_predictions.jsonl" > "eval_${arm}_pilot.log" 2>&1
 done
