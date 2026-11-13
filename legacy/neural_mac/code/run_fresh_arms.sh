#!/bin/zsh
set -e
PY=../braillelm_venv/bin/python
COMMON=(--data data --val_data data/validation.jsonl --epochs 3 --batch 2 --grad_accum 8 --eval_every 200 --lr 0.0001 --seed 42 --train_embeddings)
$PY -u train.py --mode baseline --model HuggingFaceTB/SmolLM2-135M --out baseline_3ep --max_len 512 "${COMMON[@]}" > baseline_3ep.log 2>&1
$PY -u train.py --mode native --init random --tokens native_tokens_exact.json --model HuggingFaceTB/SmolLM2-135M --out random_3ep --max_len 256 "${COMMON[@]}" > random_3ep.log 2>&1
$PY -u train.py --mode native --init bkft --tokens native_tokens_exact.json --model HuggingFaceTB/SmolLM2-135M --out bkft_3ep --max_len 256 "${COMMON[@]}" > bkft_3ep.log 2>&1
