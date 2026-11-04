#!/bin/zsh
set -e
PY=../braillelm_venv/bin/python
COMMON=(--data data --val_data data/validation.jsonl --epochs 3 --batch 2 --grad_accum 8 --eval_every 200 --lr 0.0001 --max_len 256 --seed 42 --train_embeddings --tokens native_tokens_exact.json --model HuggingFaceTB/SmolLM2-135M)
$PY -u train.py --mode native --init random --out random_3ep "${COMMON[@]}" > random_3ep.log 2>&1
$PY -u train.py --mode native --init bkft --out bkft_3ep "${COMMON[@]}" > bkft_3ep.log 2>&1
