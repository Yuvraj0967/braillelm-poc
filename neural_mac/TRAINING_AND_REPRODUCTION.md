# Training and reproduction

## Recorded environment

- Host: Apple M3 Pro with 19 GB unified memory.
- Device and precision: PyTorch MPS, float32.
- Python packages recorded locally: `torch==2.15.0`,
  `transformers==4.57.6`, `peft==0.21.2`, `sacrebleu==2.6.0`,
  `datasets==4.8.5`.
- Base model: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
  The released file is a PEFT adapter and tokenizer, not a copy of the Qwen
  base weights.
- Training code: `code/train.py`; scoring code: `code/eval.py`; added token
  list: `code/native_tokens_exact.json`.
- Exact datasets, hashes, source commit, and run metadata:
  `experiment_manifest.json` and `DATA_AND_EVALUATION.md`.

The following commands express the three saved training stages. They are a
reproduction recipe assembled from the recorded arguments, checkpoint metadata,
and logs; they are not a claim that these exact shell lines were saved in a
terminal transcript. Run from this `neural_mac/` directory. The first stage
trains from the base model; later stages resume the saved adapter.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python code/train.py \
  --mode native --init bkft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data --val_data data/validation.jsonl \
  --tokens code/native_tokens_exact.json \
  --out checkpoints/epoch1 --epochs 1 \
  --batch 2 --grad_accum 8 --lr 0.0001 --max_len 256 \
  --lora_rank 8 --train_new_tokens_only \
  --seed 42 --eval_every 1000 --save_every 2000

python code/train.py \
  --mode native --reuse_tokenizer \
  --resume_adapter checkpoints/epoch1/epoch1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data --val_data data/validation.jsonl \
  --out checkpoints/continuation1 --epochs 1 \
  --batch 2 --grad_accum 8 --lr 0.00005 --max_len 256 \
  --lora_rank 8 --train_new_tokens_only \
  --seed 43 --eval_every 1000 --save_every 2000 --max_steps 4000

python code/train.py \
  --mode native --reuse_tokenizer \
  --resume_adapter checkpoints/continuation1/step4000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data --val_data data/validation.jsonl \
  --out checkpoints/continuation2 --epochs 1 \
  --batch 2 --grad_accum 8 --lr 0.00005 --max_len 256 \
  --lora_rank 8 --train_new_tokens_only \
  --seed 44 --eval_every 1000 --save_every 2000 --max_steps 4000
```

`data/train.jsonl` contains the 15,886 pairs used by these stages. The prompt
is `Translate the following braille to English text.\nBraille: ...\nEnglish:`.
Targets are the English sentence plus EOS. Prompt labels are masked, so the
cross-entropy loss is computed only on the answer. AdamW is used with an
effective batch size of 16. The LoRA adapter targets `q_proj` and `v_proj`
with rank 8, alpha 16 and dropout 0.05. The first run initializes 177 braille
token embeddings from the corresponding print fragments; PEFT records their
trainable token indices. Later runs reuse the tokenizer and adapter.

The original stage-2 process saved `step4000` and then was stopped during the
next batches. The chosen checkpoint is exactly that `step4000` state. The final
continuation saved a further `step4000` checkpoint with validation loss 0.2155.
Optimizer state was not saved with these adapter files, so a new AdamW optimizer
is created at each continuation. Exact bitwise reproduction across hardware and
dependency versions is not guaranteed.

## Evaluate the released model

Download and extract the release asset into `model/` so that
`model/adapter_model.safetensors` and `model/tokenizer.json` exist. Then:

```bash
mkdir -p model
tar -xf BrailleLM_Qwen2.5_0.5B_BKFT_Mac_adapter_step4000.tar -C model
shasum -a 256 model/adapter_model.safetensors
# Expected: ccffaa08ac5fdbb95ffbf393c447724bd9a3a133632bb60a9d7a781fb975f9b6

python code/eval.py \
  --ckpt model --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --test_file data/pilot_plain.jsonl --max_new 96 --num_beams 4 \
  --out rerun_plain.json --predictions rerun_plain_predictions.jsonl
```

Use `data/pilot_extended.jsonl` for the extended development pilot. The
locked final set was already evaluated once with the same selected checkpoint
and `--num_beams 4`; its saved metrics and predictions are in `results/`.
For one sentence, use `code/predict.py`.

The full training run takes substantial time and disk space. The notebook in
this folder opens the measured results immediately and includes the same
training commands as a record and reproduction guide.
