# BrailleLM baseline — LoRA adapter

LoRA adapter for the **baseline** arm of the BrailleLM proof of concept:
SmolLM2-135M reading raw Unicode braille with its stock tokenizer.

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M` |
| PEFT type | LoRA |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Modules to save | none |
| Tokenizer | stock 49,152-token SmolLM2 tokenizer |
| Training data | 480 synthetic braille→print pairs (Grade 1) |
| Training | 1 epoch, batch 4, lr 1e-3, max_len 192, CPU |
| Result | 0/120 exact, 63.8% CER, 4.0 BLEU on the 120-item test |

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tok = AutoTokenizer.from_pretrained(".")          # this directory
base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, ".")
model.eval()
```

Prompt:

```text
Translate the following braille to English text.
Braille: <braille>
English:
```

## What it demonstrates

This arm is the cautionary control. It produces fluent English that ignores the
braille input: 9 distinct sentences across 120 different inputs, 22 of the 138
reference words, and 2.7% accuracy on the actor slot. See
[`../../docs/RESULTS.md`](../../docs/RESULTS.md) and
[`../../results/`](../../results/) for the rows.

## Limitations

Research artifact, closed synthetic distribution, Grade 1 only. Not a
translation system for real braille readers. The base model is
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); check its
model card for license terms.
