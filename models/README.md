# Model artifacts

Two trained artifacts accompany the 120-item evaluation. The small baseline
adapter is tracked in git; the merged native checkpoint is too large for a
normal git repository and is published as a release asset.

## Native merged checkpoint

**Download:** [braillelm-native-smollm2-135m.tar](https://github.com/Yuvraj0967/braillelm-poc/releases/latest/download/braillelm-native-smollm2-135m.tar) (~652 MB)

Contents:

```text
braillelm_native/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

This is the model behind the 120/120 row in [`../docs/RESULTS.md`](../docs/RESULTS.md).
It is a **merged** SmolLM2-135M with a 49,329-token vocabulary (177 braille
tokens added and BKFT-initialised), fine-tuned with LoRA on 480 braille→print
pairs. The adapter is already merged into the weights, so no PEFT step is
needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("braillelm_native")
model = AutoModelForCausalLM.from_pretrained("braillelm_native", dtype="float32")
model.eval()

prompt = (
    "Translate the following braille to English text.\n"
    "Braille: " + braille + "\nEnglish:"
)
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                     pad_token_id=tok.pad_token_id)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

Training configuration: 1 epoch, batch 4, learning rate 1e-3, max sequence
length 128, CPU, LoRA r=8 / alpha=16 on `q_proj` and `v_proj`.

## Baseline LoRA adapter

Tracked at [`baseline-lora/`](baseline-lora/). It uses the **stock** 49,152-token
SmolLM2 tokenizer (no added braille tokens), so load the tokenizer from the
adapter directory and resize before attaching:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tok = AutoTokenizer.from_pretrained("models/baseline-lora")
base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, "models/baseline-lora")
model.eval()
```

Training configuration: 1 epoch, batch 4, learning rate 1e-3, max sequence
length 192, CPU, LoRA r=8 / alpha=16.

## What these models are, and are not

Both were trained on a synthetic, closed 138-word corpus with three sentence
patterns. They are research artifacts for the proof of concept, not braille
translation systems for real readers. Grade 1 only; no contractions,
formatting, math or music codes. See
[`../docs/RESULTS.md`](../docs/RESULTS.md) and the limitations section of the
main README.

The base model is [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M);
check its model card for the license and terms that apply to the base weights.
