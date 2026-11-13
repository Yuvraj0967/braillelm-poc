# BrailleLM Qwen2.5-0.5B Grade 1 adapter

## Summary

This is a neural Braille-to-English transcription research checkpoint for clean
Unicode Unified English Braille Grade 1. It is a PEFT adapter trained on top of
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
The tokenizer has 177 added braille tokens initialized using the print strings
they represent. The LoRA adapter was trained on 15,886 sentence pairs and
continued twice. The selected output uses deterministic four-beam decoding.

Download the checkpoint archive from the
[neural Mac release](https://github.com/Yuvraj0967/braillelm-poc/releases/tag/neural-qwen-mac-2026-11-04).
The release contains the `step4000` adapter, tokenizer, configuration, and
training metadata. The base model weights are not bundled.
The adapter archive is about 1 GB because this PEFT checkpoint includes copies
of the resized input and output embedding matrices alongside the small LoRA
and trainable-token tensors. Its file size should not be mistaken for the
number of parameters optimized in the LoRA layers.
Archive SHA-256:
`fb9d88a62bf85d47d5fefb62996dccc2592390d02c22a3e96d3e2307d60dc3b3`.

## Measured performance

On the **locked final test**: 216/399 (54.1%) strict exact match,
sacreBLEU 89.16, mean CER 0.0231. Among the 263 final sentences with at least
one training-unseen word, the score was 116/263 (44.1%); among 136 sentences
without such a word, it was 100/136 (73.5%). The final exact-match Wilson
95% interval is 49.2%–59.0%. The input braille was still machine-generated.

On the **development** plain pilot: 109/200 (54.5%) strict exact match,
sacreBLEU 90.02, mean CER 0.0178. On a separate extended development pilot:
93/199 (46.7%) strict exact, sacreBLEU 89.27, mean CER 0.0253. Beam width was
chosen after seeing development results. The development pilots are not
independent final or human-entered braille scores. Raw metric JSON and all
generated hypotheses are in `results/`.

## Usage

From this directory, after extracting the adapter archive into `model/`:

```bash
python code/predict.py --ckpt model \
  --braille '⠠⠞⠓⠑⠀⠃⠁⠞⠞⠇⠑⠀⠑⠭⠞⠑⠝⠙⠑⠙⠀⠺⠑⠇⠇⠀⠊⠝⠞⠕⠀⠞⠓⠑⠀⠝⠊⠛⠓⠞⠲'
```

The exact input format is Unicode braille cells. The command loads Qwen base
weights and the adapter, and uses four-beam decoding by default. The printed
answer must be checked; this is an experimental system.

## Intended and unsupported use

This checkpoint is intended for capstone demonstration and research on
braille-aware tokenization and small neural models. It has not been validated
for accessibility-critical transcription, human-entered braille, full Grade 2,
braille OCR, mathematical notation, multilingual braille, or noisy strings.
Do not rely on it where an incorrect transcription could matter to a reader.
It frequently misspells rare names or substitutes a plausible word.

The training labels were produced from human-written English using liblouis;
therefore the test distribution is clean generated braille. A strong liblouis
back-translation baseline is exactly correct on the generated final pairs and
should be considered for practical Grade 1 transcription. This model tests a
neural method; it is not a claim that neural inference is superior to that
translator.

## Provenance and attribution

- Original project: [Yuvraj0967/braillelm-poc](https://github.com/Yuvraj0967/braillelm-poc),
  source commit `352fd4af6aeb808a2e97adf17781a78c866c8a9c`.
- Related method: [BrailleLLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1454/).
- English source: [Salesforce WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext).
- Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Braille generator: liblouis 3.39.0, `en-ueb-g1.ctb`, `unicode.dis`.

See `DATA_AND_EVALUATION.md` for the WikiText card's inconsistent license
metadata, split construction, and SHA-256 values. The adapter's SHA-256 is
`ccffaa08ac5fdbb95ffbf393c447724bd9a3a133632bb60a9d7a781fb975f9b6`.
