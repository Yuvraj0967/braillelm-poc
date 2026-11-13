# Results — 120-item evaluation

This is the completed evaluation that the report and the comparison kit are
built around. It is a **closed-distribution** test on a small synthetic corpus,
not an open-domain braille benchmark. Read the caveats before quoting the
numbers.

The full 26-page report, including every one of the 120 rows, is
[`BrailleLM_Report.pdf`](BrailleLM_Report.pdf). The raw rows and the scoring
code are in [`../benchmark/`](../benchmark/) and [`../results/`](../results/).

---

## Protocol

| Item | Value |
|---|---|
| Task | UEB Grade 1 braille → English print |
| Corpus | 600 synthetic sentences, closed 138-word vocabulary |
| Split | random, 480 train / 120 test |
| Base model | `HuggingFaceTB/SmolLM2-135M` (135M parameters) |
| Adaptation | LoRA r=8, alpha=16, dropout 0.05, `q_proj` / `v_proj` |
| Native arm | +177 braille tokens, BKFT initialisation, merged checkpoint |
| Baseline arm | stock tokenizer, raw Unicode braille |
| Baseline-256 control | same baseline, longer sequence budget |
| Training | 1 epoch, batch 4, lr 1e-3, CPU |
| Decoding | greedy, short output |
| Test size | 120 sentences |

The native checkpoint is a **merged** model with a 49,329-token vocabulary. The
baseline is a LoRA adapter over the stock 49,152-token tokenizer.

## Headline table

| Arm | Exact | CER | BLEU | Distinct outputs | Vocabulary emitted | Actor slot |
|---|---:|---:|---:|---:|---:|---:|
| **BrailleLM native** | **120 / 120** | **0.0%** | **100.0** | 120 / 120 | 138 / 138 | **100%** |
| Baseline (raw braille) | 0 / 120 | 63.8% | 4.0 | 9 / 120 | 22 / 138 | 2.7% |
| Baseline 256 control | 0 / 120 | 60.2% | 4.2 | 15 / 120 | 29 / 138 | 8.0% |

Metric definitions:

- **Exact** — whole sentence matches after trimming; the headline number.
- **CER** — mean Levenshtein distance ÷ reference length.
- **BLEU** — corpus sacreBLEU, coarse on 120 short sentences.
- **Distinct outputs** — how many different sentences the arm produced across
  the 120 inputs. A model that ignores its input collapses to a handful.
- **Vocabulary emitted** — how many of the 138 reference words ever appear.
- **Actor slot** — accuracy on word 2 of the 112 nine-word sentences. A cheap
  check that the model reads the actor noun rather than only the sentence frame.

## What the numbers say

**The native arm reproduces every held-out sentence exactly.** CER is zero,
BLEU is the maximum, all 120 outputs are distinct, and all 138 vocabulary items
appear. On this closed distribution, the braille-native tokenizer plus BKFT
makes the task learnable in one epoch on a CPU.

**The baseline fails by ignoring the input.** It emits only 9 distinct
sentences for 120 different braille inputs, uses 22 of the 138 words, and gets
the actor right 2.7% of the time. Its BLEU of 4.0 is not a near miss; it is
fluent, grammatical output that does not depend on the braille. The 256-token
control does not fix this (15 distinct, 29 words, 8.0% actor). This is the
cautionary result the report highlights: a language model can look plausible
while failing to condition on its input at all.

**The tokenizer is doing real work.** The native vocabulary collapses frequent
cell sequences, so the same braille costs fewer model tokens and the model can
learn the cell→letter mapping within the budget. The baseline fragments braille
into many tokens and never learns to read it in this setting.

## What this does not show

- **It is not open-domain translation.** The corpus is generated from three
  sentence patterns and a closed 138-word vocabulary. Every test word also
  occurs in training because the split is random. Strong exact-match numbers
  here partly reflect a small, closed distribution.
- **It is not a general braille model.** Grade 1 only, no contractions, no
  formatting, no math or music codes.
- **It is not a ranking against other LLMs.** A general model evaluated
  zero-shot is at a disadvantage on this distribution; the comparison kit
  exists to make that comparison explicit and fair.
- **n = 120 is small.** Near 100%, the 95% Wilson interval is roughly ±3
  points; near 50%, roughly ±9. Do not treat a two-row difference as real.
- **The harder generalisation test is separate.** The `word_disjoint` split in
  `run_experiment.py` holds out vocabulary so every test sentence contains a
  word never seen in training. That is the right test for generalisation; run
  it with the command in the main README.

## Reproducing the score

The benchmark folder contains the 120 references, the braille-only inputs, the
prompt, the eight few-shot examples, and a standalone scorer:

```bash
cd benchmark
python score_llm.py predictions.tsv --name "your model"
```

See [`../benchmark/README.md`](../benchmark/README.md) for accepted prediction
formats, the zero-shot conditions, and the reference rows.

## Artifact map

| Artifact | Location |
|---|---|
| Full report (PDF) | [`BrailleLM_Report.pdf`](BrailleLM_Report.pdf) |
| Report (HTML) | [`BrailleLM_Report.html`](BrailleLM_Report.html) |
| 120 test items and scorer | [`../benchmark/`](../benchmark/) |
| Native predictions | [`../results/preds_native.tsv`](../results/preds_native.tsv) |
| Baseline predictions | [`../results/preds_baseline.tsv`](../results/preds_baseline.tsv) |
| 256-token control predictions | [`../results/preds_baseline_256.tsv`](../results/preds_baseline_256.tsv) |
| Machine-readable rows and scores | [`../results/comparison_pack.json`](../results/comparison_pack.json) |
| Baseline LoRA adapter | [`../models/baseline-lora/`](../models/baseline-lora/) |
| Native merged checkpoint | [release asset](../models/README.md) |
