# BrailleLM benchmark and comparison kit

This folder lets you run any other language model on the exact 120 held-out
sentences used in the BrailleLM report, score it the same way, and compare the
result to the three BrailleLM arms.

The task is braille → print: each input is a UEB Grade 1 (uncontracted) braille
sentence, and the expected output is the corresponding English sentence.

This is the **120-item random split** from the report (480 train / 120 test,
closed 138-word vocabulary), not the harder word-disjoint split that
`run_experiment.py` can build. See [`docs/RESULTS.md`](../docs/RESULTS.md) for
the protocol and caveats.

## Files

| File | What it is |
| --- | --- |
| `test_set.jsonl` | 120 items: `id`, `braille`, `print` (the reference). |
| `braille_only.txt` | The 120 braille inputs, one per line, in `id` order. |
| `references.txt` | The 120 English references, one per line, in `id` order. |
| `prompt.txt` | The exact prompt template used for the BrailleLM arms. |
| `few_shot.txt` | Eight train-split examples (braille → print) for a few-shot condition. |
| `predictions_template.tsv` | `id` + empty text column; fill the second column. |
| `train_pairs.jsonl` | All 480 training pairs, if you want to fine-tune a comparator. |
| `score_llm.py` | Standalone scorer and comparison table. |
| `reference_scores.json` | BrailleLM native, baseline, and 256-token control scores. |
| [`../results/comparison_pack.json`](../results/comparison_pack.json) | Same data in one machine-readable file (rows, prototypes, vocabulary, scores). |

`test_set.jsonl` is UTF-8. Do not reorder it; align by `id`. The braille inputs
are Unicode braille (U+2800 block), not ASCII braille, and use U+2800 (braille
pattern blank) as the word separator inside a sentence.

## Run a comparison

1. Produce one English sentence for every input, using the conditions below.
2. Save them as `predictions_template.tsv` with the `id` column intact, or as a
   plain text file with 120 lines in `id` order.
3. Score:

```bash
python score_llm.py predictions.tsv --name "GPT-5"
python score_llm.py predictions.jsonl --name "Claude" --out results/
python score_llm.py predictions.txt --name "local-7B" --order
```

The scorer prints:

```
model                   exact     CER   BLEU distinct    vocab   actor
-------------------------------------------------------------------------
BrailleLM native      120/120    0.0%  100.0     120  138/138  100.0%
BrailleLM baseline      0/120   63.8%    4.0       9   22/138    2.7%
Baseline 256 control    0/120   60.2%    4.2      15   29/138    8.0%
your model              ?/120      ?       ?       ?       ?/138      ?
```

It writes `scores_<name>.json` and `per_row_<name>.csv` next to the script
(or in `--out`). Only the standard library is required; corpus BLEU needs
`sacrebleu`.

Accepted prediction formats:

- TSV/CSV with `id, text` (a header row is optional).
- JSONL with `{"id": 1, "prediction": "..."}` or one JSON string per line.
- TXT with one sentence per line, in test-set order (pass `--order`).

## Conditions for a fair run

Use these for the headline number:

- **Zero-shot.** No task-specific fine-tuning, no retrieval, no access to
  `references.txt`.
- **Same prompt** as `prompt.txt`: `Translate the following braille to English
  text.\nBraille: {braille}\nEnglish:`
- **Greedy decoding** (temperature 0) where the API exposes it.
- **Short output.** Stop at the end of the sentence or 40 new tokens. Do not ask
  the model to explain its answer in the same field.
- **One pass.** Do not retry rows with a second prompt; report the first answer.
- **All 120 rows.** Missing rows are not scored as correct.

Optional second condition: prepend the eight pairs in `few_shot.txt` to the
prompt. Report it separately as “8-shot”; never mix it with the zero-shot score.

The BrailleLM models were fine-tuned on 480 sentences from the same generator.
A general model evaluated zero-shot is therefore at a disadvantage on *this*
distribution but has an advantage in prior knowledge and scale. State the
condition alongside the score; a single number is not a fair ranking of models.

## What the numbers mean

- **Exact** is the headline: the whole sentence must match after trimming.
  Article changes (“A” vs “The”) count as errors, as they do for BrailleLM.
- **CER** is mean Levenshtein distance over reference length. It separates a
  near miss from an unrelated sentence.
- **BLEU** is corpus-level sacreBLEU. On 120 short, templated sentences it is
  coarse; read it with the n-gram precisions in `scores_*.json`.
- **Distinct** and **vocab** expose input bypass. A model that emits the same
  few sentences for every input has low distinct count even if BLEU looks
  plausible. The 138-word reference vocabulary is closed.
- **Actor** is accuracy on word slot 2 of the 112 nine-word sentences. It is a
  cheap test of whether the model reads the actor noun.

With n = 120, a proportion near 100% has a 95% Wilson interval of roughly
±3 points; near 50%, roughly ±9 points. Do not treat a two-row difference as a
real ranking.

## Reproducing the reference rows

The BrailleLM rows come from `../results/preds_native.tsv`,
`../results/preds_baseline.tsv`, and `../results/preds_baseline_256.tsv`,
aligned to `test_set.jsonl` by braille string. `reference_scores.json` is the
scored form of those predictions. The raw rows, prototypes and vocabulary are
in `../results/comparison_pack.json`.

## Files that are not included

The base SmolLM2-135M weights and any GPU run are not in git. The merged native
checkpoint is published as a release asset; the baseline LoRA adapter is in
[`../models/baseline-lora/`](../models/baseline-lora/). See
[`../models/README.md`](../models/README.md).
