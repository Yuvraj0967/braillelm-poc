# BrailleLM neural Mac experiment

**Capstone presentation package, 4 November 2026.** This folder documents the
locally trained neural Braille-to-English model, its data, code, checkpoints,
measured predictions, and limitations. The selected system is
`Qwen/Qwen2.5-0.5B-Instruct` with 177 added braille tokens, BKFT initialization,
LoRA, and deterministic four-beam decoding. The model adapter is distributed as
a [GitHub release asset](https://github.com/Yuvraj0967/braillelm-poc/releases/tag/neural-qwen-mac-2026-11-04);
the base Qwen weights are downloaded separately.

## Result to present

**Independent locked final result: 216/399 strict exact (54.1%)**, sacreBLEU
89.16 and mean CER 0.0231 on converter-generated Grade 1 braille. The final
set was held aside until after checkpoint and beam-width selection. Of the
263 final sentences containing at least one word absent from training,
116/263 (44.1%) were exact; on the other 136 sentences, 100/136 (73.5%) were
exact. The final result clears 50% overall, while unseen-word performance
remains below 50%. The 95% Wilson interval for the final exact-match rate is
49.2%–59.0%, so this sample does not establish a population rate above 50%.

| Evaluation | Strict exact | sacreBLEU | Mean CER | Braille tokens/cell |
| --- | ---: | ---: | ---: | ---: |
| Plain development pilot, 200 sentences, greedy | 95/200 (47.5%) | 88.25 | 0.0212 | 0.731 |
| Plain development pilot, 200 sentences, four beams | **109/200 (54.5%)** | **90.02** | **0.0178** | 0.731 |
| Extended development pilot, 199 sentences, four beams | 93/199 (46.7%) | 89.27 | 0.0253 | 0.759 |
| Both development pilots pooled | 202/399 (50.6%) | — | — | — |
| **Locked final, 399 sentences, four beams** | **216/399 (54.1%)** | **89.16** | **0.0231** | **0.748** |

The first four rows are **development** scores. The checkpoint and beam width
were selected using development results. The last row is the independent
post-selection test on 399 previously locked rows. See `results/` for the
exact JSON metrics, per-sentence predictions and an independent audit. Even
this final test is clean, converter-generated Grade 1 braille rather than
human-entered real-world braille.

## What was built

1. English sentences came from the original 459-pair project corpus and the
   WikiText-2 raw training split. Braille was generated with liblouis 3.39.0,
   `en-ueb-g1.ctb` and `unicode.dis`, then filtered by round-trip translation.
2. The original project's 177 braille token strings were added to the Qwen
   tokenizer. The vocabulary grew from 151,665 to 151,842. Each added token
   received a BKFT embedding based on the print text its cells represent;
   175 mapped and 2 used the fallback mean.
3. Qwen2.5-0.5B-Instruct was fine-tuned with answer-only cross-entropy, AdamW,
   LoRA rank 8 on `q_proj` and `v_proj`, batch size 2, gradient accumulation 8,
   learning rate 1e-4, and maximum sequence length 256. The first epoch ran
   7,943 microbatches on Apple M3 Pro MPS. Two further 4,000-microbatch
   continuations were run. The saved final adapter is the second continuation's
   step-4000 checkpoint. The first epoch took 2,142 seconds; the last
   continuation took 1,133 seconds. See `logs/` and `train_meta.json` in the
   release asset for the recorded losses, steps, and seeds.
4. The neural model was evaluated on sentence-disjoint WikiText-2 test-split
   development pilots. Strict exact match compares full strings character for
   character. BLEU and CER report partial-credit quality. Token efficiency is
   measured with the saved tokenizer.

The selected score comes from neural generation. Evaluation strips surrounding
whitespace and keeps the first generated line; it does not run liblouis or a
character-level correction rule on the model output.

This is a **reproduction and small-model extension** of ideas from
[BrailleLLM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1454/),
not a claim to have invented BKFT or braille-token fine-tuning. Our additional
controls compare standard tokenization, random initialization, and BKFT on the
same local pilot. Qwen scaling and four-beam decoding are further engineering
steps, not proof that tokenizer choice alone caused every improvement.

## What the unsuccessful runs showed

| Locally run model | Pilot strict exact | Mean CER | sacreBLEU |
| --- | ---: | ---: | ---: |
| SmolLM2 standard-tokenizer baseline | 0/200 | 0.7258 | 0.31 |
| SmolLM2 native tokens, random initialization | 0/200 | 0.7255 | 0.39 |
| SmolLM2 native tokens, fresh BKFT | 2/200 | 0.3751 | 30.08 |
| SmolLM2 native BKFT, scaled first epoch | 17/200 | 0.1614 | 61.21 |
| SmolLM2 native BKFT, scaled second-epoch interim | 20/200 | 0.1944 | 58.29 |
| SmolLM2 native BKFT, train-only augmentation | 19/200 | 0.1801 | 60.84 |
| Qwen native BKFT, first epoch, greedy | 89/200 | 0.0260 | 86.66 |
| Qwen native BKFT, second-epoch interim, greedy | 94/200 | 0.0265 | 87.24 |
| Qwen native BKFT, final continuation, greedy | 95/200 | 0.0212 | 88.25 |

The 100% score on the original small templated dataset is not a real-world
generalization result. Its narrow vocabulary and sentence patterns made it
possible to memorize. The original 6% diverse-run checkpoint was unavailable
on this Mac, so that particular two-checkpoint ensemble was not reproduced.
An earlier confidence ensemble with the available published and continued
SmolLM2 models gave 0/22 exact on a small word-disjoint check. A separate
learned Grade 1 cell-to-character transducer scored much higher on clean data;
it is **not this neural language model** and is reported separately in the
repository's Grade 1 follow-up.

## Reproduce and present

- Open [`BrailleLM_Qwen_Mac_Results.ipynb`](BrailleLM_Qwen_Mac_Results.ipynb) to
  see the measured results, plots, and real prediction examples. Its analysis
  cells were executed against the committed result files. Training itself was
  performed by the command-line scripts, so the notebook gives those commands
  for a rerun rather than inventing notebook training output.
- Read [`TRAINING_AND_REPRODUCTION.md`](TRAINING_AND_REPRODUCTION.md) for the
  stages, package versions, and exact loading/evaluation commands.
- Read [`DATA_AND_EVALUATION.md`](DATA_AND_EVALUATION.md) for split provenance,
  metric definitions, hashes, and the distinction between development and
  locked final data. `code/audit_final.py` independently recalculates the
  locked final scores from the saved hypotheses.
- Read [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, known failures, and
  the checkpoint download.

## Limits that should appear in a presentation

The English is human-written; the braille was generated by a Grade 1 converter.
There is no independent human-entered braille test here. The model has not been
validated on Grade 2 contractions, noisy input, braille OCR, Nemeth, or
assistive-technology use. Proper names and rare words remain common error
sources. liblouis is a strong rule-based baseline for clean Grade 1 and can
transcribe the generated test pairs exactly; this project studies a neural
tokenization approach, not a claim to replace liblouis.
