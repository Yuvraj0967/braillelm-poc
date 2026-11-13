# Model card: BrailleLM v3

## What it is

One bidirectional Transformer encoder that reads Unified English Braille into
print text, in Grade 1 (uncontracted) and Grade 2 (contracted).

- **Input:** one token per six-dot cell (64 symbols). No subword merging, so the
  model cannot skip or invent cells.
- **Output:** CTC over print characters, eight frames per cell. A Grade 2 cell
  can stand for several letters or a whole word (`⠅` alone is `knowledge`), so
  each cell may emit up to eight characters. Every output character has a
  source cell and a confidence.
- **Two learned switches:** *grade* (Grade 1 or Grade 2) and *mode*. Faithful
  mode (trained on clean braille) trusts every dot. Repair mode (trained with
  dot flips, wrong cells and missing cells in 0.5 to 8% of cells) uses context
  to fix damage.
- **Size:** 8 layers, width 384, 6 heads, rotary positions; 14.4M parameters.

## Training

| | |
|---|---|
| Data | 815,769 sentences, each rendered in both grades |
| Text source | WikiText-103 train split (apostrophes rejoined) + the print side of 15,886 earlier rows |
| Grade 2 braille | liblouis 3.29.0, table `en-ueb-g2.ctb` |
| Grade 1 braille | this repository's converter; the 49 sentences where it differed from liblouis `en-ueb-g1.ctb` were dropped |
| Mix | each row shown as Grade 2 with probability 0.6, in repair mode with probability 0.6 |
| Schedule | 20,000 steps, mixed precision, `configs/train_g2.yaml` |
| Hardware and time | Kaggle Tesla T4, 115 minutes of training (127 minutes including data build and evaluation) |
| Checkpoint | step 18,000, chosen on development data only |

The run is `kaggle/train_v3.py`; its training curve is in
`models/braillelm_v3/train_log.json`.

## Results

"Exact" means the whole sentence matches character for character. The
rule-based baseline is a lookup table for Grade 1 and liblouis
back-translation for Grade 2, each given exactly the same cells as the model.

### Locked final set (399 unseen sentences)

| Input | Mode | Grade 1 model | Grade 1 lookup table | Grade 2 model | Grade 2 liblouis |
|---|---|---|---|---|---|
| Clean | faithful | 398 (99.7%) | 399 | 398 (99.7%) | 399 |
| Clean | repair | 394 (98.7%) | 399 | 382 (95.7%) | 399 |
| 2% of cells damaged | repair | 346 (86.7%) | 111 (27.8%) | 312 (78.2%) | 137 (34.3%) |
| 5% of cells damaged | repair | 258 (64.7%) | 23 (5.8%) | 240 (60.2%) | 40 (10.0%) |

The 95% interval for 398 of 399 is 98.6 to 100%. Each damaged cell has one dot
added or missing. On the 61 sentences containing a word absent from training,
the model is exact on 61 in both grades.

### Other held-out sets, clean braille, faithful mode

| Test set | Sentences | Grade 1 model | Grade 2 model | Grade 2 liblouis |
|---|---|---|---|---|
| Pilot plain | 200 | 200 | 200 | 200 |
| Pilot extended | 199 | 199 | 199 | 198 |
| Sentences with numbers | 400 | 400 | 400 | 398 |
| Sentences with apostrophes | 290 | 290 | 289 | 288 |
| Everyday sentences written by hand | 40 | 40 | 36 | 39 |

### What these numbers say

- **Clean braille:** the model equals the rule-based readers to within one
  sentence. It does not beat them, and on clean input a rule-based reader is a
  sound choice.
- **Damaged braille:** the model reads two to three times as many sentences
  correctly at 2% damage and six to eleven times as many at 5%.
- **Compared with the previous Grade 1 model (v2.1):** the same clean score,
  and 346 instead of 279 sentences at 2% damage.

## How the locked set was used

The locked final set was never used for training or checkpoint selection. This
model was evaluated on it once. Earlier Grade 1 only models were evaluated on
the Grade 1 version three times (399, 399 and 398 of 399); those models are
superseded. The released ONNX file reproduces the clean-input counts above, and
a CI test checks this on every push.

## Known errors

- `IRAs` is read as `IRA's` (Grade 2) or `IRAU's` (Grade 1): a capitals
  terminator followed by a lowercase letter.
- On the 40 everyday sentences in Grade 2 the model missed four that liblouis
  read correctly: slang (`tmrw` became `tmoorw`), a sentence in all capitals,
  made-up words (`xyz` became `xyzs`) and a brand name with a capital inside
  (`OnePlus` became `OneClus`). Grade 2 reading leans on having seen similar
  words, so unusual spellings are its weak point.
- liblouis has its own errors on the same kind of input, for example `U.S.`
  read as `U.So.` and `H.D.` as `H.Do.`; the model read those correctly.

## Limits

- **Machine-generated braille.** All training and test braille comes from
  liblouis or a converter that matches it. Not tested on braille typed or
  embossed by people, or captured by a camera or scanner.
- **Simulated damage.** Random single-dot errors, not errors measured from a
  real device.
- **Clean input is not where the model adds value.** Rule-based readers are as
  accurate there, and faster.
- **Repair mode changes some correct words.** On clean Grade 2 input it scores
  382 rather than 398. Use faithful mode unless the input is known to be noisy.
- **One domain.** Wikipedia prose. Other styles were checked only with the 40
  everyday sentences, where Grade 2 was 36 of 40.
- **Character set:** letters, digits and `. , ; : ? ! ' -`. Other characters
  are rejected by the service.
- **One liblouis version.** Grade 2 follows liblouis 3.29.0's reading of UEB.
- **Speed:** about 1 ms per sentence on the T4 GPU; about 120 ms per sentence
  on the development laptop (x64 Python emulated on an ARM CPU). Not measured
  on other CPUs.

## Data and leakage controls

- Only the WikiText-103 train split is used for training.
- Sentences equal to any held-out row are removed, and the build asserts that
  no overlap remains.
- The exact test files are in `data/test_sets/`, with the liblouis baseline
  reading stored beside each row; `meta.json` there records row counts, hashes
  and the liblouis version.

## Files

`models/braillelm_v3/braillelm_v3.onnx` (inference), `model_card.json`
(config, training arguments, development scores) and `train_log.json`. The
PyTorch weights for further training are in the Kaggle run's output and are
not stored in git because of their size.
