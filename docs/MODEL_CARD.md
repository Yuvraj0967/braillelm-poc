# Model card: BrailleLM v3.2

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
| Data | 1,307,142 sentences, each rendered in both grades |
| Text source | WikiText-103 train split (apostrophes rejoined) + the print side of 15,886 earlier rows |
| Added variations | 491,391 of the rows are rewrites of training sentences: all capitals (119,972), title case (59,751), all lower case (59,494), one or two words swapped for a made-up, vowel-dropped or oddly capitalised word (58,390), one to three word fragments (43,845), and, new in v3.2, words whose capitals switch partway such as `DVDs`, `DVD's` and `HEllo` (60,007), letter-digit codes such as `B3` and `3rd` (51,939), and short fragments built around a stand-alone Grade 2 wordsign (37,993). See `src/braillelm/augment.py` |
| Grade 2 braille | liblouis 3.29.0, table `en-ueb-g2.ctb` |
| Grade 1 braille | this repository's converter; ordinary sentences where it differed from liblouis `en-ueb-g1.ctb` were dropped (67). All-capital rows are kept in both the converter's and liblouis's Grade 1 form |
| Mix | each row shown as Grade 2 with probability 0.6, in repair mode with probability 0.6 |
| Schedule | 48,000 steps, mixed precision, `configs/train_g2.yaml` |
| Hardware and time | Kaggle Tesla T4, 271 minutes of training (287 minutes including data build and evaluation) |
| Checkpoint | step 48,000, chosen on development data only |

The run is `kaggle/train_v3.py`; its training curve is in
`models/braillelm_v3/train_log.json`.

## Results

"Exact" means the whole sentence matches character for character. The
rule-based baseline is a lookup table for Grade 1 and liblouis
back-translation for Grade 2, each given exactly the same cells as the model.

### Locked final set (399 unseen sentences)

| Input | Mode | Grade 1 model | Grade 1 lookup table | Grade 2 model | Grade 2 liblouis |
|---|---|---|---|---|---|
| Clean | faithful | 399 (100%) | 399 | 399 (100%) | 399 |
| Clean | repair | 397 (99.5%) | 399 | 390 (97.7%) | 399 |
| 2% of cells damaged | repair | 358 (89.7%) | 111 (27.8%) | 335 (84.0%) | 137 (34.3%) |
| 5% of cells damaged | repair | 296 (74.2%) | 23 (5.8%) | 266 (66.7%) | 40 (10.0%) |

The 95% interval for 399 of 399 is 99.0 to 100%, so this does not show the
model is free of errors. Each damaged cell has one dot added or missing. On
the 55 sentences containing a word absent from training, the model is exact
on 55 in both grades.

### Other held-out sets, clean braille, faithful mode

| Test set | Sentences | Grade 1 model | Grade 2 model | Grade 2 liblouis |
|---|---|---|---|---|
| Pilot plain | 200 | 200 | 200 | 200 |
| Pilot extended | 199 | 199 | 199 | 198 |
| Sentences with numbers | 400 | 400 | 400 | 398 |
| Sentences with apostrophes | 290 | 290 | 289 | 288 |
| Everyday sentences written by hand | 40 | 40 | 40 | 39 |
| Unusual text (see below) | 400 | 400 | 399 | 310 |
| Tricky text (see below) | 251 | 251 | 247 | 231 |

### Unusual text

400 sentences from the WikiText test split, 80 rewritten in each of five
ways. The release before the rewrites were added to training (v3) is shown
because this set was built to measure a weakness found in it. v3.1 scored
the same totals as v3.2.

| Kind | Grade 1 v3.2 | Grade 1 v3 | Grade 2 v3.2 | Grade 2 v3 | Grade 2 liblouis |
|---|---|---|---|---|---|
| All capitals | 80 | 65 | 80 | 0 | 0 |
| Title case | 80 | 80 | 80 | 80 | 71 |
| All lower case | 80 | 80 | 80 | 78 | 79 |
| Made-up, shortened or oddly capitalised words | 80 | 80 | 80 | 68 | 80 |
| One to three word fragments | 80 | 80 | 79 | 62 | 80 |
| Total | 400 | 385 | 399 | 288 | 310 |

The test sentences come from a different split than the training text, but
the five kinds of rewriting are the same ones added to training, so this
measures whether the weakness was fixed, not how the model handles kinds of
text it has never been shown.

### Tricky text

251 inputs built to measure the errors found in v3.1: sentences from the
WikiText test split rewritten in three more ways, plus 24 lines written by
hand (`data/tricky_sentences.txt`). v3.1 is the released v3.1 file scored on
the same rows.

| Kind | Inputs | Grade 1 v3.2 | Grade 1 v3.1 | Grade 2 v3.2 | Grade 2 v3.1 | Grade 2 liblouis |
|---|---|---|---|---|---|---|
| Capitals that switch inside a word (`DVDs`, `HEllo`) | 80 | 80 | 30 | 80 | 27 | 61 |
| Letter-digit codes (`B3`, `42a`, `3rd`) | 67 | 67 | 37 | 67 | 26 | 67 |
| Fragments around a stand-alone wordsign | 80 | 80 | 80 | 78 | 68 | 80 |
| Written by hand | 24 | 24 | 13 | 22 | 8 | 23 |
| Total | 251 | 251 | 160 | 247 | 129 | 231 |

As with the unusual-text set, the three kinds of rewriting are the ones added
to training, and the hand-written lines were chosen from v3.1's known errors,
so this shows those errors were fixed and nothing more general. The four
remaining Grade 2 errors are all a wordsign standing alone; liblouis reads
those correctly.

### What these numbers say

- **Clean braille:** the model equals the rule-based readers on ordinary
  sentences. It does not beat them there, and on clean input a rule-based
  reader is a sound choice.
- **Damaged braille:** the model reads two to three times as many sentences
  correctly at 2% damage and six to thirteen times as many at 5%.
- **Compared with v3.1 (the previous release):** 399 instead of 398 clean in
  both grades, tricky text up from 160 to 251 (Grade 1) and from 129 to 247
  (Grade 2) of 251, and repair about level: at 2% damage 358 and 335 instead
  of 349 and 330; at 5% damage 296 instead of 276 in Grade 1 but 266 instead
  of 271 in Grade 2. Differences of a few sentences under damage are within
  run-to-run variation.

## How the locked set was used

The locked final set was never used for training or checkpoint selection. This
model was evaluated on it once. Its predecessors v3 and v3.1 were each also
evaluated on it once (398 of 399 in both grades), and earlier Grade 1 only
models three times (399, 399 and 398 of 399). So the set has now been looked at six times across
model versions; no change was made in response to its sentences, but it is
no longer a first-look test.

The 40 everyday sentences are weaker evidence than they were: v3's four
Grade 2 errors on them are what prompted the added variations, so v3.2's 40 of
40 on that set is partly a fix aimed at those sentences. The sentences
themselves were not added to training. The released ONNX file reproduces the clean-input counts above, and
a CI test checks this on every push.

## Known errors

- A Grade 2 wordsign given entirely on its own, with no other word, is
  sometimes read as its letter: `as` as `z`, `but` as `bt`, `people` as `p`,
  `rather` as `r`. Inside a sentence these words are read correctly.
- One of the 290 apostrophe sentences is wrong in Grade 2 (`characterless`
  read as `characteress`); liblouis misses two.
- v3.1's errors on `IRAs`, `DVDs`, `HEllo` and `B3` are fixed.
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
  390 rather than 399. Use faithful mode unless the input is known to be noisy.
- **One domain.** Wikipedia prose and mechanical rewrites of it. Other styles
  were checked only with the 40 everyday sentences.
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
