# Model card: BrailleLM v2.1

## What it is

A bidirectional Transformer encoder that reads Unified English Braille, Grade 1
(uncontracted), into print text.

- **Input:** one token per six-dot cell (64 symbols). No subword merging, so the
  model cannot skip or invent cells.
- **Output:** CTC over print characters, two frames per cell, so a cell can
  produce zero, one or two characters. This covers the capital sign, number
  sign and spaces without a decoder and gives every output character a source
  cell and a confidence.
- **Size:** 4 layers, width 256, 4 heads, rotary positions; 3.2M parameters;
  13 MB as ONNX.
- **Two modes, one model:** *faithful* (trained on clean braille) trusts every
  dot. *Repair* (trained with dot flips, wrong cells and missing cells in 0.5 to
  8% of cells) uses context to fix damage.

## Training

| Stage | Data | Steps | Config |
|---|---|---|---|
| Base | 415,886 sentences: WikiText-103 train split + 15,886 liblouis rows | 5,000 | `configs/train.yaml` |
| Apostrophe fine-tune | 60,000 WikiText-103 train sentences with apostrophes + 120,000 base rows | 800 | `configs/finetune_apostrophe.yaml` |

The base run was interrupted at step 3,500 and resumed from those weights with
a fresh optimiser. Total training time was about three hours on an 8-core
laptop CPU. Checkpoints were chosen on development data only.

The fine-tune exists because the first data build dropped every sentence with
an apostrophe, so the base model misread all of them (`can't` became `cankt`).
The locked test set has no apostrophes and did not reveal this; a hand-written
list of everyday sentences did. That list is now a CI test.

## Results

"Exact" means the whole sentence matches character for character. The rule
decoder is a lookup table with no context.

### Clean braille, faithful mode

| Test set | Sentences | Model exact | Rule decoder exact |
|---|---|---|---|
| Locked final set (`legacy/neural_mac/data/final_test.jsonl`) | 399 | 398 (99.7%, 95% interval 98.6 to 100%) | 399 |
| Pilot plain | 200 | 200 | 200 |
| Pilot extended | 199 | 199 | 199 |
| Sentences with numbers | 400 | 399 | 400 |
| Sentences with apostrophes | 500 | 500 | 500 |
| Everyday sentences written by hand | 40 | 40 | 40 |

The v1 LLM scored 216 of 399 (54.1%) on the locked final set.

### Damaged braille, repair mode (locked final set)

Each damaged cell has one dot added or missing.

| Input | Model exact | Rule decoder exact |
|---|---|---|
| 2% of cells damaged | 279 (69.9%) | 111 (27.8%) |
| 5% of cells damaged | 174 (43.6%) | 23 (5.8%) |
| Clean input in repair mode | 382 (95.7%) | 399 |

Repair mode costs accuracy on clean input because it sometimes changes a rare
word that was already right. Use faithful mode unless the input is known to be
noisy.

## How the locked set was used

The locked final set was never used for training or checkpoint selection. It
has been evaluated three times, each on a checkpoint already chosen on
development data:

| Checkpoint | Clean exact |
|---|---|
| Base, step 3,500 | 399 / 399 |
| Base, step 5,000 | 399 / 399 |
| After apostrophe fine-tune (released) | 398 / 399 |

The released model is the last row. The fine-tune fixed apostrophes and
introduced one error on this set, described below.

## Known errors

- `IRAs` is read as `IRA's`. The capitals terminator before the `s` shares a
  cell with the apostrophe, and after the fine-tune the model sometimes emits
  an apostrophe there.
- `WWE.com` is read as `WWE.COM` (the one miss in the numbers set): a capitals
  terminator followed by a full stop inside a word.

## Limits

- **Grade 1 only.** Most real English braille is Grade 2 (contracted). The
  model has never seen a contraction and cannot read Grade 2.
- **Converter-generated braille.** Not tested on braille typed or embossed by
  people, or captured by a camera or scanner.
- **Simulated damage.** Random single-dot errors, not errors measured from a
  real device.
- **Clean Grade 1 is deterministic.** A lookup table also reads it perfectly.
  The clean-input scores show that the model is faithful, not that the task is
  hard; the value of the model is repair, and its path to Grade 2.
- **Character set:** letters, digits and `. , ; : ? ! ' -`. Other characters
  are rejected by the service.
- **English prose from Wikipedia.** Other styles of text were checked only with
  the 40 everyday sentences.
- Number handling follows UEB rules but has not been checked against liblouis,
  because the liblouis-generated data contains no digits.

## Data and leakage controls

- Only the WikiText-103 train split is used for training.
- Sentences equal to any held-out row are removed, and the build asserts that
  no overlap remains.
- The converter matches all 16,783 liblouis-generated rows in this repository
  with zero mismatches; CI re-checks this on every push.

## Files

`models/braillelm_v2/braillelm_v2.onnx` (inference),
`braillelm_v2.pt` (PyTorch weights for further training),
`model_card.json` (config, training arguments and development scores).
