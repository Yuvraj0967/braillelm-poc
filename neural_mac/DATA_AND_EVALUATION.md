# Dataset and evaluation protocol

## Provenance and splits

The base training corpus combines 459 pairs from the original project and
15,427 English sentences from the WikiText-2 raw training split. The English
sentences are human-written. Their Unicode Grade 1 UEB braille was produced by
liblouis 3.39.0 with `en-ueb-g1.ctb` and `unicode.dis`; only pairs that
back-translated to the same English text, ignoring case, were retained.

| File | Rows | Role |
| --- | ---: | --- |
| `data/train.jsonl` | 15,886 | Qwen fitting; 459 project + 15,427 WikiText train |
| `data/validation.jsonl` | 99 | WikiText validation; validation loss |
| `data/pilot_plain.jsonl` | 200 | WikiText test; model and decoding development |
| `data/pilot_extended.jsonl` | 199 | WikiText test; broader development |
| `data/final_test.jsonl` | 399 | WikiText test; locked post-selection evaluation |
| `data/augmented_train.jsonl` | 47,625 | Base training plus 31,739 train-only augmented pairs tried on SmolLM2; not used for the selected Qwen checkpoint |

Every row is a JSON object with `print` and `braille`. WikiText-derived rows
also record the source split and line. Direct sentence-overlap checks found
zero overlap for train versus validation, train versus either pilot, train
versus final, and the plain pilot versus final. The final set was not used to
choose the selected Qwen checkpoint or the beam width. It was evaluated after
selection and scored 216/399 strict exact (54.1%).

`code/build_wikitext_grade1.py` reconstructs the initial 2,000-candidate
WikiText sample and the pilot/final split. `code/verify_scaled_wikitext.py`
reconstructs all 15,427 round-trip-valid WikiText training rows and compares
them with rows 460 onward in the published scaled training file. The first
459 rows are the original project's training pairs.

The source is the [Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext).
Its body says CC BY-SA 4.0, while its metadata currently lists CC BY-SA 3.0
and GFDL. We preserve attribution and the link to the source card rather than
silently resolving that inconsistency. The Qwen base model is linked in
`TRAINING_AND_REPRODUCTION.md`. These pairs are derived data and are not
human-entered braille.

## Measurements

- **Strict exact match** counts a sentence only when the generated English
  string equals the reference character for character. `109/200 = 54.5%` on
  the plain development pilot. Case-insensitive normalized exact is reported
  separately in JSON; it is not the headline score.
- **CER** is Levenshtein edit distance divided by reference length, averaged
  over sentences. Lower is better.
- **sacreBLEU** is corpus-level BLEU using `sacrebleu==2.6.0`. Higher is
  better. It gives partial credit and should not be called exact translation.
- **Tokens per cell** is the average number of model tokenizer tokens per
  Unicode braille cell in each input. Lower means more compact input.
- **Four beams** means deterministic beam search with `do_sample=False` and
  `num_beams=4`; the greedy run uses one beam. Beam width was selected after
  observing development results.

The 200 plain and 199 extended pilot rows are development data, even though
their source file is the WikiText test split. The locked 399-row final file is
separate and was scored after selecting the model and beam width. The
`results/` folder contains raw metric JSON and every saved prediction so the
headline fractions can be recomputed. `code/audit_final.py` verifies row order,
hashes, the 216/399 count, CER, BLEU, and the unseen-word subgroup scores.

## Integrity identifiers

The original local files have the following SHA-256 values; paths were renamed
for publication but file bytes were copied unchanged:

```text
train:          041651d8e760d7c8da1aff0b4a693c03b04f55c5e31aaa091de28598f7cad765
validation:     370032e05ea27e7821f8da9dc953b6586419cf3f496870fe8bb7bf360c1bccfe
pilot plain:    d5075c1a549e3688359424ee8ae0cd2c2da1ba1f00d8865af591c9c78f5146db
pilot extended: a590726959e872be3e6bd4b7508567001ba8ba16ea937d3d247c7f326df5c61c
final combined: 74726e62a8e31ca396fd2b02a61230368c0f15d5963360a0deb91c85a3b48c91
braille tokens: 31ca1883938ce94ea78abd600fcce166ac17ffd97207fe8a474959f1f09de127
```

The adapter weight file has SHA-256
`ccffaa08ac5fdbb95ffbf393c447724bd9a3a133632bb60a9d7a781fb975f9b6`.
Other source hashes and the source repository commit appear in
`experiment_manifest.json`.

## Interpretation

The strongest plain pilot reached more than 50% strict exact, but the extended
pilot reached 46.7%. The pooled development result is 202/399 (50.6%). The
independent locked final result was 216/399 (54.1%). These figures show
substantial improvement over the smaller local neural runs, but represent
only synthetically converted clean Grade 1 braille. The final 95% Wilson
interval is 49.2%–59.0%, and the 263 sentences containing training-unseen
words scored 116/263 (44.1%). Human-entered braille and full Grade 2 still
need independent evaluation for a real-world claim.
