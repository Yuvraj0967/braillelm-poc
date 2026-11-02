# Brief for the comparison model

Paste this into the model you want to compare. Attach or paste the contents of
`braille_only.txt` where indicated. It contains 120 braille sentences, one per
line, in a fixed order.

---

## Task

You will translate 120 sentences from Unified English Braille (Grade 1,
uncontracted) into English print. The input is Unicode braille (U+2800 block).
Each braille cell maps to one letter, a capital sign, a period, or a word space
(U+2800). There are no contractions, digits, or other punctuation.

Use this prompt for every line:

```
Translate the following braille to English text.
Braille: <BRAILLE LINE>
English:
```

Return one line per input, numbered 1–120, in the same order as the input.
Output only the English sentence after `English:`. Do not explain, transliterate
character by character, or add notes. Keep the original capitalisation and the
final period. If a line is ambiguous, give your single best sentence; do not
offer alternatives.

## What is being measured

The 120 sentences are held out from a synthetic corpus of 600 sentences built
from a closed vocabulary of 138 words and three sentence patterns:

- 525 sentences have 9 words: article–actor–verb–article–modifier–noun–
  preposition–article–noun.
- 72 sentences have 10 words, and 3 have 11 words, with small local variants.

Examples of the distribution (training sentences, not test items):

```
A sailor polished a heavy box before the storm.
The captain closed the quiet garden around the market.
A musician prepared the empty basket in the morning.
```

The vocabulary includes 28 actors, 30 verbs, 23 modifiers, 20 objects, 15
places, 15 prepositions, 3 determiners, 3 quantifiers, and 2 person nouns.
Every word in the test set also occurs in the 480 training sentences. This is a
closed-distribution test, not open-domain braille.

## Reference points

| Model | Exact | CER | BLEU | Distinct outputs | Vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrailleLM native (fine-tuned, braille tokens) | 120 / 120 | 0.0% | 100.0 | 120 | 138 / 138 |
| BrailleLM baseline (fine-tuned, raw braille) | 0 / 120 | 63.8% | 4.0 | 9 | 22 / 138 |
| Baseline, 256-token control | 0 / 120 | 60.2% | 4.2 | 15 | 29 / 138 |

The baseline is the cautionary case: it produces fluent, grammatical sentences
that ignore the input. A high BLEU with a low distinct-output count is that
failure mode, not a near success.

Report your exact match, CER, BLEU, distinct outputs, vocabulary coverage, and
accuracy on the actor word (slot 2) of the 112 nine-word items. State whether
you ran zero-shot or with the eight examples in `few_shot.txt`, the decoding
settings, and the model version. Do not use the reference file while
translating, and do not retry individual rows.

---

## Scoring your answers

Save the numbered answers as `predictions.tsv` with two columns, `id` and text,
then run:

```
python score_llm.py predictions.tsv --name "your model"
```

The scorer aligns by `id`, applies the same trimming and metrics as the
BrailleLM report, prints a four-row comparison table, and writes per-row CERs.
With n = 120, differences of a few points are inside sampling noise; report the
condition and the confidence interval, not just the rank.
