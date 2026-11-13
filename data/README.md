# Data

Training data is generated, not stored in git. `make data` rebuilds it (needs
liblouis: `apt-get install liblouis-bin liblouis-data`).

| Set | Rows | Source | Built by |
|---|---|---|---|
| `v3/train.jsonl` | 815,769 | WikiText-103 train split + print side of 15,886 earlier rows, each in Grade 1 and Grade 2 | `braillelm.pipelines.build_data_g2` |
| `test_sets/final_test.jsonl` | 399 | locked set from `legacy/neural_mac/data`, Grade 2 added | same |
| `test_sets/validation.jsonl`, `pilot_plain.jsonl`, `pilot_extended.jsonl` | 99, 200, 199 | development sets from `legacy/neural_mac/data`, Grade 2 added | same |
| `test_sets/digits_test.jsonl`, `apostrophe_test.jsonl` | 400, 290 | WikiText-103 test split | same |

Row format: `{"print", "braille" (Grade 1), "braille_g2" (Grade 2)}`. Test rows
also carry `louis_g2_back`, the liblouis rule-based reading of the Grade 2
braille, which is the baseline.

`test_sets/` is committed: these are the exact files the reported numbers were
measured on, produced by the Kaggle run with liblouis 3.29.0. `test_sets/meta.json`
records row counts, the seed, SHA-256 hashes and how many rows were dropped and why.

Leakage controls: only the WikiText train split is used for training, any
sentence equal to a held-out row is dropped, and the build fails if an overlap
remains.

The earlier Grade 1 only builders (`build_data`, `build_apostrophe_data`) are
kept for reproducing the v2 model.
