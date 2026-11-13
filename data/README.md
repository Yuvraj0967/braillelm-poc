# Data

Training data is generated, not stored in git. `make data` rebuilds it.

| Set | Rows | Source | Built by |
|---|---|---|---|
| `v2/train.jsonl` | 415,886 | WikiText-103 train split + 15,886 liblouis rows | `braillelm.pipelines.build_data` |
| `v2_apostrophe/train.jsonl` | 180,000 | 60,000 WikiText-103 train sentences with apostrophes rejoined + 120,000 rows sampled from the set above | `braillelm.pipelines.build_apostrophe_data` |
| `v2/digits_test.jsonl` | 400 | WikiText-103 test split, sentences with numbers | `braillelm.pipelines.build_data` |
| Held-out sets | 399 + 200 + 199 + 99 | `legacy/neural_mac/data/` (committed, liblouis-generated) | fixed |

Leakage controls: only the WikiText train split is used for training, any sentence
equal to a held-out row is dropped, and the build fails if an overlap remains.
`meta.json` in the base output directory records row counts, the seed and SHA-256
hashes of the files written.
