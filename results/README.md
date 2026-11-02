# Raw results

This folder holds the machine-readable output of the 120-item evaluation
described in [`../docs/RESULTS.md`](../docs/RESULTS.md).

| File | What it is |
|---|---|
| `preds_native.tsv` | Native-arm output for all 120 test items, keyed by braille input |
| `preds_baseline.tsv` | Baseline output, same keys |
| `preds_baseline_256.tsv` | Baseline with a 256-token sequence budget |
| `comparison_pack.json` | 120 rows (reference, baseline output, CER), prototypes, vocabulary, scores |
| `comparison.md` | Human-readable side-by-side comparison |
| `comparison.html` | The same comparison as a standalone page |
| `reference_scores.json` (in `../benchmark/`) | Scores for the three arms |
| `report_data.json` | Numbers and tables used to build the PDF report |
| `inference_bench.json` | CPU inference timing for native vs baseline |
| `train_bench.json` | CPU training-step timing for native vs baseline |

The `.tsv` prediction files are keyed by the **braille string**, not by test id,
because that is how the generation scripts wrote them. The scorer in
`../benchmark/score_llm.py` accepts id-keyed predictions; use the rows in
`comparison_pack.json` when you need the reference/arm mapping in one place.

Provenance: these files come from the 120-item random split of the 600-sentence
corpus, using the configuration in `docs/RESULTS.md`. They are a snapshot, not
a live leaderboard. Generated checkpoints and training logs are not tracked in
git.
