# BrailleLM

A small neural model that reads Unified English Braille (Grade 1) into print
text, one cell at a time. It runs on a laptop CPU in a few milliseconds per
sentence, and a second mode uses sentence context to repair damaged cells.

| | |
|---|---|
| Unseen test sentences, clean braille | **398 / 399 exact (99.7%)** |
| Unseen sentences with apostrophes | **500 / 500 exact** |
| 2% of cells damaged, repair mode | 279 / 399 exact (69.9%); a lookup table gets 27.8% |
| Model | 3.2M parameters, 13 MB ONNX file, CPU only |

Details, limits and how these were measured: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
The earlier LLM-based proof of concept is preserved unchanged in [`legacy/`](legacy/README.md).

## Quick start

```bash
pip install -e ".[serve]"
uvicorn braillelm.serving.api:app --port 7861
```

Open http://127.0.0.1:7861 for the demo, or http://127.0.0.1:7861/docs for the API.

```bash
braillelm --english "Dr Mehta's clinic opens at 9."
```

```bash
curl -X POST http://127.0.0.1:7861/v1/translate -H "Content-Type: application/json" -d "{\"braille\": \"⠠⠓⠑⠇⠇⠕⠲\"}"
```

No FastAPI? `python -m braillelm.serving.server --open` serves the same demo
with only `numpy` and `onnxruntime` installed.

## Repository layout

```
src/braillelm/
  braille_g1.py        Grade 1 converter and rule decoder (checked against liblouis)
  model.py  noise.py   model definition and damage simulation (PyTorch, training only)
  runtime.py           inference with numpy + onnxruntime only
  tracking.py          optional experiment tracking (MLflow)
  cli.py               `braillelm` command
  serving/             FastAPI app, zero-dependency server, demo page
  pipelines/           build_data, train, evaluate, export_onnx, make_summary
configs/               training configs (YAML)
models/braillelm_v2/   released model, checkpoint and model card
reports/               metrics and per-sentence predictions from the last evaluation
data/                  generated data (not in git; see data/README.md)
tests/                 unit tests, API tests and the model regression gate
dvc.yaml  Makefile     the pipeline, as DVC stages and as plain commands
Dockerfile             CPU serving image
.github/workflows/     CI: lint, converter check, tests, model gate
legacy/                the v1 LLM experiments
```

## Pipeline

```
check_converter -> build_data -> train -> finetune_apostrophe -> evaluate -> export -> summary
```

```bash
pip install -e ".[dev,serve,train]"
make data          # or: dvc repro build_apostrophe_data
make train finetune
make evaluate export summary
make test
```

Each step is a module under `braillelm.pipelines` and can be run on its own.
Training options live in `configs/*.yaml`; command-line flags override the file.

## Tooling

Every tool here is optional and sits outside the model code.

| Need | Tool | How it is used |
|---|---|---|
| Serving | FastAPI + Uvicorn, Docker | `braillelm.serving.api`, `Dockerfile` |
| Experiment tracking and model registry | MLflow | add `--mlflow <experiment>` to `train` or `export_onnx`; view with `mlflow ui` |
| Data and pipeline versioning | DVC | `dvc.yaml`; `dvc repro` rebuilds only what changed |
| Quality gate | pytest + GitHub Actions | CI fails if the released model drops below 99% on the locked set or misreads the everyday-sentence list |

The tracking backend is isolated in `src/braillelm/tracking.py`, so moving to
Weights & Biases, Comet or Neptune changes one file. Spark and dbt are not used:
the training set is 416,000 short sentences and builds in about a minute on one
machine. They become worth adding if the corpus grows to many millions of rows.

## Extending to Grade 2 or a larger model

- **Grade 2 data:** add a data stage that writes `data/g2/` in the same
  `{"print", "braille"}` JSONL format (for example generated with liblouis
  `en-ueb-g2.ctb`). Training, evaluation and export take a data directory and
  need no changes.
- **Model size:** `layers`, `d`, `heads` and `ff` are config values. The ONNX
  export and the runtime read the shape from the checkpoint.
- **Output length:** the model emits up to two print characters per cell
  (`FRAMES_PER_CELL` in `model.py`). Grade 2 contractions expand to more, so
  raise this constant (and the matching `f // 2` in `runtime.py`) when training on Grade 2.
- **GPU:** training is plain PyTorch and currently runs on CPU; moving tensors
  to a device is the only change needed for a GPU.

## Limits

Grade 1 only; tested on converter-generated braille, not on braille produced by
people or devices; damage is simulated. See the model card.
