# BrailleLM

One small neural model that reads Unified English Braille, both Grade 1
(uncontracted) and Grade 2 (contracted), into print text. It reads one cell at
a time, and a second mode uses sentence context to repair damaged cells.

Results on the locked test set of 399 unseen sentences ("exact" = the whole
sentence is right, character for character):

| Input | Grade 1 | Grade 2 |
|---|---|---|
| Clean braille | **399 / 399** (lookup table: 399) | **399 / 399** (liblouis: 399) |
| 2% of cells damaged | **358 / 399** (lookup table: 111) | **335 / 399** (liblouis: 137) |
| 5% of cells damaged | **296 / 399** (lookup table: 23) | **266 / 399** (liblouis: 40) |

On clean braille the model matches rule-based readers; on damaged braille it
is far ahead of them. On 400 unusual inputs (all capitals, made-up words,
shortened words, one to three word fragments) it reads 400 in Grade 1 and 399
in Grade 2, where liblouis back-translation reads 310. On 251 tricky inputs
(capitals that switch mid-word such as `DVDs` and `HEllo`, letter-digit codes
such as `B3`, stand-alone Grade 2 wordsigns) it reads 251 in Grade 1 and 247
in Grade 2, where liblouis reads 231.
Model: 14.4M parameters, 58 MB ONNX file, CPU inference.

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
curl -X POST http://127.0.0.1:7861/v1/translate -H "Content-Type: application/json" -d "{\"braille\": \"⠠⠮⠀⠅⠀⠷⠀⠮⠀⠏⠲\", \"grade\": \"g2\"}"
```

No FastAPI? `python -m braillelm.serving.server --open` serves the same demo
with only `numpy` and `onnxruntime` installed.

The demo is one screen: type English or paste braille and it shows the reading.
Pasted braille can be either grade; the grade is detected from the cells
(`braillelm.runtime.detect_grade`, also the default in the API). The rule was
tuned on the committed test sets, where it is right for all 2,027 full
sentences in each grade. Input of one or two words can be ambiguous (the same
two cells are `C` in Grade 1 and `Can` in Grade 2); on the 251 tricky inputs it
is wrong for 11. Typed English is converted to Grade 2 when liblouis is installed
and to Grade 1 otherwise. On Windows,
unzip the official `liblouis-3.29.0-win64.zip` release into a folder named
`liblouis-3.29.0` in your user folder and the demo finds it.

## Repository layout

```
src/braillelm/
  braille_g1.py        Grade 1 converter and rule decoder (checked against liblouis)
  louis.py             wrapper around liblouis, for Grade 2 data and the baseline
  model.py  noise.py   model definition and damage simulation (PyTorch, training only)
  runtime.py           inference with numpy + onnxruntime only
  tracking.py          optional experiment tracking (MLflow)
  cli.py               `braillelm` command
  serving/             FastAPI app, zero-dependency server, demo page
  pipelines/           build_data_g2, train, evaluate, export_onnx, make_summary
configs/               training configs (YAML)
models/braillelm_v3/   released two-grade model, model card, training log
models/braillelm_v2/   previous Grade 1 only model (3.2M parameters)
reports/               metrics and per-sentence predictions from the last evaluation
data/test_sets/        the exact test files the reported numbers were measured on
kaggle/                the GPU training job
tests/                 unit tests, API tests and model regression gates
dvc.yaml  Makefile     the pipeline, as DVC stages and as plain commands
Dockerfile             CPU serving image
.github/workflows/     CI: lint, converter check, tests, model gates
legacy/                the v1 LLM experiments
```

## Pipeline

```
check_converter -> build_data_g2 -> train -> evaluate -> export -> summary
```

The released model was produced by `kaggle/train_v3.py`, which runs these
steps on a Kaggle T4 GPU in under five hours:

```bash
kaggle kernels push -p kaggle
kaggle kernels output <user>/braillelm-v3-train -p kaggle_out
```

To run the steps yourself (needs liblouis and, realistically, a GPU):

```bash
pip install -e ".[dev,serve,train]"
make data train evaluate export summary
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
| GPU training | Kaggle | `kaggle/train_v3.py` |
| Quality gate | pytest + GitHub Actions | CI fails if the released model no longer reproduces the reported numbers, or misreads the everyday-sentence list |

The tracking backend is isolated in `src/braillelm/tracking.py`, so moving to
Weights & Biases, Comet or Neptune changes one file. Spark and dbt are not used:
the training set is 1.3 million short sentences and builds in minutes on one
machine. They become worth adding if the corpus grows to many millions of rows.

## Limits

Tested on machine-generated braille only, not on braille produced by people or
captured by devices; damage is simulated; a Grade 2 wordsign given entirely on
its own (`as`, `but`, `people`, `rather`) is sometimes misread. See the model card.
