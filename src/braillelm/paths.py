"""Where things live in the repository.

Set BRAILLELM_HOME to point at a checkout if the package is installed somewhere
else (for example inside a container)."""
import os
from pathlib import Path

REPO = Path(os.environ.get("BRAILLELM_HOME") or Path(__file__).resolve().parents[2])
MODEL_DIR = REPO / "models" / "braillelm_v2"
MODEL_ONNX = MODEL_DIR / "braillelm_v2.onnx"
REPORTS = REPO / "reports"
LEGACY = REPO / "legacy"
HELDOUT_DATA = LEGACY / "neural_mac" / "data"          # liblouis-generated rows and the locked test set
V1_PREDICTIONS = LEGACY / "neural_mac" / "results" / "eval_qwen_final_locked_beam4_predictions.jsonl"
SERVING = Path(__file__).resolve().parent / "serving"
