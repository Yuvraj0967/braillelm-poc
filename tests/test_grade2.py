"""Grade 2 regression gate. Skipped until a two-grade model is released in models/braillelm_v3."""
import json

import pytest

from braillelm.paths import MODEL_ONNX, REPO, REPORTS
from braillelm.runtime import Reader

TEST_SETS = REPO / "data" / "test_sets"


def _two_grade():
    return MODEL_ONNX.exists() and "braillelm_v3" in MODEL_ONNX.name


pytestmark = pytest.mark.skipif(not _two_grade(), reason="no two-grade model released")


@pytest.fixture(scope="module")
def reader():
    return Reader()


def _rows(name):
    return [json.loads(line) for line in open(TEST_SETS / f"{name}.jsonl", encoding="utf-8")]


def test_model_reads_both_grades(reader):
    assert reader.grades == ("g1", "g2")


def test_reported_numbers_match_the_released_model(reader):
    """The ONNX file in the repo reproduces the clean-input numbers in reports/metrics.json."""
    metrics = json.loads((REPORTS / "metrics.json").read_text())["final_test"]
    rows = _rows("final_test")
    for grade, key in (("g1", "braille"), ("g2", "braille_g2")):
        exact = sum(reader.translate(r[key], "faithful", grade)["text"] == r["print"] for r in rows)
        assert abs(exact - metrics[grade]["clean"]["neural"]["exact"]) <= 1, (grade, exact)


def test_grade2_gate(reader):
    rows = _rows("final_test")
    exact = sum(reader.translate(r["braille_g2"], "faithful", "g2")["text"] == r["print"] for r in rows)
    assert exact / len(rows) >= 0.95, f"{exact}/{len(rows)}"


def test_grade1_not_regressed(reader):
    rows = _rows("final_test")
    exact = sum(reader.translate(r["braille"], "faithful", "g1")["text"] == r["print"] for r in rows)
    assert exact / len(rows) >= 0.99, f"{exact}/{len(rows)}"
