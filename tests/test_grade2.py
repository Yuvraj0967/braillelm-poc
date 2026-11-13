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


@pytest.fixture(scope="module")
def locked(reader):
    """Exact-match counts on the locked final set, per grade, from the released ONNX file."""
    rows = _rows("final_test")
    return {grade: sum(reader.translate(r[key], "faithful", grade)["text"] == r["print"] for r in rows)
            for grade, key in (("g1", "braille"), ("g2", "braille_g2"))}, len(rows)


def test_reported_numbers_match_the_released_model(locked):
    """The ONNX file in the repo reproduces the clean-input numbers in reports/metrics.json."""
    metrics = json.loads((REPORTS / "metrics.json").read_text())["final_test"]
    for grade, exact in locked[0].items():
        assert exact == metrics[grade]["clean"]["neural"]["exact"], (grade, exact)


def test_grade2_gate(locked):
    assert locked[0]["g2"] / locked[1] >= 0.98, locked


def test_grade1_not_regressed(locked):
    assert locked[0]["g1"] / locked[1] >= 0.99, locked


def test_one_cell_words_do_not_split_sentences(reader):
    """Regression: a lone Grade 2 cell ("his") was once mistaken for a question mark ending a sentence."""
    rows = [r for r in _rows("final_test") if " his " in r["print"] and len(r["braille_g2"]) > 80][:5]
    assert rows
    for r in rows:
        assert reader.translate(r["braille_g2"], "faithful", "g2")["text"] == r["print"]


def test_grade_is_detected_from_the_cells():
    """The demo and API need no grade switch: every committed test sentence is told apart correctly."""
    import json

    from braillelm.paths import REPO
    from braillelm.runtime import detect_grade
    seen = 0
    for path in sorted((REPO / "data" / "test_sets").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            assert detect_grade(r["braille"]) == "g1", (path.name, r["print"])
            if r["braille_g2"] != r["braille"]:
                assert detect_grade(r["braille_g2"]) == "g2", (path.name, r["print"])
            seen += 1
    assert seen > 2000


def test_demo_reads_pasted_braille_without_being_told_the_grade():
    from braillelm.paths import MODEL_ONNX
    from braillelm.serving.server import Engine
    eng = Engine(MODEL_ONNX)
    for braille, grade, text in (("⠠⠮⠀⠅⠀⠷⠀⠮⠀⠏⠲", "g2", "The knowledge of the people."),
                                 ("⠠⠞⠓⠑⠀⠉⠁⠞⠀⠎⠁⠞⠲", "g1", "The cat sat.")):
        out = eng.translate({"braille": braille})
        assert out["grade"] == grade and out["faithful"]["text"] == text, out["faithful"]["text"]
