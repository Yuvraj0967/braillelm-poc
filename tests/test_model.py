"""Regression gate for the released ONNX model. CI fails if a new model is worse."""
import json

import pytest

from braillelm.braille_g1 import rule_decode, to_braille
from braillelm.paths import HELDOUT_DATA, MODEL_ONNX
from braillelm.runtime import Reader

pytestmark = pytest.mark.skipif(not MODEL_ONNX.exists(), reason="model file not present")


@pytest.fixture(scope="module")
def reader():
    return Reader()


EVERYDAY = [
    "Hello, how are you today?",
    "My name is Priya and I live in Bengaluru.",
    "Please call me at 9845012345 after 6 pm.",
    "The meeting is on 14 November at 10:30 in room 204.",
    "I can't find my keys; have you seen them?",
    "What's the wifi password?",
    "Dr Mehta's clinic opens at 9.",
    "Don't forget to buy milk, eggs and bread.",
    "Rohan's birthday is on 3 March.",
    "I'll be there in 15 minutes.",
    "Mr O'Brien won't be in today.",
    "WHO IS THE PRIME MINISTER OF INDIA?",
    "Flight AI 302 departs at 22:45 from Terminal 3.",
    "Taxol was given to Dr Okonkwo's patients in Nairobi.",
    "Colonel Fordham met Hugh Durham; the quartz waltz began at 9.",
]


@pytest.mark.parametrize("text", EVERYDAY)
def test_everyday_sentences(reader, text):
    braille = to_braille(text)
    assert rule_decode(braille) == text
    assert reader.translate(braille, "faithful")["text"] == text


def test_locked_set_gate(reader):
    rows = [json.loads(line) for line in open(HELDOUT_DATA / "final_test.jsonl", encoding="utf-8")]
    exact = sum(reader.translate(r["braille"], "faithful")["text"] == r["print"] for r in rows)
    assert exact / len(rows) >= 0.99, f"{exact}/{len(rows)}"


def test_alignment_and_confidence(reader):
    out = reader.translate(to_braille("Hello world."), "faithful")
    assert out["text"] == "Hello world."
    assert all(0 <= c["cell"] < out["cells"] and 0 < c["p"] <= 1 for c in out["chars"])
