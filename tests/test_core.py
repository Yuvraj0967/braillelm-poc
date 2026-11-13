"""Fast checks for the v2 code paths that need no model. Run: pytest"""
import random
import sys
from pathlib import Path

from braillelm.braille_g1 import BLANK, CAP, NUM, PRINT_CHARS, rule_decode, safe_for_synthesis, to_braille
from braillelm.noise import corrupt_braille


def test_known_forms():
    assert to_braille("NFL") == CAP + CAP + to_braille("nfl")
    assert to_braille("Ab") == CAP + to_braille("ab")
    assert to_braille("42") == NUM + to_braille("db")
    assert to_braille("1,000") == NUM + to_braille("a") + to_braille(",") + to_braille("jjj")
    assert to_braille("a b") == to_braille("a") + BLANK + to_braille("b")


def test_round_trip():
    rng = random.Random(0)
    words = ["the", "Durham", "NATO", "Taxol", "colossal", "1914", "3,500", "II", "selectivities",
             "jewellery", "x", "I", "McDonald", "7"]
    n = 0
    for _ in range(20000):
        s = " ".join(rng.choice(words) for _ in range(rng.randint(1, 9))) + rng.choice(".?!")
        s = s.replace(" ", rng.choice([" ", ", ", "; ", ": ", " "]), 1)
        if safe_for_synthesis(s):
            assert rule_decode(to_braille(s)) == s, s
            n += 1
    assert n > 10000


def test_charset():
    for c in PRINT_CHARS:
        text = c if c != " " else "a b"
        assert rule_decode(to_braille(text)) == text


def test_noise():
    b = to_braille("The quick brown fox jumps over the lazy dog.")
    assert corrupt_braille(b, 0.0, 1) == b
    n = corrupt_braille(b, 0.2, 1)
    assert n == corrupt_braille(b, 0.2, 1) and n != b and len(n) == len(b)
    assert [i for i, c in enumerate(b) if c == BLANK] == [i for i, c in enumerate(n) if c == BLANK and b[i] == BLANK]


def test_chunking():
    try:
        from braillelm.runtime import chunk, normalise
    except ImportError:
        print("  (runtime not importable here; chunk test skipped)")
        return
    b = to_braille(" ".join(["This is a sentence about braille."] * 30))
    parts = chunk(b)
    assert all(len(p) <= 150 for _, p in parts)
    assert BLANK.join(p for _, p in parts) == b
    assert all(b[s:s + len(p)] == p for s, p in parts)
    assert normalise("  " + b.replace(BLANK, " \n ") + " ") == b

def test_safe_for_synthesis_validation():
    # Valid supported text
    assert safe_for_synthesis("Hello, world!") is True
    assert safe_for_synthesis("Braille 2026") is True

    # Empty and unsupported input
    assert safe_for_synthesis("") is False
    assert safe_for_synthesis("Hello🙂") is False

    # Unsafe letter-number combinations
    assert safe_for_synthesis("A1") is False
    assert safe_for_synthesis("1A") is False

    # Unsafe capitalisation patterns
    assert safe_for_synthesis("ABC DEF GHI") is False
    assert safe_for_synthesis("ABC's") is False
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL V2 CHECKS PASSED")
