"""Torch-free inference for BrailleLM v2 (numpy + onnxruntime only).

This is the module a product would embed: it loads the exported ONNX model and
turns Unicode braille into print text, with a confidence and a source cell for
every output character.
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from braillelm.braille_g1 import PRINT_CHARS, PUNCT
from braillelm.paths import MODEL_ONNX

MODES = ("faithful", "repair")
GRADES = ("g1", "g2")
BLANK_CELL = chr(0x2800)
MAX_CHUNK = 110          # cells per forward pass; training sentences are 20-150 cells
_SENT_END = {PUNCT['.'], PUNCT['?'], PUNCT['!']}


def normalise(braille):
    """Keep six-dot cells, map ASCII whitespace to the blank cell."""
    out = []
    for ch in braille:
        o = ord(ch)
        if 0x2800 <= o <= 0x283F:
            out.append(ch)
        elif ch.isspace():
            out.append(BLANK_CELL)
    s = ''.join(out).strip(BLANK_CELL)
    while BLANK_CELL * 2 in s:
        s = s.replace(BLANK_CELL * 2, BLANK_CELL)
    return s


def chunk(braille):
    """Split long input at word boundaries, preferring sentence ends.
    Returns [(start_index, text)]."""
    words, pos = [], 0
    for w in braille.split(BLANK_CELL):
        words.append((pos, w))
        pos += len(w) + 1
    chunks, cur = [], []
    for pos, w in words:
        length = (pos + len(w)) - cur[0][0] if cur else len(w)
        if cur and length > MAX_CHUNK:
            chunks.append(cur)
            cur = []
        cur.append((pos, w))
        if w and w[-1] in _SENT_END and (pos + len(w)) - cur[0][0] > MAX_CHUNK // 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return [(c[0][0], BLANK_CELL.join(w for _, w in c)) for c in chunks]


class Reader:
    def __init__(self, onnx_path=None):
        onnx_path = Path(onnx_path or MODEL_ONNX)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        # A two-grade model has a third input; a Grade 1 model reads only "g1".
        self.grades = GRADES if "grade" in {i.name for i in self.session.get_inputs()} else GRADES[:1]
        self.frames = self._logits("⠁", "faithful", "g1").shape[0]      # CTC frames per cell

    def _logits(self, braille, mode, grade):
        ids = np.array([[ord(c) - 0x2800 for c in braille]], dtype=np.int64)
        feed = {"cells": ids, "mode": np.array([MODES.index(mode)], dtype=np.int64)}
        if len(self.grades) > 1:
            feed["grade"] = np.array([GRADES.index(grade)], dtype=np.int64)
        return self.session.run(None, feed)[0][0]

    def _run(self, braille, mode, grade):
        logits = self._logits(braille, mode, grade)
        logits = logits - logits.max(-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(-1, keepdims=True)
        return p.argmax(-1), p.max(-1)

    def translate(self, braille, mode="faithful", grade="g1"):
        """mode "faithful" trusts every dot; "repair" may fix damaged cells.
        grade "g1" is uncontracted braille, "g2" contracted.
        Returns {"text", "chars": [{"c", "p", "cell"}], "ms", "cells"}."""
        if grade not in self.grades:
            raise ValueError(f"this model does not read {grade}")
        braille = normalise(braille)
        t0 = time.perf_counter()
        chars = []
        for start, piece in chunk(braille) if braille else []:
            if chars:
                chars.append({"c": " ", "p": 1.0, "cell": start - 1})
            ids, prob = self._run(piece, mode, grade)
            prev = 0
            for f, (i, p) in enumerate(zip(ids.tolist(), prob.tolist())):
                if i != prev and i != 0:
                    chars.append({"c": PRINT_CHARS[i - 1], "p": round(p, 4), "cell": start + f // self.frames})
                prev = i
        return {"text": ''.join(c["c"] for c in chars), "chars": chars, "cells": len(braille),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "braille": braille, "grade": grade}
