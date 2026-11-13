"""Local demo for BrailleLM v2.

    python -m braillelm.serving.server --open

Needs only numpy and onnxruntime (no PyTorch, no network, no API key). The
page lets you paste braille or type English, optionally damages the braille
with single-dot errors, and shows the model's two readings (faithful and
repair) next to a deterministic rule decoder on exactly the same cells.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

from braillelm.braille_g1 import (BLANK, CAP, NUM, CELL_TO_LETTER, CELL_TO_PUNCT, SUPPORTED,
                        rule_decode, to_braille)
from braillelm import louis
from braillelm.paths import MODEL_ONNX, REPORTS
from braillelm.runtime import Reader, normalise, detect_grade

MAX_CELLS = 2000


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def damage(braille, rate, seed):
    """Flip one dot in a share of the non-blank cells. Returns (text, changed indexes)."""
    rng = random.Random(seed)
    out, changed = [], []
    for i, ch in enumerate(braille):
        if ch != BLANK and rng.random() < rate:
            out.append(chr(0x2800 + ((ord(ch) - 0x2800) ^ (1 << rng.randrange(6)))))
            changed.append(i)
        else:
            out.append(ch)
    return ''.join(out), changed


def describe(braille, changed, grade="g1"):
    changed = set(changed)
    cells = []
    for i, ch in enumerate(braille):
        v = ord(ch) - 0x2800
        if ch == BLANK or grade != "g1":      # a Grade 2 cell has no single meaning out of context
            label = ""
        elif ch == CAP:
            label = "cap"
        elif ch == NUM:
            label = "num"
        else:
            label = CELL_TO_LETTER.get(ch) or CELL_TO_PUNCT.get(ch) or "?"
        cells.append({"dots": [d + 1 for d in range(6) if v & (1 << d)], "label": label,
                      "space": ch == BLANK, "damaged": i in changed})
    return cells


class Engine:
    def __init__(self, model_path):
        self.reader = Reader(model_path)
        self.lock = threading.Lock()
        self.samples = read_json(HERE / "samples.json", [])
        self.metrics = read_json(REPORTS / "summary.json", {})

    def info(self):
        """What this server can do, for the demo page."""
        return {"grades": list(self.reader.grades), "english_to_g2": louis.available()}

    def baseline(self, grade, shown, clean_baseline, damaged):
        """Rule-based reading of exactly the cells the model saw, or None if unavailable."""
        if grade == "g1":
            return rule_decode(shown), "Rule decoder (lookup table, no context)"
        name = "liblouis back-translation (rule-based)"
        if louis.available():
            return louis.to_print([shown], "g2")[0].replace("…", "..."), name
        if not damaged and clean_baseline:
            return clean_baseline, name + ", recorded for this test sentence"
        return None, name + ": not installed on this machine"

    def translate(self, payload):
        text = (payload.get("print") or "").strip()
        reference = payload.get("reference") or None
        grade = payload.get("grade") or "auto"
        two = len(self.reader.grades) > 1
        if grade == "auto":      # typed English becomes contracted braille, the everyday form, when possible
            grade = (("g2" if two and louis.available() else "g1") if text
                     else detect_grade(payload.get("braille") or "") if two else "g1")
        if grade not in self.reader.grades:
            raise ValueError(f"The loaded model does not read {grade}.")
        if text:
            if not SUPPORTED.match(text):
                bad = sorted({c for c in text if not SUPPORTED.match(c)})
                raise ValueError("English input supports letters, digits and . , ; : ? ! ' - only. "
                                 "Not supported: " + " ".join(bad[:8]))
            if grade == "g1":
                clean = to_braille(text)
            elif louis.available():
                clean = normalise(louis.to_braille([" ".join(text.split())], "g2")[0])
            else:
                raise ValueError("Converting English to Grade 2 needs liblouis, which is not installed here. "
                                 "Paste Grade 2 braille or pick a test sentence instead.")
            reference = text
        else:
            clean = normalise(payload.get("braille") or "")
        if not clean:
            raise ValueError("Type some English or paste braille first.")
        if len(clean) > MAX_CELLS:
            raise ValueError(f"Input is limited to {MAX_CELLS} cells.")
        rate = min(max(float(payload.get("noise") or 0), 0.0), 0.3)
        shown, changed = damage(clean, rate, int(payload.get("seed") or 1)) if rate else (clean, [])
        with self.lock:
            faithful = self.reader.translate(shown, "faithful", grade)
            repair = self.reader.translate(shown, "repair", grade)
        rule, rule_name = self.baseline(grade, shown, payload.get("baseline"), bool(changed))

        def exact(text):
            return None if reference is None or text is None else text == reference
        return {"braille": shown, "cells": describe(shown, changed, grade), "damaged": len(changed), "grade": grade,
                "faithful": faithful, "repair": repair, "rule": {"text": rule, "name": rule_name},
                "reference": reference, "faithful_exact": exact(faithful["text"]),
                "repair_exact": exact(repair["text"]), "rule_exact": exact(rule)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/meta" not in fmt % args:
            sys.stderr.write("[demo] " + fmt % args + "\n")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        engine = self.server.engine
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._json(200, {"samples": engine.samples, "metrics": engine.metrics, **engine.info()})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/translate":
            return self._json(404, {"error": "Not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= 200_000:
                raise ValueError("Expected a small JSON body.")
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            self._json(200, self.server.engine.translate(payload))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    ap = argparse.ArgumentParser(description="BrailleLM v2 local demo")
    ap.add_argument("--model", default=str(MODEL_ONNX))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model not found at {args.model}. See README.md.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.engine = Engine(args.model)
    url = f"http://{args.host}:{args.port}"
    print(f"[demo] BrailleLM v2 ready at {url}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped")


if __name__ == "__main__":
    main()
