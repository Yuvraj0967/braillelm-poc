"""
server.py -- local demo server for BrailleLM.

Serves a single-page interface and runs the trained native checkpoint on
braille typed or pasted into the page. Optionally loads the baseline LoRA
adapter so the two arms can be compared side by side.

Usage (from the repository root):
    python demo/server.py --open
    python demo/server.py --port 7860 --no-baseline
    python demo/server.py --native-model path/to/braillelm_native

Only the Python standard library is used for the server itself; the model
stack is the same one the training pipeline uses.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from braille_ueb import (  # noqa: E402
    CAPITAL,
    NUMBER,
    SPACE,
    _cell_fragment,
    to_braille,
)

PROMPT = "Translate the following braille to English text.\nBraille: {brl}\nEnglish:"
CELL_MIN, CELL_MAX = 0x2800, 0x28FF
MAX_CELLS = 600
MAX_PRINT = 240
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
SAMPLE_IDS = [1, 2, 3, 17, 34, 55, 77, 104]

NATIVE_CANDIDATES = [
    ROOT / "models" / "braillelm-native",
    ROOT / "models" / "braillelm_native",
    ROOT.parent / "braillelm_delivery" / "braillelm_native",
    ROOT.parent / "braillelm_native",
    Path.home() / "braillelm_native",
]


# --------------------------------------------------------------------------
# Model paths and corpus metadata
# --------------------------------------------------------------------------
def resolve_native(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "config.json").exists():
            raise SystemExit(f"Native checkpoint not found at {p}")
        return p
    for candidate in NATIVE_CANDIDATES:
        if (candidate / "config.json").exists() and (
            candidate / "model.safetensors"
        ).exists():
            return candidate
    raise SystemExit(
        "\nNative checkpoint not found.\n\n"
        "Download it from the GitHub release and extract it:\n"
        "  https://github.com/Yuvraj0967/braillelm-poc/releases/latest/download/"
        "braillelm-native-smollm2-135m.tar\n\n"
        "  mkdir -p models\n"
        "  tar -xf braillelm-native-smollm2-135m.tar -C models\n\n"
        "Then run this server again, or pass --native-model <directory>.\n"
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scores(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for arm in ("native", "baseline", "control"):
        row = raw.get(arm, {})
        if row:
            out[arm] = {
                "n": row.get("n"),
                "exact": row.get("exact"),
                "cer": row.get("cer_mean"),
                "bleu": row.get("sacrebleu"),
                "distinct": row.get("distinct_outputs"),
                "vocab": row.get("vocab_emitted"),
                "vocab_reference": row.get("vocab_reference"),
                "actor": (row.get("slot_accuracy") or [None, None])[1],
            }
    return out


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, native_dir: Path, adapter_dir: Path | None, use_baseline: bool):
        self.native_dir = native_dir
        self.adapter_dir = adapter_dir
        self.use_baseline = use_baseline

        self.native_tok = None
        self.native_model = None
        self.native_ready = threading.Event()
        self.native_error: str | None = None

        self.baseline_tok = None
        self.baseline_model = None
        self.baseline_ready = threading.Event()
        self.baseline_error: str | None = None

        self.generate_lock = threading.Lock()

        self.test_rows = read_jsonl(ROOT / "benchmark" / "test_set.jsonl")
        self.by_braille = {row["braille"]: row for row in self.test_rows}
        self.samples = self._pick_samples()
        self.scores = load_scores(ROOT / "benchmark" / "reference_scores.json")

        self.stage = "loading native checkpoint"

    def _pick_samples(self) -> list[dict]:
        by_id = {row.get("id"): row for row in self.test_rows}
        picked = [by_id[i] for i in SAMPLE_IDS if i in by_id]
        if len(picked) < 6:
            picked = self.test_rows[:8]
        return [
            {
                "id": row.get("id"),
                "braille": row["braille"],
                "print": row["print"],
                "words": len(row["print"].split()),
            }
            for row in picked
        ]

    # -- loading -----------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._load_native, name="native-loader", daemon=True).start()

    def _load_native(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            self.native_tok = AutoTokenizer.from_pretrained(self.native_dir)
            self.native_model = AutoModelForCausalLM.from_pretrained(
                self.native_dir, dtype=torch.float32
            )
            self.native_model.eval()
            elapsed = time.perf_counter() - started
            print(
                f"[demo] native ready in {elapsed:.1f}s "
                f"({len(self.native_tok):,} tokens) from {self.native_dir}",
                flush=True,
            )
            self.stage = "ready"
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self.native_error = f"{type(exc).__name__}: {exc}"
            self.stage = "native model failed to load"
            print(f"[demo] native load failed: {self.native_error}", flush=True)
        finally:
            self.native_ready.set()

        if self.use_baseline:
            threading.Thread(
                target=self._load_baseline, name="baseline-loader", daemon=True
            ).start()

    def _load_baseline(self) -> None:
        self.stage = "loading control arm"
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            self.baseline_tok = AutoTokenizer.from_pretrained(self.adapter_dir)
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, dtype=torch.float32, local_files_only=True
                )
            except Exception:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, dtype=torch.float32
                )
            base.resize_token_embeddings(len(self.baseline_tok))
            self.baseline_model = PeftModel.from_pretrained(base, self.adapter_dir)
            self.baseline_model.to("cpu").eval()
            elapsed = time.perf_counter() - started
            print(
                f"[demo] control arm ready in {elapsed:.1f}s "
                f"({len(self.baseline_tok):,} tokens)",
                flush=True,
            )
            self.stage = "ready"
        except Exception as exc:  # noqa: BLE001 - the native arm still works
            self.baseline_error = f"{type(exc).__name__}: {exc}"
            self.stage = "ready"
            print(f"[demo] control arm unavailable: {self.baseline_error}", flush=True)
        finally:
            self.baseline_ready.set()

    # -- generation --------------------------------------------------------
    def _generate(self, tok, model, braille: str) -> dict:
        prompt = PROMPT.format(brl=braille)
        inputs = tok(prompt, return_tensors="pt")
        started = time.perf_counter()
        with self.generate_lock, torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_len = inputs["input_ids"].shape[1]
        text = tok.decode(
            generated[0][prompt_len:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].strip()
        token_count = len(tok(braille, add_special_tokens=False)["input_ids"])
        cells = sum(1 for ch in braille if CELL_MIN <= ord(ch) <= CELL_MAX)
        return {
            "text": text,
            "latency_ms": round(elapsed_ms),
            "tokens": token_count,
            "cells": cells,
            "tokens_per_cell": round(token_count / cells, 3) if cells else None,
        }

    def _wait_for_native(self) -> bool:
        return self.native_ready.wait(timeout=300) and self.native_error is None

    def get_baseline(self):
        if not self.use_baseline or self.baseline_error:
            return None
        if not self.baseline_ready.wait(timeout=180):
            return None
        if self.baseline_model is None:
            return None
        return self.baseline_tok, self.baseline_model

    # -- request -----------------------------------------------------------
    def translate(self, payload: dict) -> dict:
        compare = bool(payload.get("compare", True))
        print_text = (payload.get("print") or "").strip()
        braille_raw = payload.get("braille") or ""
        encoded = False

        if print_text:
            if len(print_text) > MAX_PRINT:
                raise ValueError(f"Print input is limited to {MAX_PRINT} characters.")
            braille = to_braille(print_text, 1)
            encoded = True
        else:
            braille = (
                braille_raw.replace("\r", "")
                .replace("\n", SPACE)
                .replace(" ", SPACE)
                .strip(SPACE)
            )

        if not braille:
            raise ValueError("Enter a braille sentence or switch to English input.")

        bad = sorted({ch for ch in braille if not CELL_MIN <= ord(ch) <= CELL_MAX})
        if bad:
            shown = " ".join(f"{ch!r}" for ch in bad[:5])
            raise ValueError(
                "Braille input contains characters outside the U+2800 block: "
                + shown
            )
        cells = sum(1 for ch in braille if CELL_MIN <= ord(ch) <= CELL_MAX)
        if cells > MAX_CELLS:
            raise ValueError(f"Input is limited to {MAX_CELLS} braille cells.")

        if not self._wait_for_native():
            raise RuntimeError(self.native_error or "The native model is still loading.")

        native = self._generate(self.native_tok, self.native_model, braille)

        baseline = None
        baseline_error = None
        if compare:
            pair = self.get_baseline()
            if pair:
                baseline = self._generate(pair[0], pair[1], braille)
            else:
                baseline_error = self.baseline_error or "Control arm is not available."

        reference = self.by_braille.get(braille)
        if reference:
            exact = reference["print"].strip().lower() == native["text"].strip().lower()
            reference = {
                "id": reference.get("id"),
                "print": reference["print"],
                "exact": exact,
            }

        return {
            "braille": braille,
            "encoded": encoded,
            "source_print": print_text or None,
            "native": native,
            "baseline": baseline,
            "baseline_error": baseline_error,
            "reference": reference,
            "cells": describe_cells(braille),
            "engine": {
                "vocab": len(self.native_tok) if self.native_tok else None,
                "baseline_vocab": len(self.baseline_tok) if self.baseline_tok else None,
            },
        }

    def meta(self) -> dict:
        ready = self.native_ready.is_set() and self.native_error is None
        return {
            "status": "ready" if ready else "loading",
            "stage": self.stage,
            "error": self.native_error,
            "native": {
                "path": str(self.native_dir),
                "vocab": len(self.native_tok) if self.native_tok else None,
            },
            "baseline": {
                "enabled": self.use_baseline,
                "ready": self.baseline_ready.is_set() and self.baseline_model is not None,
                "error": self.baseline_error,
                "vocab": len(self.baseline_tok) if self.baseline_tok else None,
            },
            "samples": self.samples,
            "scores": self.scores,
        }


def describe_cells(braille: str) -> list[dict]:
    out = []
    for ch in braille:
        if ch == SPACE:
            out.append({"label": "space", "dots": [], "space": True})
            continue
        code = ord(ch)
        value = code - CELL_MIN
        dots = [i + 1 for i in range(8) if value & (1 << i)]
        if ch == CAPITAL:
            label = "CAP"
        elif ch == NUMBER:
            label = "NUM"
        else:
            label = _cell_fragment(ch) or "blank"
        out.append({"label": label, "dots": dots, "space": False})
    return out


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "BrailleLMDemo/1.0"

    @property
    def engine(self) -> Engine:
        return self.server.engine  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        line = fmt % args
        if "GET /api/meta" in line or "GET /favicon.ico" in line:
            return
        sys.stderr.write("[demo] " + line + "\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._json(200, self.engine.meta())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/translate":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 64_000:
            self._json(400, {"error": "Expected a small JSON body."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._json(400, {"error": f"Bad request: {exc}"})
            return

        try:
            result = self.engine.translate(payload)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self.log_message("translate failed: %s", exc)
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(200, result)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="BrailleLM local demo")
    parser.add_argument("--native-model", help="directory of the merged native checkpoint")
    parser.add_argument(
        "--baseline-adapter",
        default=str(ROOT / "models" / "baseline-lora"),
        help="directory of the baseline LoRA adapter",
    )
    parser.add_argument("--no-baseline", action="store_true", help="skip the control arm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args()

    native_dir = resolve_native(args.native_model)
    adapter_dir = Path(args.baseline_adapter).expanduser().resolve()
    use_baseline = not args.no_baseline and (adapter_dir / "adapter_config.json").exists()
    if not use_baseline and not args.no_baseline:
        print(f"[demo] no baseline adapter at {adapter_dir}; running native only", flush=True)

    engine = Engine(native_dir, adapter_dir, use_baseline)
    engine.start()

    try:
        server = DemoServer((args.host, args.port), Handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not listen on {args.host}:{args.port} ({exc}).\n"
            "Try a different port, for example: python demo/server.py --port 8000"
        )
    server.engine = engine  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"[demo] serving {url}", flush=True)
    print("[demo] model is loading in the background; the page will update.", flush=True)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
