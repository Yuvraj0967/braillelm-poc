"""FastAPI service for BrailleLM.

    uvicorn braillelm.serving.api:app --port 7861

Serves the demo page at / and a JSON API (interactive docs at /docs):

    GET  /health          liveness and model version
    POST /v1/translate    braille -> print text, with per-character confidence
    POST /api/translate   the richer payload the demo page uses
    GET  /api/meta        demo samples and reported metrics

Set BRAILLELM_MODEL to serve a different ONNX file.
"""
from __future__ import annotations

import json
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from braillelm import __version__
from braillelm.paths import MODEL_DIR, MODEL_ONNX
from braillelm.runtime import detect_grade
from braillelm.serving.server import HERE, MAX_CELLS, Engine

app = FastAPI(title="BrailleLM", version=__version__,
              description="Cell-level neural reader for Unified English Braille, Grade 1 and Grade 2.")
_engine: Optional[Engine] = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(os.environ.get("BRAILLELM_MODEL") or MODEL_ONNX)
    return _engine


class TranslateRequest(BaseModel):
    braille: str = Field(..., description="Unicode braille cells (U+2800 to U+283F); spaces are allowed")
    mode: Literal["faithful", "repair"] = Field(
        "faithful", description="faithful trusts every dot; repair may fix damaged cells")
    grade: Literal["auto", "g1", "g2"] = Field(
        "auto", description="g1 is uncontracted braille, g2 is contracted; auto detects it from the cells")


class Char(BaseModel):
    c: str
    p: float = Field(..., description="model confidence for this character")
    cell: int = Field(..., description="index of the input cell it was read from")


class TranslateResponse(BaseModel):
    text: str
    chars: list[Char]
    cells: int
    ms: float
    mode: str
    grade: str
    model_version: str


class DemoRequest(BaseModel):
    braille: Optional[str] = None
    print_: Optional[str] = Field(None, alias="print")
    reference: Optional[str] = None
    baseline: Optional[str] = None
    grade: Literal["auto", "g1", "g2"] = "auto"
    noise: float = 0.0
    seed: int = 1


def _card():
    try:
        return json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@app.get("/health")
def health():
    engine()
    return {"status": "ok", "version": __version__, "grades": list(engine().reader.grades),
            "model_file": _card().get("file"), "model_step": _card().get("training", {}).get("step")}


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    eng = engine()
    if len(req.braille) > MAX_CELLS * 2:
        raise HTTPException(413, f"Input is limited to {MAX_CELLS} cells.")
    grade = req.grade
    if grade == "auto":
        grade = detect_grade(req.braille) if len(eng.reader.grades) > 1 else "g1"
    if grade not in eng.reader.grades:
        raise HTTPException(422, f"The loaded model does not read {grade}.")
    with eng.lock:
        out = eng.reader.translate(req.braille, req.mode, grade)
    if not out["cells"]:
        raise HTTPException(422, "No braille cells found in the input.")
    if out["cells"] > MAX_CELLS:
        raise HTTPException(413, f"Input is limited to {MAX_CELLS} cells.")
    return {"text": out["text"], "chars": out["chars"], "cells": out["cells"], "ms": out["ms"],
            "mode": req.mode, "grade": grade, "model_version": __version__}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta", include_in_schema=False)
def meta():
    eng = engine()
    return {"samples": eng.samples, "metrics": eng.metrics, **eng.info()}


@app.post("/api/translate", include_in_schema=False)
def demo_translate(req: DemoRequest):
    try:
        return engine().translate({"braille": req.braille, "print": req.print_, "reference": req.reference,
                                   "baseline": req.baseline, "grade": req.grade,
                                   "noise": req.noise, "seed": req.seed})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)   # shape the demo page expects
