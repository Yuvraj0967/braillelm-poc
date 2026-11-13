"""Export a v2 checkpoint to ONNX and check it matches PyTorch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from braillelm.model import load, encode_cells
from braillelm import tracking
from braillelm.paths import MODEL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(MODEL_DIR))
    ap.add_argument("--mlflow", metavar="EXPERIMENT", help="also log and register the ONNX model in MLflow")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, extra = load(args.ckpt)
    example = torch.tensor([encode_cells("⠠⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙⠲")])
    path = out / "braillelm_v2.onnx"
    torch.onnx.export(model, (example, torch.tensor([0])), str(path),
                      input_names=["cells", "mode"], output_names=["logits"],
                      dynamic_axes={"cells": {0: "batch", 1: "cells"}, "mode": {0: "batch"},
                                    "logits": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for n in (1, 7, 40, 150):
        for mode in (0, 1):
            x, m = torch.randint(0, 64, (1, n)), torch.tensor([mode])
            ref = model(x, m).detach().numpy()
            got = sess.run(None, {"cells": x.numpy(), "mode": m.numpy()})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
    assert worst < 1e-3, worst
    card = {"file": path.name, "bytes": path.stat().st_size, "max_abs_diff_vs_torch": worst,
            "config": model.cfg, "training": extra}
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    torch.save(torch.load(args.ckpt, map_location="cpu", weights_only=True), out / "braillelm_v2.pt")
    print(json.dumps(card, indent=2))
    with tracking.run(args.mlflow, run_name="export") as tracker:
        tracker.params({"source_checkpoint": args.ckpt, **model.cfg})
        tracker.metrics({k: v for k, v in (extra or {}).items() if k.startswith("dev_")})
        tracker.artifact(out / "model_card.json")
        tracker.register_onnx(path, "braillelm")


if __name__ == "__main__":
    main()
