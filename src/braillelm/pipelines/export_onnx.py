"""Export a checkpoint to ONNX and check it matches PyTorch.

Grade 1 models (v2) export inputs `cells, mode`; two-grade models (v3) add a
`grade` input. The runtime detects which kind it has loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from braillelm import tracking
from braillelm.model import encode_cells, load
from braillelm.paths import REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", help="default: models/braillelm_v3 for two-grade models, else models/braillelm_v2")
    ap.add_argument("--mlflow", metavar="EXPERIMENT", help="also log and register the ONNX model in MLflow")
    args = ap.parse_args()
    model, extra = load(args.ckpt)
    two = model.grades > 1
    name = "braillelm_v3" if two else "braillelm_v2"
    out = Path(args.out) if args.out else REPO / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    example = torch.tensor([encode_cells("⠠⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙⠲")])
    path = out / f"{name}.onnx"
    inputs = (example, torch.tensor([0])) + ((torch.tensor([0]),) if two else ())
    names = ["cells", "mode"] + (["grade"] if two else [])
    axes = {"cells": {0: "batch", 1: "cells"}, "mode": {0: "batch"}, "logits": {0: "batch", 1: "frames"}}
    if two:
        axes["grade"] = {0: "batch"}
    torch.onnx.export(model, inputs, str(path), input_names=names, output_names=["logits"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for n in (1, 7, 40, 150):
        for mode in (0, 1):
            for grade in range(model.grades):
                x, m, g = torch.randint(0, 64, (1, n)), torch.tensor([mode]), torch.tensor([grade])
                ref = model(x, m, g).detach().numpy()
                feed = {"cells": x.numpy(), "mode": m.numpy()}
                if two:
                    feed["grade"] = g.numpy()
                worst = max(worst, float(np.abs(ref - sess.run(None, feed)[0]).max()))
    assert worst < 1e-3, worst
    card = {"file": path.name, "bytes": path.stat().st_size, "max_abs_diff_vs_torch": worst,
            "config": model.cfg, "grades": ["g1", "g2"][:model.grades], "training": extra}
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    torch.save(torch.load(args.ckpt, map_location="cpu", weights_only=True), out / f"{name}.pt")
    print(json.dumps(card, indent=2))
    with tracking.run(args.mlflow, run_name="export") as tracker:
        tracker.params({"source_checkpoint": args.ckpt, **model.cfg})
        tracker.metrics({k: v for k, v in (extra or {}).items() if k.startswith("dev_")})
        tracker.artifact(out / "model_card.json")
        tracker.register_onnx(path, "braillelm")


if __name__ == "__main__":
    main()
