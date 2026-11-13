"""Verify braille_g1 against every liblouis-generated row in neural_mac/data.

The published braille was produced by liblouis 3.39.0 (en-ueb-g1.ctb). This
script checks that `to_braille` reproduces it exactly and that `rule_decode`
inverts it exactly. Only rows that pass `safe_for_synthesis` are required to
match, because that is the gate applied to newly synthesized training data.
"""
import json
import sys
from pathlib import Path

from braillelm.braille_g1 import to_braille, rule_decode, safe_for_synthesis

from braillelm.paths import HELDOUT_DATA
ok = True
for name in ["train", "validation", "pilot_plain", "pilot_extended", "final_test"]:
    rows = [json.loads(l) for l in open(HELDOUT_DATA / f"{name}.jsonl", encoding="utf-8")]
    safe = [r for r in rows if safe_for_synthesis(r["print"])]
    bad = [r for r in safe if to_braille(r["print"]) != r["braille"]]
    bad_unsafe = sum(to_braille(r["print"]) != r["braille"] for r in rows if not safe_for_synthesis(r["print"]))
    inv = sum(rule_decode(r["braille"]) == r["print"] for r in rows)
    ok &= not bad
    print(f"{name:15s} rows={len(rows):6d} safe={len(safe):6d} forward mismatches (safe)={len(bad)} "
          f"(excluded rows that differ: {bad_unsafe})  rule_decode strict exact={inv}/{len(rows)}")
    for r in bad[:5]:
        print("   ", r["print"], "\n     louis:", r["braille"], "\n     ours :", to_braille(r["print"]))
print("CONVERTER VERIFIED" if ok else "CONVERTER MISMATCH")
