"""Rebuild and verify the 15,427 WikiText rows in the published train file.

This uses the same candidate extraction, seed, liblouis calls, and round-trip
filter as build_wikitext_grade1.py, but keeps all eligible WikiText-2 training
sentences rather than the initial 2,000-candidate subset. The first 459 rows
in the publication file are the original project's training pairs.
"""

import argparse
import json
from pathlib import Path
import random

from datasets import load_dataset

from build_wikitext_grade1 import SEED, candidates, translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--published_train", default="data/train.jsonl")
    args = parser.parse_args()

    source = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    chosen = candidates(source["train"])
    random.Random(SEED).shuffle(chosen)
    braille = translate([row["print"] for row in chosen])
    roundtrip = translate(braille, backward=True)
    rebuilt = [
        {"print": row["print"], "braille": brl,
         "source": "Salesforce/wikitext:wikitext-2-raw-v1",
         "wikitext_split": "train", "source_line": row["line"]}
        for row, brl, back in zip(chosen, braille, roundtrip)
        if row["print"].lower() == back.lower()
    ]
    published = [json.loads(line) for line in Path(args.published_train).read_text().splitlines()]
    assert len(published) == 15886, len(published)
    assert len(rebuilt) == 15427, len(rebuilt)
    assert published[459:] == rebuilt, "WikiText rows differ from published train file"
    print("Verified: 459 original pairs + 15,427 reconstructed WikiText pairs")


if __name__ == "__main__":
    main()
