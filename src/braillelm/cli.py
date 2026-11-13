"""Command-line reader.

    braillelm "⠠⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙⠲"
    braillelm --english "Colonel Fordham met Hugh Durham in 1914."
    braillelm --file notes.brl
"""
import argparse
import sys
from pathlib import Path

from braillelm.braille_g1 import to_braille
from braillelm.runtime import Reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("braille", nargs="?")
    ap.add_argument("--english", help="convert this English to Grade 1 braille first, then read it back")
    ap.add_argument("--file", help="UTF-8 text file of Unicode braille")
    ap.add_argument("--repair", action="store_true", help="assume some cells are damaged and use context to fix them")
    ap.add_argument("--model")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.english:
        braille = to_braille(args.english)
    elif args.file:
        braille = Path(args.file).read_text(encoding="utf-8")
    elif args.braille:
        braille = args.braille
    else:
        ap.error("give braille, --english or --file")
    out = Reader(args.model).translate(braille, "repair" if args.repair else "faithful")
    print("braille:", out["braille"])
    print("print  :", out["text"])
    print(f"({out['cells']} cells, {out['ms']} ms)")


if __name__ == "__main__":
    main()
