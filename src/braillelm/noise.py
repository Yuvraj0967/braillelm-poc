"""Cell-level noise that imitates damaged or mis-read braille.

A braille cell is six dots, i.e. six bits. The dominant real-world errors
(worn embossing, OCR of dot patterns, slips on a six-key brailler) add or drop
a single dot, which is a one-bit flip of the cell. Less often a cell is
replaced outright or lost.
"""
import random

BLANK_ID = 0


def corrupt_ids(ids, rate, rng, p_flip=0.8, p_sub=0.1):
    """ids: list of cell ids 0..63. Blank (space) cells are left alone."""
    out = []
    for c in ids:
        if c == BLANK_ID or rng.random() >= rate:
            out.append(c)
            continue
        u = rng.random()
        if u < p_flip:
            out.append(c ^ (1 << rng.randrange(6)))
        elif u < p_flip + p_sub:
            out.append(rng.randrange(1, 64))
        # else: the cell is dropped
    return out or [BLANK_ID]


def corrupt_braille(braille, rate, seed, flips_only=True):
    rng = random.Random(seed)
    ids = [ord(ch) - 0x2800 for ch in braille]
    ids = corrupt_ids(ids, rate, rng, p_flip=1.0 if flips_only else 0.8)
    return ''.join(chr(0x2800 + i) for i in ids)
