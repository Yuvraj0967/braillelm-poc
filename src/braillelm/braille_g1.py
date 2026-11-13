"""Grade 1 (uncontracted) UEB conversion used by BrailleLM v2.

`to_braille` reproduces the liblouis `en-ueb-g1.ctb` output for the character
set used in this project (letters, digits, space and , . ; : ? ! ' -). It is
verified against every liblouis-generated row published in neural_mac/data by
`check_converter.py`. Unlike the original `braille_ueb.py` it implements the
UEB capitalised-word indicator (dot 6, dot 6) and numeric mode.

`rule_decode` is the deterministic inverse. It is the rule-based reference the
neural model is compared against; it is never used inside the neural model.
"""
import re

BLANK = chr(0x2800)


def _cell(*dots):
    v = 0
    for d in dots:
        v |= 1 << (d - 1)
    return chr(0x2800 + v)


LETTER = {
    'a': _cell(1), 'b': _cell(1, 2), 'c': _cell(1, 4), 'd': _cell(1, 4, 5), 'e': _cell(1, 5),
    'f': _cell(1, 2, 4), 'g': _cell(1, 2, 4, 5), 'h': _cell(1, 2, 5), 'i': _cell(2, 4),
    'j': _cell(2, 4, 5), 'k': _cell(1, 3), 'l': _cell(1, 2, 3), 'm': _cell(1, 3, 4),
    'n': _cell(1, 3, 4, 5), 'o': _cell(1, 3, 5), 'p': _cell(1, 2, 3, 4), 'q': _cell(1, 2, 3, 4, 5),
    'r': _cell(1, 2, 3, 5), 's': _cell(2, 3, 4), 't': _cell(2, 3, 4, 5), 'u': _cell(1, 3, 6),
    'v': _cell(1, 2, 3, 6), 'w': _cell(2, 4, 5, 6), 'x': _cell(1, 3, 4, 6), 'y': _cell(1, 3, 4, 5, 6),
    'z': _cell(1, 3, 5, 6),
}
CELL_TO_LETTER = {v: k for k, v in LETTER.items()}
CAP = _cell(6)
NUM = _cell(3, 4, 5, 6)
TERM = _cell(3)            # second cell of the capitals terminator (dot 6, dot 3)
PUNCT = {'.': _cell(2, 5, 6), ',': _cell(2), ';': _cell(2, 3), ':': _cell(2, 5),
         '?': _cell(2, 3, 6), '!': _cell(2, 3, 5), "'": _cell(3), '-': _cell(3, 6)}
CELL_TO_PUNCT = {v: k for k, v in PUNCT.items()}
DIGIT = dict(zip('1234567890', 'abcdefghij'))
LETTER_TO_DIGIT = {v: k for k, v in DIGIT.items()}

# Characters the v2 model reads and writes.
PRINT_CHARS = ("abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "0123456789" " .,;:?!'-")
SUPPORTED = re.compile(r"[A-Za-z0-9 .,;:?!'\-]+\Z")


def _word(word):
    """One alphabetic run."""
    out, i, n = [], 0, len(word)
    while i < n:
        if word[i].isupper():
            j = i
            while j < n and word[j].isupper():
                j += 1
            if j - i >= 2:                       # capitalised word indicator
                out.append(CAP + CAP)
                out.extend(LETTER[c.lower()] for c in word[i:j])
                if j < n:                        # lowercase follows: terminate
                    out.append(CAP + TERM)
            else:
                out.append(CAP + LETTER[word[i].lower()])
            i = j
        else:
            out.append(LETTER[word[i]])
            i += 1
    return ''.join(out)


def to_braille(text):
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == ' ':
            out.append(BLANK); i += 1
        elif ch.isdigit():
            out.append(NUM)
            while i < n:
                if text[i].isdigit():
                    out.append(LETTER[DIGIT[text[i]]]); i += 1
                elif text[i] in '.,' and i + 1 < n and text[i + 1].isdigit():
                    out.append(PUNCT[text[i]]); i += 1   # numeric mode continues
                else:
                    break
        elif ch.isalpha() and ch.isascii():
            j = i
            while j < n and text[j].isalpha() and text[j].isascii():
                j += 1
            out.append(_word(text[i:j])); i = j
        elif ch in PUNCT:
            out.append(PUNCT[ch]); i += 1
        else:
            raise ValueError(f"unsupported character {ch!r}")
    return ''.join(out)


def rule_decode(brl):
    """Deterministic Grade 1 back-translation. Unknown cells are dropped."""
    out, i, n = [], 0, len(brl)
    caps_word = numeric = False
    while i < n:
        c = brl[i]
        if c == BLANK or c == ' ':
            out.append(' '); caps_word = numeric = False; i += 1
        elif c == NUM:
            numeric = True; i += 1
        elif c == CAP:
            if i + 1 < n and brl[i + 1] == CAP:
                caps_word = True; i += 2
            elif i + 1 < n and brl[i + 1] == TERM and caps_word:
                caps_word = False; i += 2
            elif i + 1 < n and brl[i + 1] in CELL_TO_LETTER:
                out.append(CELL_TO_LETTER[brl[i + 1]].upper()); numeric = False; i += 2
            else:
                i += 1
        elif numeric and c in CELL_TO_LETTER and CELL_TO_LETTER[c] in LETTER_TO_DIGIT:
            out.append(LETTER_TO_DIGIT[CELL_TO_LETTER[c]]); i += 1
        elif numeric and c in (PUNCT['.'], PUNCT[',']) and i + 1 < n and \
                CELL_TO_LETTER.get(brl[i + 1]) in LETTER_TO_DIGIT:
            out.append(CELL_TO_PUNCT[c]); i += 1
        elif c in CELL_TO_LETTER:
            ch = CELL_TO_LETTER[c]
            out.append(ch.upper() if caps_word else ch); numeric = False; i += 1
        elif c in CELL_TO_PUNCT:
            out.append(CELL_TO_PUNCT[c]); caps_word = numeric = False; i += 1
        else:
            i += 1
    return ''.join(out)


# Sentences whose braille this converter can produce with certainty. Patterns
# that liblouis handles with rules not verified here are excluded from
# synthesized training data instead of being guessed.
_UNSAFE = [
    re.compile(r"[A-Z]{2,}[a-z]"),               # capitals terminator inside a word
    re.compile(r"\b[A-Z]{2,}\b[ ,]+\b[A-Z]{2,}\b[ ,]+\b[A-Z]{2,}\b"),  # capitals passage
    re.compile(r"[A-Za-z][0-9]|[0-9][A-Za-z]"),  # letters touching digits
    re.compile(r"[0-9][.,:;'\-][A-Za-z]|[A-Za-z][.,:;'\-][0-9]"),
    re.compile(r"[A-Z]{2,}['\-]|['\-][A-Z]{2,}"),  # all-caps next to hyphen/apostrophe
]


def safe_for_synthesis(text):
    return bool(SUPPORTED.match(text)) and not any(p.search(text) for p in _UNSAFE)
