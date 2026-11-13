"""
braille_ueb.py -- UEB braille conversion for the BrailleLM proof-of-concept.

- Grade 1 (uncontracted) is an EXACT, invertible letter<->cell mapping.
- Grade 2 (contracted) applies a subset of common UEB contractions.
- Provides the braille->print inverse (cell_to_print) that BKFT embedding
  initialisation and round-trip filtering both need.

For the full standard you would swap this for liblouis (en-ueb-g1/g2 tables);
prep_data.py will use liblouis automatically if it is installed, and fall back
to this engine otherwise. The training pipeline does not care which produced
the data.
"""

def dots_to_char(dots):
    v = 0
    for d in dots:
        v |= 1 << (d - 1)
    return chr(0x2800 + v)

LETTER_DOTS = {
    'a':[1],'b':[1,2],'c':[1,4],'d':[1,4,5],'e':[1,5],'f':[1,2,4],'g':[1,2,4,5],
    'h':[1,2,5],'i':[2,4],'j':[2,4,5],'k':[1,3],'l':[1,2,3],'m':[1,3,4],
    'n':[1,3,4,5],'o':[1,3,5],'p':[1,2,3,4],'q':[1,2,3,4,5],'r':[1,2,3,5],
    's':[2,3,4],'t':[2,3,4,5],'u':[1,3,6],'v':[1,2,3,6],'w':[2,4,5,6],
    'x':[1,3,4,6],'y':[1,3,4,5,6],'z':[1,3,5,6],
}
LETTER = {c: dots_to_char(d) for c, d in LETTER_DOTS.items()}
CELL_TO_LETTER = {v: k for k, v in LETTER.items()}

CAPITAL = dots_to_char([6])
NUMBER  = dots_to_char([3,4,5,6])
SPACE   = chr(0x2800)

DIGIT = {'1':'a','2':'b','3':'c','4':'d','5':'e','6':'f','7':'g','8':'h','9':'i','0':'j'}
LETTER_TO_DIGIT = {v: k for k, v in DIGIT.items()}

PUNCT = {
    '.': dots_to_char([2,5,6]), ',': dots_to_char([2]), ';': dots_to_char([2,3]),
    ':': dots_to_char([2,5]), '?': dots_to_char([2,3,6]), '!': dots_to_char([2,3,5]),
    "'": dots_to_char([3]), '-': dots_to_char([3,6]),
}
CELL_TO_PUNCT = {v: k for k, v in PUNCT.items()}

# Grade 2 groupsigns: printed sequence <-> single cell
GROUPSIGNS = [
    ('ing',dots_to_char([3,4,6])),('the',dots_to_char([2,3,4,6])),
    ('and',dots_to_char([1,2,3,4,6])),('for',dots_to_char([1,2,3,4,5,6])),
    ('of',dots_to_char([1,2,3,5,6])),('with',dots_to_char([2,3,4,5,6])),
    ('ch',dots_to_char([1,6])),('sh',dots_to_char([1,4,6])),('th',dots_to_char([1,4,5,6])),
    ('wh',dots_to_char([1,5,6])),('ou',dots_to_char([1,2,5,6])),('st',dots_to_char([3,4])),
    ('ar',dots_to_char([3,4,5])),('ed',dots_to_char([1,2,4,6])),('er',dots_to_char([1,2,4,5,6])),
    ('ow',dots_to_char([2,4,6])),('en',dots_to_char([2,6])),('in',dots_to_char([3,5])),
]
CELL_TO_GROUP = {cell: seq for seq, cell in GROUPSIGNS}

def _translate_word(word, grade):
    low = word.lower()
    out, i = [], 0
    while i < len(low):
        matched = False
        if grade == 2:
            for seq, cell in GROUPSIGNS:
                if low.startswith(seq, i):
                    if word[i].isupper():
                        out.append(CAPITAL)
                    out.append(cell); i += len(seq); matched = True; break
        if matched:
            continue
        ch = low[i]
        if ch in LETTER:
            if word[i].isupper():
                out.append(CAPITAL)
            out.append(LETTER[ch])
        else:
            out.append(ch)
        i += 1
    return ''.join(out)

def to_braille(text, grade=1):
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == ' ':
            out.append(SPACE); i += 1; continue
        if ch == '\n':
            out.append('\n'); i += 1; continue
        if ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            out.append(NUMBER)
            out.extend(LETTER[DIGIT[d]] for d in text[i:j])
            i = j; continue
        if ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            out.append(_translate_word(text[i:j], grade))
            i = j; continue
        if ch in PUNCT:
            out.append(PUNCT[ch]); i += 1; continue
        out.append(ch); i += 1
    return ''.join(out)

def back_translate(brl):
    """Grade-1 exact inverse (cell sequence -> print). Used for round-trip
    filtering and as a fallback expansion. Grade-2 contraction cells are
    expanded to their letter group (good enough for BKFT / filtering)."""
    out, i, n = [], 0, len(brl)
    while i < n:
        ch = brl[i]
        if ch == SPACE:
            out.append(' '); i += 1; continue
        if ch == '\n':
            out.append('\n'); i += 1; continue
        if ch == CAPITAL:
            # capitalise the next translated chunk
            i += 1
            if i < n:
                nxt = brl[i]
                frag = _cell_fragment(nxt)
                out.append(frag.capitalize()); i += 1
            continue
        if ch == NUMBER:
            i += 1
            while i < n and brl[i] in CELL_TO_LETTER and CELL_TO_LETTER[brl[i]] in LETTER_TO_DIGIT:
                out.append(LETTER_TO_DIGIT[CELL_TO_LETTER[brl[i]]]); i += 1
            continue
        out.append(_cell_fragment(ch)); i += 1
    return ''.join(out)

def _cell_fragment(cell):
    """Most common print fragment for a single braille cell."""
    if cell in CELL_TO_GROUP:
        return CELL_TO_GROUP[cell]
    if cell in CELL_TO_LETTER:
        return CELL_TO_LETTER[cell]
    if cell in CELL_TO_PUNCT:
        return CELL_TO_PUNCT[cell]
    return ''

def cell_to_print(token):
    """BKFT helper: map a braille token (one or more cells) to the print
    string it represents, by expanding each cell. Capital / number / space
    indicators expand to '' so they don't pollute the averaged embedding."""
    out = []
    for ch in token:
        if ch in (CAPITAL, NUMBER, SPACE, '\n'):
            continue
        out.append(_cell_fragment(ch))
    return ''.join(out)

def count_cells(brl):
    return sum(1 for ch in brl if ch != '\n')

# ASCII braille (BRF) for reference / baseline experiments
BRAILLE_ASCII = (" A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=")
def unicode_to_ascii(brl):
    out = []
    for ch in brl:
        o = ord(ch)
        if 0x2800 <= o <= 0x283F:
            out.append(BRAILLE_ASCII[o - 0x2800])
        elif ch == '\n':
            out.append('\n')
        else:
            out.append(ch)
    return ''.join(out)

if __name__ == '__main__':
    for s in ["The quick brown fox.", "Reading braille with your fingers."]:
        g1 = to_braille(s, 1)
        print("print:", s)
        print("  g1 :", g1)
        print("  <- :", back_translate(g1), "(round-trip ok:", back_translate(g1).lower()==s.lower(), ")")
        print("  g2 :", to_braille(s, 2))
