"""Text variations that teach the reader not to rely on having seen a word.

Wikipedia prose has few all-capital sentences, made-up words, vowel-dropped
slang, brand-style capitals or one-word inputs. A model trained only on it
reads Grade 2 partly from memory of familiar words and stumbles on these. Each
function below turns an ordinary sentence into one such variant; the braille
is then produced by the usual converters, so the pairs stay correct.
"""
import string

_VOWELS = set("aeiou")
_STRIP = ".,;:!?"


def _core(word):
    """Split trailing punctuation off a word: 'dog,' -> ('dog', ',')."""
    core = word.rstrip(_STRIP)
    return core, word[len(core):]


def nonsense(rng):
    n = rng.randint(2, 9)
    w = "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
    u = rng.random()
    if u < 0.2:
        return w.capitalize()
    if u < 0.3:
        return w.upper()
    if u < 0.45 and n >= 4:
        k = rng.randint(2, n - 2)
        return w[:k].capitalize() + w[k:].capitalize()
    return w


def devowel(word, rng):
    """'tomorrow' -> 'tmrrw': the way people shorten words in messages."""
    out = word[0] + "".join(c for c in word[1:] if c.lower() not in _VOWELS or rng.random() < 0.2)
    return out if len(out) >= 2 else word


def inner_capital(word, rng):
    """'oneplus' -> 'OnePlus'."""
    if len(word) < 4:
        return word.capitalize()
    k = rng.randint(2, len(word) - 2)
    return word[:k].capitalize() + word[k:].capitalize()


def odd_words(sentence, rng):
    """Replace one or two plain words with a made-up, shortened or oddly capitalised one."""
    words = sentence.split(" ")
    plain = [i for i, w in enumerate(words) if _core(w)[0].isalpha() and len(_core(w)[0]) >= 3]
    if not plain:
        return None
    for i in rng.sample(plain, min(len(plain), rng.randint(1, 2))):
        core, tail = _core(words[i])
        u = rng.random()
        new = nonsense(rng) if u < 0.45 else devowel(core, rng) if u < 0.75 else inner_capital(core.lower(), rng)
        words[i] = new + tail
    return " ".join(words)


def all_caps(sentence, rng=None):
    return sentence.upper()


def title_case(sentence, rng=None):
    return " ".join(w[:1].upper() + w[1:] for w in sentence.split(" "))


def lower_case(sentence, rng=None):
    return sentence.lower()


def fragment(sentence, rng):
    """One to three consecutive words, as someone might type into a demo."""
    words = sentence.split(" ")
    n = rng.randint(1, 3)
    start = rng.randrange(0, max(1, len(words) - n + 1))
    piece = " ".join(words[start:start + n])
    if rng.random() < 0.6:
        piece = piece.rstrip(_STRIP)
    return piece or None


KINDS = {"all_caps": all_caps, "title_case": title_case, "lower_case": lower_case,
         "odd_words": odd_words, "fragment": fragment}


# ---- Second round: patterns the v3.1 model misread ("DVDs" as "DVD's", "HEllo" as "HE'lLO",
# "B3" as "Bc", the lone word "as" as "z"). Kept apart from KINDS so the unusual-text test set
# built from KINDS stays exactly the same from one release to the next.

_ACRONYMS = ("DVD CD ATM PIN MP ID TV PC UFO NGO CEO SUV GPU URL PDF SMS OTP EMI MLA IAS IRA USA UK UN EU "
             "FAQ GP HR IT PM VIP MRI CCTV LED SIM").split()
# Words that are a single cell (or a short sign) in Grade 2 when they stand alone.
_WORDSIGNS = ("but can do every from go have just knowledge like more not people quite rather so that us very "
              "will it you as child shall this which out still be enough were his in was and for of the with "
              "to by into").split()


def _acronym(rng):
    if rng.random() < 0.5:
        return rng.choice(_ACRONYMS)
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(2, 5)))


def _switch(word, rng):
    """One word that changes case partway through, or an acronym with an ending."""
    u = rng.random()
    if u < 0.30:
        return _acronym(rng) + "s"                       # DVDs
    if u < 0.40:
        return _acronym(rng) + "'s"                      # DVD's, so the two stay distinct
    if u < 0.50:
        return _acronym(rng)
    if u < 0.55:
        return _acronym(rng) + "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(1, 3)))
    w, n = word.lower(), len(word)
    if u < 0.75:
        k = rng.randint(2, n - 1)
        return w[:k].upper() + w[k:]                     # HEllo
    if u < 0.90 and n >= 4:
        j = rng.randint(1, n - 3)
        k = rng.randint(j + 2, n - 1)
        return w[:j].capitalize() + w[j:k].upper() + w[k:]   # HeLLo
    k = rng.randint(1, n - 2)
    return w[:k] + w[k:].upper()                         # heLLO


def _code(rng):
    """A short mix of letters and digits: B3, 42a, AB12, x200, R2D2, 3rd."""
    letters = lambda n, up: "".join(rng.choice(string.ascii_uppercase if up else string.ascii_lowercase)
                                    for _ in range(n))
    digits = lambda n: str(rng.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1))
    u, up = rng.random(), rng.random() < 0.7
    if u < 0.40:
        return letters(rng.randint(1, 2), up) + digits(rng.randint(1, 3))
    if u < 0.65:
        return digits(rng.randint(1, 3)) + letters(rng.randint(1, 2), up)
    if u < 0.80:
        return letters(1, up) + digits(1) + letters(1, up) + digits(1)
    if u < 0.90:
        return digits(rng.randint(1, 2)) + rng.choice(["st", "nd", "rd", "th"])
    return letters(1, up) + digits(rng.randint(1, 2)) + letters(1, up)


def _replace(sentence, rng, make):
    words = sentence.split(" ")
    plain = [i for i, w in enumerate(words) if _core(w)[0].isalpha() and len(_core(w)[0]) >= 3]
    if not plain:
        return None
    for i in rng.sample(plain, min(len(plain), rng.randint(1, 2))):
        core, tail = _core(words[i])
        words[i] = make(core) + tail
    return " ".join(words)


def caps_switch(sentence, rng):
    return _replace(sentence, rng, lambda core: _switch(core, rng))


def codes(sentence, rng):
    return _replace(sentence, rng, lambda core: _code(rng))


def wordsigns(sentence, rng):
    """One to three words with at least one Grade 2 wordsign, as a stand-alone fragment."""
    sign = rng.choice(_WORDSIGNS)
    u = rng.random()
    sign = sign.capitalize() if u < 0.2 else sign.upper() if u < 0.3 else sign
    others = [_core(w)[0] for w in sentence.split(" ") if _core(w)[0].isalpha()]
    v = rng.random()
    if v < 0.25 or not others:
        piece = sign
    elif v < 0.6:
        piece = sign + " " + rng.choice(others)
    elif v < 0.9:
        piece = rng.choice(others) + " " + sign
    else:
        piece = rng.choice(others) + " " + sign + " " + rng.choice(others)
    return piece + (rng.choice(".,?!") if rng.random() < 0.25 else "")


MORE_KINDS = {"caps_switch": caps_switch, "codes": codes, "wordsigns": wordsigns}
ALL_KINDS = {**KINDS, **MORE_KINDS}
