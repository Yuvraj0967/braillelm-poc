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
