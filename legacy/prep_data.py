"""
prep_data.py -- build a tiny, accurate braille<->print dataset.

Pipeline:
  1. Take English sentences (built-in list, or --input textfile, one per line).
  2. Convert each to UEB braille (liblouis if installed, else braille_ueb.py).
  3. Round-trip filter: keep a pair only if braille back-translates to the
     original (so every kept pair is known-correct).
  4. Split into train/test and write JSONL of {"braille":..., "print":...}.

Default is Grade 1 (exact, robust for a first run). Use --grade 2 for contracted.

Usage:
  python prep_data.py --grade 1 --out data
  python prep_data.py --input my_sentences.txt --grade 1 --out data
"""
import argparse, json, os, random, re

import braille_ueb as B

# ---- try liblouis; fall back to the built-in engine --------------------------
def get_translators(grade):
    try:
        import louis  # python3-louis bindings
        tables = ["unicode.dis", f"en-ueb-g{grade}.ctb"]
        def fwd(text):
            return louis.translateString(tables, text)
        def back(brl):
            return louis.backTranslateString(tables, brl)
        # smoke-test the bindings
        louis.translateString(tables, "test")
        print("[prep] using liblouis tables:", tables)
        return fwd, back, "liblouis"
    except Exception as e:
        print(f"[prep] liblouis unavailable ({type(e).__name__}); using built-in UEB engine")
        return (lambda t: B.to_braille(t, grade)), (lambda b: B.back_translate(b)), "builtin"

# ---- a small built-in corpus (public-domain style, varied) -------------------
BUILTIN = [
"The quick brown fox jumps over the lazy dog.",
"Braille is a tactile writing system used by people who are blind.",
"Reading and writing are fundamental skills for everyone.",
"A language model could help millions of braille readers.",
"She opened the book and ran her fingers across the raised dots.",
"Every child deserves access to literacy.",
"The weather today is bright and clear.",
"Science and mathematics require their own braille codes.",
"Technology should be designed so that it works for everyone.",
"He wrote a letter to his friend about the trip.",
"The students studied for their examinations in the library.",
"Many common words appear again and again in ordinary writing.",
"Learning to read with the fingers takes practice and patience.",
"The government published the report in several formats.",
"Teachers and transcribers work together to produce textbooks.",
"Access to information is a right that technology can help deliver.",
"The morning sun rose slowly over the hills.",
"Understanding language requires grammar and context.",
"People read and write and share their thoughts.",
"The library contained thousands of books.",
"A small boat sailed quietly across the calm lake.",
"The children played in the garden all afternoon.",
"Music can bring people together from many places.",
"The old clock on the wall had stopped ticking.",
"Fresh bread was baking in the oven downstairs.",
"The train arrived at the station exactly on time.",
"Rain fell softly against the window last night.",
"The museum displayed paintings from many centuries.",
"A gentle wind moved through the tall grass.",
"The scientist recorded her results in a notebook.",
"Birds sang in the trees as the day began.",
"The road wound slowly up the side of the mountain.",
"He poured a cup of tea and sat by the fire.",
"The city lights glowed brightly after sunset.",
"Students asked the teacher many good questions.",
"A kind stranger helped them find the way home.",
"The river flowed gently toward the distant sea.",
"She planted flowers along the garden path.",
"The engineer checked every part of the machine.",
"Snow covered the fields in a blanket of white.",
"The bakery smelled of cinnamon and warm sugar.",
"A dog ran happily across the open field.",
"The lecture covered the history of written language.",
"Workers repaired the bridge over the busy road.",
"The telescope revealed thousands of distant stars.",
"Children learned to count using small wooden blocks.",
"The farmer harvested the wheat before the storm.",
"A quiet melody drifted from the open window.",
"The committee met to discuss the new plan.",
"Leaves turned red and gold in the autumn air.",
"The nurse spoke gently to the worried patient.",
"A ladder leaned against the side of the barn.",
"The author signed copies of her latest book.",
"Waves crashed against the rocks below the cliff.",
"The map showed a shorter route through the valley.",
"He fixed the broken chair with a few nails.",
"The orchestra tuned their instruments before the show.",
"A cat slept peacefully in the warm sunlight.",
"The factory produced hundreds of bicycles each day.",
"Travelers rested at the inn beside the river.",
"The garden was full of bees and butterflies.",
"She solved the difficult puzzle in a few minutes.",
"The captain steered the ship through the narrow channel.",
"Fresh vegetables were sold at the morning market.",
"The painter mixed blue and yellow to make green.",
"A long line formed outside the new cinema.",
"The teacher explained the lesson very clearly.",
"Thunder rolled across the dark evening sky.",
"The baker arranged the cakes in the window.",
"Students gathered in the hall for the ceremony.",
"A small lamp lit the corner of the quiet room.",
"The hikers reached the summit before noon.",
"She wrote her name carefully at the top of the page.",
"The clockmaker repaired the tiny gears by hand.",
"Fish swam slowly beneath the clear blue water.",
"The gardener watered the plants every morning.",
"A warm breeze carried the scent of flowers.",
"The library opened early on weekday mornings.",
"He read the newspaper over breakfast each day.",
"The puppy chased its tail around the room.",
"Snowflakes melted as they touched the warm ground.",
"The choir sang softly in the candlelit church.",
"A tall tree shaded the bench in the park.",
"The chef prepared a simple meal for the guests.",
"Rain clouds gathered over the distant hills.",
"The young artist sketched the busy street scene.",
"Books lined every wall of the old study.",
"The ferry crossed the harbor twice each hour.",
"She hummed a tune while she washed the dishes.",
"The carpenter measured the wood before cutting it.",
"A flock of geese flew south for the winter.",
"The shop sold candles of every color and size.",
"He planted an apple tree behind the house.",
"The lantern swayed gently in the evening wind.",
"Students practiced the new words until they knew them.",
"The baker opened the shop before the sun rose.",
"A narrow path led down to the sandy beach.",
"The violinist played a slow and gentle song.",
"Fresh milk was delivered to the door each morning.",
"The old bridge creaked under the heavy cart.",
"She folded the letter and placed it in an envelope.",
"The garden gate opened with a soft squeak.",
"A bright kite rose high above the green field.",
"The teacher wrote the date on the board.",
"Rainwater collected in the barrel by the wall.",
"The sailors watched the lighthouse flash at night.",
"He counted the coins and put them in a jar.",
"The bakery sold out of bread by early afternoon.",
"A quiet stream ran behind the small cottage.",
"The students listened carefully to the story.",
"Morning fog hung low over the sleeping town.",
"The clock tower chimed twelve times at noon.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, default=1, choices=[1, 2])
    ap.add_argument("--input", default=None, help="optional textfile, one sentence per line")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["random", "word_disjoint"], default="random",
                    help="word_disjoint = every test sentence contains a word NEVER seen "
                         "in training (tests generalisation, not memorisation)")
    ap.add_argument("--holdout_frac", type=float, default=0.15,
                    help="word_disjoint only: fraction of the vocabulary to hold out")
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            sents = [ln.strip() for ln in f if ln.strip()]
    else:
        sents = list(BUILTIN)
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents]

    fwd, back, engine = get_translators(args.grade)

    pairs, dropped = [], 0
    for s in sents:
        brl = fwd(s)
        rt = back(brl).strip()
        # round-trip accuracy filter (case-insensitive, punctuation kept)
        if rt.lower() == s.lower():
            pairs.append({"print": s, "braille": brl})
        else:
            dropped += 1

    rng = random.Random(args.seed)
    extra = {}
    def words_of(s):
        return re.findall(r"[a-z]+", s.lower())

    if args.split == "word_disjoint":
        # Hold out a fraction of the vocabulary; any sentence containing a
        # held-out word goes to TEST, so every test sentence has >=1 word the
        # model never saw in training. This tests generalisation.
        vocab = sorted({w for p in pairs for w in words_of(p["print"])})
        rng.shuffle(vocab)
        n_hold = max(1, int(len(vocab) * args.holdout_frac))
        held = set(vocab[:n_hold])
        train, test = [], []
        for p in pairs:
            (test if held & set(words_of(p["print"])) else train).append(p)
        train_vocab = {w for p in train for w in words_of(p["print"])}
        # sanity: every test sentence must carry an unseen word
        unseen_ok = all(set(words_of(p["print"])) - train_vocab for p in test)
        extra = {"split": "word_disjoint", "held_out_words": n_hold,
                 "vocab": len(vocab), "every_test_has_unseen_word": unseen_ok}
    else:
        rng.shuffle(pairs)
        n_test = max(1, int(len(pairs) * args.test_frac))
        test, train = pairs[:n_test], pairs[n_test:]
        extra = {"split": "random"}

    if not train or not test:
        print(f"[prep] WARNING: empty split (train={len(train)}, test={len(test)}); "
              f"try a smaller --holdout_frac")

    os.makedirs(args.out, exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"engine": engine, "grade": args.grade,
            "kept": len(pairs), "dropped_by_roundtrip": dropped,
            "train": len(train), "test": len(test), **extra}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[prep]", json.dumps(meta))
    print(f"[prep] wrote {args.out}/train.jsonl and {args.out}/test.jsonl")

if __name__ == "__main__":
    main()
