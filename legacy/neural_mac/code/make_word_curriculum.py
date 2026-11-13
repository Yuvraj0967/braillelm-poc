"""Generate isolated-word neural training pairs from train text only."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

source = [json.loads(x) for x in Path('data_all/train.jsonl').open()]
counts = collections.Counter(w.lower() for row in source for w in re.findall(r"\b[A-Za-z]{2,20}\b", row['print']))
words = sorted(counts)
cmd = ['lou_translate', '-d', 'unicode.dis', 'en-ueb-g1.ctb']
forward = subprocess.run(cmd, input='\n'.join(words)+'\n', text=True,
                         capture_output=True, check=True).stdout.splitlines()
backward = subprocess.run(cmd[:3]+['-b']+cmd[3:], input='\n'.join(forward)+'\n',
                          text=True, capture_output=True, check=True).stdout.splitlines()
assert len(words) == len(forward) == len(backward)
out = Path('data_all/word_curriculum.jsonl')
with out.open('w') as f:
    for word, braille, reverse in zip(words, forward, backward):
        if word != reverse:
            continue
        f.write(json.dumps({'print': word, 'braille': braille,
                            'source': 'word from WikiText train or original train; liblouis Grade 1'},
                           ensure_ascii=False)+'\n')
print(json.dumps({'train_vocab':len(words),'kept':sum(1 for _ in out.open()),
                  'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
