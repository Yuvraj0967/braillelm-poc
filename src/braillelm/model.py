"""BrailleLM v2: a cell-level neural braille reader.

Input   one token per braille cell (64 six-dot cells), no BPE merges.
Body    bidirectional Transformer encoder with rotary position embeddings.
Output  two CTC frames per cell over print characters. CTC learns the
        cell-to-character alignment from data: no rule table, dictionary or
        converter is used inside the model or its decoder.

Reading cells individually removes the failure mode of the LLM arm, where rare
words and names were rewritten into more probable words. The encoder still
sees the whole sentence, so it can use context to repair damaged cells.

One set of weights serves two modes, selected by a learned mode embedding:
"faithful" trusts every dot; "repair" assumes some cells may be damaged.
Keeping them separate stops the repair behaviour from rewriting clean input.

A second learned switch selects the braille grade. Grade 1 spells every letter;
Grade 2 (contracted) lets one cell stand for several letters or a whole word,
so models that read Grade 2 emit more CTC frames per cell (`frames`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from braillelm.braille_g1 import PRINT_CHARS

PAD_IN = 64                      # input ids: 0..63 cells, 64 pad
IN_VOCAB = 65
CTC_BLANK = 0                    # output ids: 0 blank, 1.. PRINT_CHARS
OUT_VOCAB = len(PRINT_CHARS) + 1
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(PRINT_CHARS)}
FRAMES_PER_CELL = 2              # default; Grade 2 models use more (see BrailleReader.frames)
# faithful: every dot is trusted, nothing is "corrected".
# repair:   the input may contain damaged cells; use context to restore them.
MODES = ("faithful", "repair")
GRADES = ("g1", "g2")


def encode_cells(braille):
    ids = []
    for ch in braille:
        o = ord(ch)
        if 0x2800 <= o <= 0x283F:
            ids.append(o - 0x2800)
        elif ch == ' ':
            ids.append(0)
        # anything else (8-dot cells, stray characters) is ignored
    return ids


def encode_print(text):
    return [CHAR_TO_ID[c] for c in text]


def ctc_collapse(frame_ids):
    out, prev = [], CTC_BLANK
    for i in frame_ids:
        if i != prev and i != CTC_BLANK:
            out.append(PRINT_CHARS[i - 1])
        prev = i
    return ''.join(out)


def _rope(q, k):
    # q, k: [B, H, T, D]
    T, D = q.shape[-2], q.shape[-1]
    pos = torch.arange(T, device=q.device, dtype=q.dtype)
    freq = torch.exp(torch.arange(0, D, 2, device=q.device, dtype=q.dtype) * (-math.log(10000.0) / D))
    ang = pos[:, None] * freq[None, :]
    cos, sin = ang.cos(), ang.sin()

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return rot(q), rot(k)


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))


class BrailleReader(nn.Module):
    def __init__(self, d=256, layers=6, heads=4, ff=1024, dropout=0.1, frames=FRAMES_PER_CELL, grades=1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, dropout=dropout, frames=frames, grades=grades)
        self.frames, self.grades = frames, grades
        if grades > 1:
            self.grade = nn.Embedding(grades, d)
        self.emb = nn.Embedding(IN_VOCAB, d, padding_idx=PAD_IN)
        self.mode = nn.Embedding(len(MODES), d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, frames * OUT_VOCAB)

    def forward(self, cells, mode, grade=None):
        """cells [B, T], mode [B], grade [B] -> logits [B, frames * T, OUT_VOCAB]"""
        keep = cells != PAD_IN
        mask = keep[:, None, None, :]
        x = self.emb(cells) + self.mode(mode)[:, None, :]
        if self.grades > 1:
            if grade is None:
                grade = torch.zeros_like(mode)
            x = x + self.grade(grade)[:, None, :]
        for b in self.blocks:
            x = b(x, mask)
        B, T, _ = x.shape
        return self.head(self.norm(x)).view(B, T * self.frames, OUT_VOCAB)

    @torch.no_grad()
    def translate(self, braille_list, mode="faithful", grade="g1", batch=64):
        """Greedy CTC decode. Returns (text, mean per-character confidence)."""
        self.eval()
        mode_id, grade_id = MODES.index(mode), GRADES.index(grade)
        if grade_id >= self.grades:
            raise ValueError(f"this model was not trained for {grade}")
        dev = next(self.parameters()).device
        order = sorted(range(len(braille_list)), key=lambda i: len(braille_list[i]))
        out = [None] * len(braille_list)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            seqs = [encode_cells(braille_list[i]) or [0] for i in idx]
            T = max(map(len, seqs))
            x = torch.full((len(seqs), T), PAD_IN, dtype=torch.long)
            for r, q in enumerate(seqs):
                x[r, :len(q)] = torch.tensor(q)
            m = torch.full((len(seqs),), mode_id, dtype=torch.long)
            g = torch.full((len(seqs),), grade_id, dtype=torch.long)
            prob, ids = self.forward(x.to(dev), m.to(dev), g.to(dev)).float().softmax(-1).max(-1)
            prob, ids = prob.cpu(), ids.cpu()
            for r, i in enumerate(idx):
                n = len(seqs[r]) * self.frames
                fr, pr = ids[r, :n].tolist(), prob[r, :n]
                emitted = pr[ids[r, :n] != CTC_BLANK]
                conf = float(emitted.mean()) if len(emitted) else 1.0
                out[i] = (ctc_collapse(fr), conf)
        return out


def save(model, path, extra=None):
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "print_chars": PRINT_CHARS,
                "extra": extra or {}}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["print_chars"] == PRINT_CHARS
    m = BrailleReader(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["extra"]
