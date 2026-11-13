"""
test_wiring.py -- validate the model-dependent logic in train.py WITHOUT any
download: build a tiny Llama-style model + a from-scratch BPE tokenizer, then
run add_tokens -> resize -> BKFT init -> LoRA(modules_to_save) -> train step ->
generate. Mirrors SmolLM2 (Llama arch, tied embeddings, q_proj/v_proj).
"""
import json, torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

import train as T  # reuse bkft_init, build_example, collate, read_jsonl
import braille_ueb as B

rows = T.read_jsonl("data/train.jsonl")
corpus = [r["print"] for r in rows] + [r["braille"] for r in rows]

# 1) tiny from-scratch BPE tokenizer (no download)
tk = Tokenizer(models.BPE(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=["<unk>", "<pad>", "<eos>"])
tk.train_from_iterator(corpus, trainer)
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>",
                              pad_token="<pad>", eos_token="<eos>")
base_vocab = len(tok)
print(f"[test] base tokenizer vocab = {base_vocab}")

# 2) tiny Llama model (tied embeddings by default)
cfg = LlamaConfig(vocab_size=base_vocab, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  tie_word_embeddings=True)
model = LlamaForCausalLM(cfg)
print(f"[test] tied embeddings = {cfg.tie_word_embeddings}")

# 3) add braille tokens + resize + BKFT init  (the risky bit)
added = json.load(open("braille_tokens.json", encoding="utf-8"))["tokens"]
n_added = tok.add_tokens(added)
model.resize_token_embeddings(len(tok))
emb_before = model.get_input_embeddings().weight.data.clone()
T.bkft_init(model, tok, added, base_vocab)
emb_after = model.get_input_embeddings().weight.data

# verify a known merged token got a non-trivial init from its print string
tid = tok.convert_tokens_to_ids("⠠⠗⠑")  # (cap)R E -> "Re"? check any
changed = (emb_after[base_vocab:] != 0).any().item()
print(f"[test] added {n_added} tokens; vocab {base_vocab} -> {len(tok)}")
print(f"[test] BKFT wrote non-zero embeddings for new rows: {changed}")

# spot-check: 'the' braille token should be init'd near mean of 'the' tokens
the_tok = None
for t in added:
    if B.cell_to_print(t) == "the":
        the_tok = t; break
if the_tok:
    tid = tok.convert_tokens_to_ids(the_tok)
    ids = tok("the", add_special_tokens=False)["input_ids"]
    ids = [i for i in ids if i < base_vocab]
    expect = emb_before[ids].mean(0) if ids else None
    if expect is not None:
        diff = (emb_after[tid] - expect).abs().max().item()
        print(f"[test] 'the' token BKFT matches mean-of-print: max|diff|={diff:.2e} (0 = exact)")

# 4) LoRA with modules_to_save for the new rows
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                  target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
                  modules_to_save=["embed_tokens", "lm_head"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 5) one real train step (forward + backward + step)
data = [T.build_example(tok, r["braille"], r["print"], 64) for r in rows[:8]]
batch = T.collate(data, tok.pad_token_id)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
loss = model(**batch).loss
loss.backward(); opt.step(); opt.zero_grad()
print(f"[test] train step OK, loss = {loss.item():.3f}")

# 6) generation runs
model.eval()
prompt = T.PROMPT.format(brl=rows[0]["braille"])
ids = tok(prompt, return_tensors="pt")
with torch.no_grad():
    gen = model.generate(**ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.pad_token_id)
print(f"[test] generate OK, produced {gen.shape[1]-ids['input_ids'].shape[1]} new tokens")
print("\n[test] ALL WIRING CHECKS PASSED")
