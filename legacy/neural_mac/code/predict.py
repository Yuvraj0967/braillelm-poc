"""Run the released BrailleLM Qwen PEFT adapter on one Unicode braille string."""

import argparse

import torch


PROMPT = "Translate the following braille to English text.\nBraille: {brl}\nEnglish:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--braille", required=True)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else
             "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, extra_special_tokens={})
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.ckpt).to(device).eval()
    encoded = tokenizer(PROMPT.format(brl=args.braille), return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new,
            do_sample=False,
            num_beams=args.num_beams,
            early_stopping=args.num_beams > 1,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().split("\n")[0].strip()
    print(output)


if __name__ == "__main__":
    main()
