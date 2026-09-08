import argparse
import json
import re
import sys

sys.path.append("/root/autodl-tmp/minimind")
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer


def extract_answer(text):
    m = re.findall(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument(
        "--weight",
        default="/root/autodl-tmp/minimind/out/warmup_sft_v0_768.pth",
    )
    ap.add_argument("--out", default="/root/autodl-tmp/minimind/out/gsm8k_pass8_warmup_v0.json")
    args = ap.parse_args()

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/minimind/model")
    model = MiniMindForCausalLM(lm_config)
    weights = torch.load(args.weight, map_location="cuda")
    model.load_state_dict(weights, strict=False)
    model = model.to("cuda").eval()

    samples = [json.loads(l) for l in open("/root/autodl-tmp/minimind/dataset/gsm8k/test.jsonl")][: args.n]
    q_correct = 0
    total_correct = 0
    results = []
    for i, s in enumerate(samples):
        q = s["question"].strip()
        messages = [{"role": "user", "content": q + "\nPlease reason step by step and give the final answer after ####."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, open_thinking=False
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        input_ids = input_ids.expand(args.samples, -1)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temp,
                top_p=args.top_p,
                num_return_sequences=args.samples,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gold = extract_answer(s["answer"])
        preds = []
        plen = input_ids.shape[1]
        for j in range(args.samples):
            gen = tokenizer.decode(out[j * args.samples + j][plen:], skip_special_tokens=True)
            preds.append(extract_answer(gen))
        ok = [p is not None and gold is not None and abs(float(p) - float(gold)) < 1e-4 for p in preds]
        any_ok = any(ok)
        q_correct += int(any_ok)
        total_correct += sum(ok)
        results.append({"question": q, "gold": gold, "preds": preds, "n_correct": sum(ok)})
        if (i + 1) % 5 == 0 or any_ok:
            print(f"[{i+1}/{len(samples)}] gold={gold} pass={any_ok} n_ok={sum(ok)}", flush=True)

    n = len(samples)
    pass8 = q_correct / n
    sample_acc = total_correct / (n * args.samples)
    print(f"\npass@{args.samples}: {q_correct}/{n} = {pass8:.2%}", flush=True)
    print(f"sample_acc: {total_correct}/{n*args.samples} = {sample_acc:.2%}", flush=True)
    with open(args.out, "w") as f:
        json.dump(
            {"pass_k": pass8, "sample_acc": sample_acc, "n": n, "k": args.samples, "results": results},
            f,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
