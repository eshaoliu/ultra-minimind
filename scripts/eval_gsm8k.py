"""GSM8K eval on MiniMind SFT checkpoint.

Usage (from trainer/): python ../scripts/eval_gsm8k.py [--n 100] [--weight ../out/full_sft_768.pth]
"""
import argparse
import json
import re
import sys

sys.path.append('/root/autodl-tmp/minimind')
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer


def extract_answer(text):
    # 优先匹配 CoT 里的 "#### x" 标记；否则取最后一个数字
    m = re.findall(r'####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m[-1].replace(',', '')
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--weight', default='/root/autodl-tmp/minimind/out/full_sft_768.pth')
    parser.add_argument('--data', default='/root/autodl-tmp/minimind/dataset/gsm8k/test.jsonl')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/minimind/model')
    model = MiniMindForCausalLM(lm_config)
    weights = torch.load(args.weight, map_location='cuda')
    model.load_state_dict(weights, strict=False)
    model = model.to('cuda').eval()

    samples = [json.loads(l) for l in open(args.data)][:args.n]
    correct = 0
    results = []
    for i, s in enumerate(samples):
        q = s['question'].strip()
        messages = [{"role": "user", "content": q + "\nPlease reason step by step and give the final answer after ####."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen)
        gold = extract_answer(s['answer'])
        ok = pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-4
        correct += ok
        results.append({'question': q, 'gold': gold, 'pred': pred, 'correct': ok, 'gen': gen[-300:]})
        print(f'[{i+1}/{len(samples)}] gold={gold} pred={pred} {"✓" if ok else "✗"}', flush=True)

    acc = correct / len(samples)
    print(f'\nAccuracy: {correct}/{len(samples)} = {acc:.2%}')
    with open('/root/autodl-tmp/minimind/out/gsm8k_eval.json', 'w') as f:
        json.dump({'acc': acc, 'n': len(samples), 'results': results}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
