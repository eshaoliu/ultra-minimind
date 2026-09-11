"""GSM8K eval on MiniMind pretrain (base) checkpoint.

基座模型无 chat template，用 4-shot 纯文本补全格式：
    Question: ...
    Answer: <推理> #### <数字>

Usage (from repo root): python scripts/eval_gsm8k_pretrain.py \
    --weight out/pretrain_1536.pth --hidden_size 1536 --num_hidden_layers 32 \
    --data /root/gpufree-data/dataset/gsm8k/gsm8k_test.jsonl
"""
import argparse
import json
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

FEWSHOT = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * $2 = $<<9*2=18>>18 every day at the farmer's market. #### 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2 / 2 = <<2/2=1>>1 bolt of white fiber. So the total is 2 + 1 = <<2+1=3>>3 bolts. #### 3

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The cost of the house and repairs came out to 80000 + 50000 = $<<80000+50000=130000>>130000. He increased the value of the house by 80000 * 1.5 = $<<80000*1.5=120000>>120000. So the new value of the house is 80000 + 120000 = $<<80000+120000=200000>>200000. So he made a profit of 200000 - 130000 = $<<200000-130000=70000>>70000. #### 70000

Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Answer: If each chicken eats 3 cups of feed a day, the total daily feed is 20 * 3 = <<20*3=60>>60 cups. Morning plus afternoon feed is 15 + 25 = <<15+25=40>>40 cups. So the final meal needs 60 - 40 = <<60-40=20>>20 cups. #### 20

"""


def extract_answer(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m[-1].replace(',', '')
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='out/pretrain_1536.pth')
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--num_hidden_layers', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=32768)
    parser.add_argument('--tokenizer', default='model/tokenizer_en')
    parser.add_argument('--data', default='/root/gpufree-data/dataset/gsm8k/gsm8k_test.jsonl')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--start', type=int, default=0, help='从第几条样本继续（上次中断时用）')
    parser.add_argument('--out', default='out/gsm8k_eval_pretrain.json')
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               vocab_size=args.vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=True)
    model = model.half().cuda().eval()

    samples = [json.loads(l) for l in open(args.data)]
    prompts = [FEWSHOT + f"Question: {s['question'].strip()}\nAnswer:" for s in samples]
    golds = [extract_answer(s['answer']) for s in samples]

    results = [None] * len(samples)
    correct = 0
    done = 0
    for lo in range(args.start, len(samples), args.batch_size):
        batch_prompts = prompts[lo:lo + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        input_ids = enc.input_ids.cuda()
        attn = enc.attention_mask.cuda()
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attn, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)
        for j in range(len(batch_prompts)):
            gen = tokenizer.decode(out[j][input_ids.shape[1]:], skip_special_tokens=True)
            gen = gen.split('\nQuestion:')[0].split('\n\n')[0]  # 截断到下一题
            idx = lo + j
            pred = extract_answer(gen)
            ok = pred is not None and golds[idx] is not None and abs(float(pred) - float(golds[idx])) < 1e-4
            correct += ok
            results[idx] = {'question': samples[idx]['question'], 'gold': golds[idx],
                            'pred': pred, 'correct': ok, 'gen': gen[-300:]}
            done += 1
        print(f'[{done}/{len(samples)}] acc_so_far={correct/done:.2%}', flush=True)
        del out, input_ids, attn
        torch.cuda.empty_cache()

    acc = correct / done
    print(f'\nAccuracy: {correct}/{done} = {acc:.2%}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'acc': acc, 'n': done, 'start': args.start, 'weight': args.weight,
                   'results': [r for r in results if r is not None]},
                  f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
