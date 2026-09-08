"""ReAct 格式的带工具数学评测（配合 rlvr_react 权重）。

模型每轮输出：
    Thought: <思考>
    Action: <工具名>
    Action Input: <参数 JSON>
工具结果以 tool 消息（Observation）回填，直到模型不再输出 Action。

Usage:
  python scripts/eval_react_math.py --weight out/rlvr_react_768.pth \
      --data dataset/gsm8k/test.jsonl --n 100
"""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
import eval_tool_math as etm  # noqa: E402  (复用 execute_tool/extract_answer/load_samples)

SYSTEM_REACT = None  # 从训练数据里取，保证训评一致


def parse_action(text):
    """从生成文本解析 (name, args)。无 Action 返回 None；JSON 坏返回 (name, None)。"""
    m = re.search(r'Action:\s*([A-Za-z_]\w*)', text)
    if not m:
        return None
    name = m.group(1)
    mi = re.search(r'Action Input:\s*(\{.*)', text[m.end():], re.DOTALL)
    if not mi:
        return (name, None)
    raw = mi.group(1).strip()
    try:
        return (name, json.loads(raw))
    except Exception:
        # 截到最后一个 } 再试
        j = raw.rfind('}')
        if j > 0:
            try:
                return (name, json.loads(raw[:j + 1]))
            except Exception:
                pass
    return (name, None)


def rollout(model, tokenizer, question, args):
    messages = [{"role": "system", "content": SYSTEM_REACT},
                {"role": "user", "content": question + etm.INSTRUCTION}]
    state = {}
    n_calls = 0
    fmt_ok = True
    trace = []
    final_text = ''
    for _ in range(args.max_turns):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
        input_ids = tokenizer(text, return_tensors='pt',
                              truncation=True, max_length=2048).input_ids.to(args.device)
        greedy = args.temperature <= 0
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=not greedy,
                temperature=1.0 if greedy else args.temperature,
                top_p=1.0 if greedy else args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        final_text = gen
        parsed = parse_action(gen)
        if parsed is None:
            break
        name, call_args = parsed
        if call_args is None:
            fmt_ok = False
            break
        n_calls += 1
        result = etm.execute_tool(name, call_args, state)
        trace.append({'call': {'name': name, 'arguments': call_args}, 'result': result})
        messages.append({"role": "assistant", "content": gen})
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    return final_text, n_calls, fmt_ok, trace


def main():
    global SYSTEM_REACT
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='/root/autodl-tmp/minimind/out/rlvr_react_768.pth')
    parser.add_argument('--data', default='/root/autodl-tmp/minimind/dataset/gsm8k/test.jsonl')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--max_turns', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dump', default='')
    args = parser.parse_args()

    # 与训练数据一致的 system prompt
    SYSTEM_REACT = json.loads(open('/root/autodl-tmp/minimind/dataset/react_sft.jsonl').readline())['conversations'][0]['content']

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/minimind/model')
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(args.weight, map_location=args.device), strict=False)
    model = model.half().to(args.device).eval()

    samples = etm.load_samples(args.data, args.n)
    correct = used_tool = fmt_ok_n = 0
    total_calls = 0
    results = []
    for i, s in enumerate(samples):
        final_text, n_calls, fmt_ok, trace = rollout(model, tokenizer, s['q'], args)
        pred = etm.extract_answer(final_text)
        gold = etm.extract_answer(s['a'])
        try:
            ok = pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-4
        except ValueError:
            ok = False
        correct += ok
        used_tool += n_calls > 0
        fmt_ok_n += fmt_ok
        total_calls += n_calls
        results.append({'q': s['q'][:200], 'gold': gold, 'pred': pred,
                        'correct': ok, 'n_calls': n_calls, 'trace': trace,
                        'final': final_text[-300:]})
        print(f'[{i+1}/{len(samples)}] gold={gold} pred={pred} '
              f'{"✓" if ok else "✗"} tools={n_calls}', flush=True)

    n = len(samples)
    print(f'\nAccuracy: {correct}/{n} = {correct/n:.2%}')
    print(f'使用工具样本: {used_tool}/{n} ({used_tool/n:.0%})，平均调用 {total_calls/n:.1f} 次')
    print(f'工具格式始终合法: {fmt_ok_n}/{n}')
    tag = os.path.basename(args.data).split('.')[0]
    dump = args.dump or f'/root/autodl-tmp/minimind/out/react_eval_{tag}.json'
    with open(dump, 'w') as f:
        json.dump({'acc': correct / n, 'n': n, 'used_tool': used_tool,
                   'avg_calls': total_calls / n, 'results': results}, f, ensure_ascii=False, indent=1)
    print(f'明细 -> {dump}')


if __name__ == '__main__':
    main()
