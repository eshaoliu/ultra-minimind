"""带工具（草稿箱/calculate）的数学评测。

对每道题做 multi-turn rollout：模型可调用 scratchpad_write / scratchpad_read /
calculate / get_current_time，工具执行结果以 <tool_response> 回填，直到模型
不再发工具调用或达到 max_turns，再从最终文本抽 #### 后的答案判分。

数据格式自适应：gsm8k/ladder（question+answer / conversations）、ape210k 与
math23k（original_text+ans，.json 流）、cmath（input+golden，.jsonl）。
判分容错：整数/小数/分数 a/b/百分数/千分位逗号，均归一为 float 比较。

Usage:
  python scripts/eval_tool_math.py --weight out/rlvr_tool_768.pth \
      --data dataset/gsm8k/test.jsonl --n 100
  python scripts/eval_tool_math.py --weight out/rlvr_tool_768.pth \
      --data dataset/ladder_medium.jsonl --n 20
  python scripts/eval_tool_math.py --weight out/full_sft_768.pth \
      --data dataset/cmath/cmath_dev.jsonl --n 100
  python scripts/eval_tool_math.py --weight out/cnmath_sft_v1_768.pth \
      --data dataset/ape210k/test.ape.json --n 200
"""
import argparse
import json
import os
import random
import re
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'
CH_INSTRUCTION = '\n请一步步计算，最后在 #### 后给出最终数字答案（例如：#### 42）。'
SYSTEM = "你是minimind，一个小巧但有用的语言模型。可以使用工具时优先借助工具保证计算准确。"

TOOLS = [
    {"type": "function", "function": {
        "name": "scratchpad_write",
        "description": "把计划或中间结果写入草稿箱，之后可以随时读取，避免忘记",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "要记录的内容，如计算计划、中间结果"}},
            "required": ["content"]}}},
    {"type": "function", "function": {
        "name": "scratchpad_read",
        "description": "读取草稿箱里记录的全部内容",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算算术表达式的准确结果，支持加减乘除和括号",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "算术表达式，如 246+45、13*8、(20+15)/5"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前的日期和时间",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SAFE_EXPR = re.compile(r'^[\d\s+\-*/().x×÷]+$')


def safe_calc(expr):
    expr = expr.replace(',', '')  # GSM8K 金额含千分位逗号，如 80,000
    expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/')
    if not SAFE_EXPR.match(expr):
        raise ValueError('illegal expression')
    r = eval(expr, {'__builtins__': {}}, {})
    if isinstance(r, float) and r.is_integer():
        r = int(r)
    return str(r)


def execute_tool(name, args, state):
    if name == 'scratchpad_write':
        state['scratchpad'] = str(args.get('content', ''))
        return {"ok": True}
    if name == 'scratchpad_read':
        return {"content": state.get('scratchpad', '')}
    if name == 'calculate':
        try:
            return {"result": safe_calc(str(args.get('expression', '')))}
        except Exception as e:
            return {"error": f"计算失败: {str(e)[:60]}"}
    if name == 'get_current_time':
        return {"datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    return {"error": f"未知工具: {name}"}


def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            d = json.loads(m.strip())
            if isinstance(d, dict) and 'name' in d:
                calls.append(d)
        except Exception:
            pass
    return calls


def extract_answer(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?)', text)
    if m:
        return m[-1].replace(',', '').strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


def to_float(s):
    """把整数/小数/分数/百分数/全角数字归一为 float；失败返回 None。"""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace('，', ',').replace('．', '.').replace('。', '.')
    s = s.translate(str.maketrans('０１２３４５６７８９－＋．％', '0123456789-+.%'))
    s = s.replace(',', '').replace('％', '%')
    m = re.fullmatch(r'\(?\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*\)?', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else None
    m = re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*%?', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else None
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def rollout(model, tokenizer, question, instr, args, open_thinking=True):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question + instr}]
    state = {}
    n_calls = 0
    fmt_ok = True
    trace = []
    final_text = ''
    for _ in range(args.max_turns):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=TOOLS, open_thinking=open_thinking)
        input_ids = tokenizer(text, return_tensors='pt',
                              truncation=True, max_length=2048).input_ids.to(args.device)
        with torch.no_grad():
            greedy = args.temperature <= 0
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=not greedy,
                temperature=1.0 if greedy else args.temperature,
                top_p=1.0 if greedy else args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        final_text = gen
        calls = parse_tool_calls(gen)
        raw_blocks = re.findall(r'<tool_call>.*?</tool_call>', gen, re.DOTALL)
        if raw_blocks and not calls:
            fmt_ok = False
        if not calls:
            break
        messages.append({"role": "assistant", "content": gen})
        for c in calls:
            n_calls += 1
            result = execute_tool(c.get('name', ''), c.get('arguments', {}) or {}, state)
            trace.append({'call': c, 'result': result})
            messages.append({"role": "tool",
                             "content": json.dumps(result, ensure_ascii=False)})
    return final_text, n_calls, fmt_ok, trace


def iter_json_records(path):
    """流式读取 .json 文件里的连续 JSON 对象（APE210K / Math23K 就是这种格式）。"""
    text = open(path, encoding='utf-8').read()
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in ' \r\n\t':
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        yield obj


def load_samples(path, n=0, seed=0):
    samples = []
    lower = path.lower()
    if lower.endswith('.jsonl'):
        recs = (json.loads(line) for line in open(path, encoding='utf-8') if line.strip())
    else:
        recs = iter_json_records(path)
    limit = 0 if (n and seed) else n  # 要均匀抽样时先收全量再 sample
    for d in recs:
        q = a = None
        if 'question' in d:  # gsm8k
            q, a = d['question'], d.get('answer', '')
        elif 'conversations' in d:  # ladder_medium / 已转好的 SFT 行
            q = d['conversations'][0]['content']
            a = d.get('answer', '')
        elif 'original_text' in d:  # ape210k / math23k
            q, a = d['original_text'], d.get('ans', '')
        elif 'input' in d:  # cmath
            q, a = d['input'], d.get('golden', '')
        if q is None or not str(q).strip():
            continue
        samples.append({'q': str(q).strip(), 'a': str(a).strip()})
        if limit and len(samples) >= limit:
            break
    if n and seed and len(samples) > n:
        samples = random.Random(seed).sample(samples, n)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='/root/autodl-tmp/minimind/out/rlvr_tool_768.pth')
    parser.add_argument('--data', default='/root/autodl-tmp/minimind/dataset/gsm8k/test.jsonl')
    parser.add_argument('--n', type=int, default=100, help='抽样条数，0=全量')
    parser.add_argument('--seed', type=int, default=0, help='>0 时按 seed 均匀抽样 n 条')
    parser.add_argument('--instr', choices=['auto', 'zh', 'en'], default='auto',
                        help='提问指令语言：auto 按文件名判断（gsm8k/medium 用 en，其余 zh）')
    parser.add_argument('--max_turns', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--k', type=int, default=1, help='每题采样数：1=greedy，>1=pass@k')
    parser.add_argument('--open_thinking', type=int, default=1,
                        help='生成提示是否以 <think> 开头（0/1）')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dump', default='')
    args = parser.parse_args()
    base = os.path.basename(args.data).lower()
    if args.instr == 'auto':
        args.instr = INSTRUCTION if ('gsm8k' in base or 'medium' in base) else CH_INSTRUCTION
    else:
        args.instr = INSTRUCTION if args.instr == 'en' else CH_INSTRUCTION

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/minimind/model')
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(args.weight, map_location=args.device), strict=False)
    model = model.half().to(args.device).eval()

    samples = load_samples(args.data, args.n, args.seed)
    n = len(samples)
    st = {'ok_any': 0, 'all_k': 0, 'rollouts': 0, 'hit': 0,
          'tools': 0, 'fmt_ok': 0, 'has_ans_mark': 0,
          'has_think': 0, 'think_len_ok': 0, 'calls': 0}
    results = []
    for i, s in enumerate(samples):
        n_ok = 0
        rolls = []
        for _ in range(args.k):
            final_text, n_calls, fmt_ok, trace = rollout(
                model, tokenizer, s['q'], args.instr, args,
                open_thinking=bool(args.open_thinking))
            pred = extract_answer(final_text)
            pred_v, gold_v = to_float(pred), to_float(s['a'])
            ok = pred_v is not None and gold_v is not None and \
                abs(pred_v - gold_v) < 1e-4 * max(1.0, abs(gold_v))
            n_ok += ok
            st['rollouts'] += 1
            st['hit'] += ok
            st['tools'] += n_calls > 0
            st['fmt_ok'] += fmt_ok
            st['has_ans_mark'] += '####' in final_text
            if '</think>' in final_text:
                st['has_think'] += 1
                tc = final_text.split('</think>')[0].strip()
                st['think_len_ok'] += 1 if 20 <= len(tc) <= 300 else 0
            st['calls'] += n_calls
            rolls.append({'pred': pred, 'pred_num': pred_v,
                          'correct': ok, 'n_calls': n_calls,
                          'final': final_text[-300:], 'trace': trace})
        ok_any = n_ok > 0
        st['ok_any'] += ok_any
        st['all_k'] += n_ok == args.k
        results.append({'q': s['q'][:200], 'gold': s['a'],
                        'gold_num': to_float(s['a']), 'n_ok': n_ok,
                        'ok_any': ok_any, 'rollouts': rolls})
        print(f'[{i+1}/{n}] gold={s["a"]} n_ok={n_ok}/{args.k} '
              f'{"OK" if ok_any else "xx"}', flush=True)

    rt = max(1, st['rollouts'])
    print(f'\nacc@{args.k}(any): {st["ok_any"]}/{n} = {st["ok_any"]/n:.2%}')
    print(f'k 条全对: {st["all_k"]}/{n}')
    print(f'rollout 命中率: {st["hit"]}/{st["rollouts"]} = {st["hit"]/rt:.2%}')
    print(f'用工具 rollout: {st["tools"]}/{st["rollouts"]}，'
          f'平均调用 {st["calls"]/max(1, n):.1f} 次/题')
    print(f'工具格式合法: {st["fmt_ok"]}/{st["rollouts"]}')
    print(f'含 ####: {st["has_ans_mark"]}/{st["rollouts"]}')
    print(f'含 </think>: {st["has_think"]}/{st["rollouts"]}，'
          f'think 20~300 字符: {st["think_len_ok"]}/{st["rollouts"]}')
    wstem = os.path.basename(args.weight).split('.')[0]
    dstem = os.path.basename(args.data).split('.')[0]
    dump = args.dump or f'/root/autodl-tmp/minimind/out/thinktool_eval_{wstem}_{dstem}_k{args.k}.json'
    with open(dump, 'w') as f:
        json.dump({'acc_any': st['ok_any'] / n, 'all_k': st['all_k'],
                   'rollout_hit': st['hit'] / rt, 'n': n,
                   'used_tool': st['tools'], 'avg_calls': st['calls'] / n,
                   'fmt_ok': st['fmt_ok'], 'has_ans_mark': st['has_ans_mark'],
                   'has_think': st['has_think'],
                   'think_len_ok': st['think_len_ok'],
                   'instr': args.instr[:40],
                   'results': results}, f, ensure_ascii=False, indent=1)
    print(f'明细 -> {dump}')


if __name__ == '__main__':
    main()
