"""工具调用 SFT 数据：数学题的"草稿箱 + 计算器"工具轨迹。

教模型：多步题先 scratchpad_write 写计划 → 每步 calculate 调用（不再心算）→
汇总得最终答案；简单题部分直接作答（学会判断何时不用工具），另有少量
通用工具样本保证能力不只限于数学。

数据格式与 SFTDataset 一致：system 带 tools 字段，assistant 用 tool_calls
字段（JSON 字符串），工具结果用 role=tool。

Usage: python scripts/build_tool_sft.py [--n 12000] [--dst dataset/tool_sft.jsonl]
"""
import argparse
import importlib.util
import json
import random
import sys

sys.path.append('/root/autodl-tmp/minimind')


def _load(mod_path, name):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


medium_mod = _load('/root/autodl-tmp/minimind/scripts/build_medium_sft.py', 'bm')
easy_mod = _load('/root/autodl-tmp/minimind/scripts/build_easy_rlvr.py', 'be')

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'

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

TOOL_JSON = json.dumps(TOOLS, ensure_ascii=False)
SYSTEM = "你是minimind，一个小巧但有用的语言模型。可以使用工具时优先借助工具保证计算准确。"

STEP_RE = None  # lazy


def parse_steps(steps):
    """从 medium_sft 的步骤行解析 (label, a, op, b)。"""
    import re
    out = []
    for s in steps:
        m = re.search(r':\s*(-?\d+)\s*([+\-x/])\s*(-?\d+)\s*=', s)
        if not m:
            return None
        label = s.split(':')[0].split('：')[0].strip()
        out.append((label, m.group(1), m.group(2), m.group(3)))
    return out


def calc(a, op, b):
    return {'+': lambda: a + b, '-': lambda: a - b, 'x': lambda: a * b, '/': lambda: a // b}[op]()


def tool_msg(name, args, result):
    call = {"role": "assistant", "content": "",
            "tool_calls": json.dumps([{"name": name, "arguments": args}], ensure_ascii=False)}
    return [call, {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}]


def make_medium_trace(rng, cn):
    gen = rng.choice(medium_mod.GENS)
    q, steps, ans = gen(rng, cn)
    parsed = parse_steps(steps)
    if parsed is None:
        return None
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q + INSTRUCTION}]
    # 70% 先写计划到草稿箱
    if rng.random() < 0.7:
        plan = "；".join(f"{lbl} {a}{op}{b}" for lbl, a, op, b in parsed)
        conv += tool_msg("scratchpad_write", {"content": f"解题计划：{plan}"}, {"ok": True})
        conv += tool_msg("scratchpad_read", {}, {"content": f"解题计划：{plan}"})
    lines = []
    for lbl, a, op, b in parsed:
        expr = f"{a}{op}{b}" if op != 'x' else f"{a}*{b}"
        r = calc(int(a), op if op != 'x' else 'x', int(b)) if op != '/' else int(a) // int(b)
        conv[-1] = conv[-1]  # no-op
        conv += tool_msg("calculate", {"expression": expr.replace('x', '*')}, {"result": str(r)})
        lines.append(f"{lbl}: {a} {op} {b} = {r}")
    tail = "。" if cn else "."
    conv.append({"role": "assistant", "content": "\n".join(lines) + f"\n#### {ans}"})
    return conv


def make_easy_direct(rng, cn):
    q, ans = easy_mod.one_step(rng) if rng.random() < 0.5 else easy_mod.two_step(rng)
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q + INSTRUCTION},
            {"role": "assistant", "content": f"#### {ans}"}]
    return conv


def make_easy_tool(rng, cn):
    """两步题用一次 calculate（链式表达式），一步题也统一走一次工具。"""
    if rng.random() < 0.5:
        q, ans = easy_mod.two_step(rng)
        kind = rng.choice(['add_sub', 'mul_sub', 'sub_sub', 'add_add'])
        # 从题面无法可靠反推表达式，改为 regenerate with known structure
        qq, aa = None, None
        r2 = rng
        if kind == 'add_sub':
            a, b = r2.randint(20, 300), r2.randint(10, 200)
            s = a + b
            c = r2.randint(5, s - 5)
            expr, res = f"{a}+{b}-{c}", s - c
            tpl = "add_sub"
        elif kind == 'mul_sub':
            a, b = r2.randint(4, 15), r2.randint(3, 9)
            c = r2.randint(1, a * b - 1)
            expr, res = f"{a}*{b}-{c}", a * b - c
            tpl = "mul_sub"
        elif kind == 'sub_sub':
            a = r2.randint(200, 999)
            b = r2.randint(30, 150)
            c = r2.randint(10, min(100, a - b - 1))
            expr, res = f"{a}-{b}-{c}", a - b - c
            tpl = 'sub_sub'
        else:
            a, b, c = r2.randint(15, 200), r2.randint(15, 200), r2.randint(15, 200)
            expr, res = f"{a}+{b}+{c}", a + b + c
            tpl = 'add_add'
        lang = 'cn' if cn else 'en'
        name = r2.choice(easy_mod.CN_NAMES if cn else easy_mod.NAMES)
        q = easy_mod._render(lang, tpl, n=name, a=a, b=b, c=c, p=1, cp=1)
        ans = res
        conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
                {"role": "user", "content": q + INSTRUCTION}]
        conv += tool_msg("calculate", {"expression": expr}, {"result": str(ans)})
        conv.append({"role": "assistant", "content": f"算得结果是 {ans}。\n#### {ans}"})
        return conv
    q, ans = easy_mod.one_step(rng)
    import re
    m = re.search(r'(-?\d+)\s*([\+\-])\s*(-?\d+)', q)
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q + INSTRUCTION}]
    if m:
        expr = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        conv += tool_msg("calculate", {"expression": expr}, {"result": str(ans)})
        conv.append({"role": "assistant", "content": f"结果是 {ans}。\n#### {ans}"})
    else:
        conv.append({"role": "assistant", "content": f"#### {ans}"})
    return conv


GENERIC_Q = [
    ("现在几点了？告诉我日期和时间。", "get_current_time", {}, None),
    ("今天是几号？", "get_current_time", {}, None),
    ("帮我看看现在的时间。", "get_current_time", {}, None),
]


def make_generic(rng, cn=False):
    q, name, args, _ = rng.choice(GENERIC_Q)
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q}]
    conv += tool_msg(name, args, {"datetime": "2026-09-07 12:00:00"})
    conv.append({"role": "assistant", "content": "现在是 2026-09-07 12:00:00。"})
    return conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--dst', default='dataset/tool_sft.jsonl')
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quota = [(make_medium_trace, int(args.n * 0.45)),
             (make_easy_tool, int(args.n * 0.25)),
             (make_easy_direct, int(args.n * 0.20)),
             (make_generic, int(args.n * 0.10))]
    out, seen = [], set()
    for gen, n in quota:
        made, attempts = 0, 0
        while made < n and attempts < n * 60:
            attempts += 1
            cn = rng.random() < 0.5
            conv = gen(rng, cn)
            if conv is None:
                continue
            key = conv[1]['content']
            if key in seen:
                continue
            seen.add(key)
            out.append({'conversations': conv})
            made += 1
        if made < n:
            print(f'[warn] {gen.__name__} 组合空间不足：只生成 {made}/{n}')
    rng.shuffle(out)
    with open(args.dst, 'w') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'写出 {len(out)} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
