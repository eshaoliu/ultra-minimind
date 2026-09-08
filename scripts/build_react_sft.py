"""工具 SFT ReAct 版：把 v2 的 <tool_call> 轨迹改写成 ReAct 文本协议。

格式（assistant 每轮）：
    Thought: <思考>
    Action: <工具名>
    Action Input: <参数 JSON>
工具结果作为 tool 消息返回（模板包成 <tool_response>，即 Observation）。
最终轮不再有 Action，直接输出结论 + "#### 数字"。

system prompt 内嵌工具说明（不走过模板的 tools 块），直接答样本保持原样。

Usage: python scripts/build_react_sft.py [--n 12000]
"""
import argparse
import importlib.util
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tool_sft as v1  # noqa: E402
import build_tool_sft_v2 as v2  # noqa: E402

INSTRUCTION = v1.INSTRUCTION

SYSTEM_REACT = """你是minimind，一个小巧但有用的语言模型。你可以使用以下工具：

1. scratchpad_write: 把计划或中间结果写入草稿箱，之后可以随时读取。参数: {"content": "要记录的内容"}
2. scratchpad_read: 读取草稿箱里记录的全部内容。参数: {}
3. calculate: 计算算术表达式的准确结果，支持加减乘除和括号。参数: {"expression": "算术表达式，如 246+45"}
4. get_current_time: 获取当前的日期和时间。参数: {}

每轮按如下格式输出：
Thought: 你的思考
Action: 工具名
Action Input: 工具参数的JSON

工具会返回结果（Observation），然后你继续思考下一步。得到足够信息后直接给出最终答案，最后一行写 #### 数字。如果题目很简单，也可以不用工具直接作答。"""


def thought_for(name, args):
    if name == 'scratchpad_write':
        return f"我先把计划写下来：{args.get('content', '')}"
    if name == 'scratchpad_read':
        return "读一下草稿箱，确认刚才记录的内容。"
    if name == 'calculate':
        return f"计算 {args.get('expression', '')} 的结果。"
    if name == 'get_current_time':
        return "我需要知道当前的日期和时间。"
    return "我需要使用工具。"


def to_react(conv):
    out = []
    for m in conv:
        m = dict(m)
        if m['role'] == 'system':
            m['content'] = SYSTEM_REACT
            m.pop('tools', None)
        elif m['role'] == 'assistant' and m.get('tool_calls'):
            tcs = json.loads(m['tool_calls']) if isinstance(m['tool_calls'], str) else m['tool_calls']
            parts = [f"Thought: {thought_for(tc['name'], tc.get('arguments', {}) or {})}\n"
                     f"Action: {tc['name']}\n"
                     f"Action Input: {json.dumps(tc.get('arguments', {}), ensure_ascii=False)}"
                     for tc in tcs]
            m = {'role': 'assistant', 'content': '\n'.join(parts)}
        out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--dst', default='dataset/react_sft.jsonl')
    parser.add_argument('--seed', type=int, default=779)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quota = [(v1.make_medium_trace, int(args.n * 0.24)),   # medium 带工具
             (v2.make_medium_direct, int(args.n * 0.24)),  # medium 直接答
             (v1.make_easy_tool, int(args.n * 0.20)),      # easy 用工具
             (v1.make_easy_direct, int(args.n * 0.29)),    # easy 直接答
             (v2.make_generic, int(args.n * 0.03))]        # 通用工具
    out, seen = [], set()
    stats = {}
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
            out.append({'conversations': to_react(conv)})
            made += 1
        stats[gen.__name__] = made
        if made < n:
            print(f'[warn] {gen.__name__} 组合空间不足：只生成 {made}/{n}')
    rng.shuffle(out)
    with open(args.dst, 'w') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print('配额:', stats)
    print(f'写出 {len(out)} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
