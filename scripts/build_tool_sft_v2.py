"""工具 SFT v2：medium 题一半带工具一半直接答，保住心算能力。

相比 v1（build_tool_sft.py）的变化：
- medium 45% 拆成 带工具 22.5% + 直接答 22.5%（直接答部分复刻 medium_sft 的
  CoT 格式，即 rlvr_ladder 的训练分布，但在带 tools 块的 prompt 下作答，
  与评测 prompt 分布一致）
- 扩充 GENERIC_Q 模板，让 get_current_time 真正被学到
- easy 直接答比例提到 23.5%，进一步稳定"何时不用工具"的判断

Usage: python scripts/build_tool_sft_v2.py [--n 12000]
"""
import argparse
import importlib.util
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tool_sft as v1  # noqa: E402  (v1 被 import 不会触发 main)

medium_mod = v1.medium_mod
easy_mod = v1.easy_mod
INSTRUCTION = v1.INSTRUCTION
TOOLS = v1.TOOLS
TOOL_JSON = v1.TOOL_JSON
SYSTEM = v1.SYSTEM
tool_msg = v1.tool_msg


def make_medium_direct(rng, cn):
    """medium 题直接作答（复刻 medium_sft 的 CoT 格式，无工具调用）。"""
    gen = rng.choice(medium_mod.GENS)
    q, steps, ans = gen(rng, cn)
    parsed = v1.parse_steps(steps)
    if parsed is None:
        return None
    lines = [f"{lbl}: {a} {op} {b} = {v1.calc(int(a), op, int(b))}"
             for lbl, a, op, b in parsed]
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q + INSTRUCTION},
            {"role": "assistant", "content": "\n".join(lines) + f"\n#### {ans}"}]
    return conv


# 参数化通用模板：时间查询 + 草稿箱记录/读取，覆盖 get_current_time 与 write/read
_TIME_ASKS = [
    "现在几点了？告诉我日期和时间。",
    "今天是几号？",
    "帮我看看现在的时间。",
    "现在是什么时间？请同时给出日期。",
    "能告诉我今天的日期吗？",
    "现在是上午还是下午？具体几点？",
    "顺便问一下，现在几点了？",
    "请问当前时间是多少？",
]
_NOTES = [
    "明天要交报告", "周五下午3点开会", "记得给妈妈回电话", "本月电费还没交",
    "下周四是外婆生日", "周三上午体检", "房租每月15号交", "明天限行尾号3和8",
    "冰箱里没有牛奶了", "周六约了朋友打球", "护照下个月到期", "信用卡还款日10号",
    "家里wifi密码是8个8", "下周二要出差三天", "儿子的家长会改到周五",
    "阳台的花该浇水了", "物业费每季度交一次", "图书馆的书后天到期",
    "晚上7点健身房有课", "快递明天上午到", "汽车该保养了", "猫粮快吃完了",
    "下周一是法定节假日", "水电表读数还没抄", "体检报告下周取",
]
_NOTE_ASKS = [
    "记录一下：{n}。然后告诉我现在的时间。",
    "把'{n}'记到草稿箱，然后读给我听确认。",
    "帮我在草稿箱存一条：{n}。",
    "记住这件事：{n}。存好后读出来确认。",
    "我在草稿箱里存了什么？帮我读出来。",
    "读一下草稿箱里的内容。",
]


def make_generic(rng, cn=False):
    if rng.random() < 0.45:
        q = rng.choice(_TIME_ASKS)
        conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
                {"role": "user", "content": q}]
        conv += tool_msg("get_current_time", {}, {"datetime": "2026-09-07 12:00:00"})
        conv.append({"role": "assistant", "content": "现在是 2026-09-07 12:00:00。"})
        return conv
    q = rng.choice(_NOTE_ASKS).format(n=rng.choice(_NOTES))
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": q}]
    if "存了什么" in q or "读一下" in q:
        conv += tool_msg("scratchpad_read", {}, {"content": ""})
        conv.append({"role": "assistant", "content": "草稿箱目前是空的，还没有记录任何内容。"})
        return conv
    note = q.split("：")[-1].split("。")[0].strip("'").strip()
    conv += tool_msg("scratchpad_write", {"content": note}, {"ok": True})
    if "读给" in q or "读出来" in q:
        conv += tool_msg("scratchpad_read", {}, {"content": note})
        conv.append({"role": "assistant", "content": f"已记录：{note}。读回确认无误。"})
    else:
        conv.append({"role": "assistant", "content": f"已记录：{note}。"})
    return conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--dst', default='dataset/tool_sft_v2.jsonl')
    parser.add_argument('--seed', type=int, default=778)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quota = [(v1.make_medium_trace, int(args.n * 0.24)),    # medium 带工具
             (make_medium_direct, int(args.n * 0.24)),      # medium 直接答
             (v1.make_easy_tool, int(args.n * 0.20)),       # easy 用工具
             (v1.make_easy_direct, int(args.n * 0.29)),     # easy 直接答
             (make_generic, int(args.n * 0.03))]            # 通用工具
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
            out.append({'conversations': conv})
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
