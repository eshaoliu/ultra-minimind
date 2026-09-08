"""中文数学工具 SFT v1：APE210K + Math23K -> 分层工具轨迹 + 纯 CoT + 评测子集。

与 eval_tool_math.py / build_tool_sft.py 的 schema 完全一致：
system 带 tools；assistant 用 tool_calls(JSON 字符串)；工具结果 role=tool；
终答统一 `#### <number>`。

轨迹形态（配额见 --ratio，默认 A15/B55/C20 + 10% 叠草稿箱 + CoT 15%）：
  A 直接答（一步题，不调工具，system 仍带 tools）
  B 工具-单列式：calculate("<完整表达式>") 一次调用
  C 工具-分步：按 AST 后序拆成多步 calculate，中间结果复用
  D 在 B/C 上叠加 scratchpad_write 计划（重叠升级，不单独占配额）
  CoT 纯逐步展开（无 tools system，给纯 CoT 评测口径用）

Usage:
  python scripts/build_cnmath_tool_sft.py \
      --dst dataset/cnmath_tool_sft_v1.jsonl \
      --eval_out dataset/cnmath_eval300.jsonl
"""
import argparse
import importlib.util
import json
import random
import re
import sys

from transformers import AutoTokenizer

ROOT = '/root/autodl-tmp/minimind'


def _load(mod_path, name):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cn = _load(f'{ROOT}/scripts/build_cnmath_sft.py', 'cn')

CH_INSTRUCTION = "\n请一步步计算，最后在 #### 后给出最终数字答案（例如：#### 42）。"
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
TOOL_JSON = json.dumps(TOOLS, ensure_ascii=False)

PREC = {'+': 1, '-': 1, '*': 2, '/': 2}


def rendered_len(tokenizer, conv):
    """与 dataset/lm_dataset.SFTDataset.create_chat_prompt 同口径渲染后 token 数。"""
    messages = [dict(m) for m in conv]
    tools = None
    for m in messages:
        if m.get('role') == 'system' and m.get('tools'):
            tools = json.loads(m['tools']) if isinstance(m['tools'], str) else m['tools']
        if m.get('tool_calls') and isinstance(m['tool_calls'], str):
            m['tool_calls'] = json.loads(m['tool_calls'])
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False, tools=tools)
    return len(tokenizer(prompt).input_ids)


def tool_msg(name, args, result):
    call = {"role": "assistant", "content": "",
            "tool_calls": json.dumps([{"name": name, "arguments": args}],
                                     ensure_ascii=False)}
    return [call, {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}]


def sig(q):
    return re.sub(r'\s+', '', q)


def gold_num(ans):
    """整数/小数/分数 -> float；百分数终答 v1 跳过(返回 None)。"""
    a = (ans or '').strip().replace('％', '%')
    if a.endswith('%'):
        return None
    m = re.fullmatch(r'\(?\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*\)?', a)
    if m:
        b = float(m.group(2))
        return float(m.group(1)) / b if b else None
    m = re.fullmatch(r'-?\d+(?:\.\d+)?', a)
    return float(a) if m else None


def eval_node(node):
    """只支持 + - * /；含 ^ 的方程在过滤时丢弃。"""
    if node[0] == 'NUM':
        return node[1]
    _, op, left, right = node
    lv, rv = eval_node(left), eval_node(right)
    return {'+': lv + rv, '-': lv - rv, '*': lv * rv, '/': lv / rv}[op]


def ops_count(node):
    if node[0] == 'NUM':
        return 0
    return 1 + ops_count(node[2]) + ops_count(node[3])


def leaf_text(node):
    """叶子的规范化文本：百分数字面量转小数，其余保留原样。"""
    if node[2].endswith('%'):
        return cn.fmt(node[1])
    return node[2]


def infix(node):
    """最小括号中缀（只 + - * /，叶子百分号已转小数）。"""
    if node[0] == 'NUM':
        return leaf_text(node)
    _, op, left, right = node
    l, r = infix(left), infix(right)
    p = PREC[op]
    if left[0] != 'NUM' and PREC[left[1]] < p:
        l = f'({l})'
    if right[0] != 'NUM' and PREC[right[1]] <= p:
        r = f'({r})'
    return f'{l}{op}{r}'


def postorder(node):
    """返回内部节点后序列表（先算左、右子树，再算本节点）。"""
    if node[0] == 'NUM':
        return []
    return postorder(node[2]) + postorder(node[3]) + [node]


def has_pow(node):
    if node is None:
        return False
    if node[0] == 'NUM':
        return False
    if node[1] == '^':
        return True
    return has_pow(node[2]) or has_pow(node[3])


def tree_ok(node):
    """AST 完整性校验（防悬空运算符导致 None 子树）。"""
    if node is None:
        return False
    if node[0] == 'NUM':
        return True
    if len(node) < 4:
        return False
    return tree_ok(node[2]) and tree_ok(node[3])


def parse_row(row, stats):
    q = (row.get('original_text') or '').strip()
    eq = (row.get('equation') or '').strip()
    if not q or not eq or len(q) > 240:
        stats['bad_q'] += 1
        return None
    if '=' in eq:
        eq = eq.split('=', 1)[1]
    toks = cn.tokenize(eq)
    if not toks:
        stats['bad_eq'] += 1
        return None
    ast = cn.parse(toks)
    if ast is None or not tree_ok(ast) or has_pow(ast):
        stats['bad_eq'] += 1
        return None
    try:
        v = eval_node(ast)
    except Exception:
        stats['bad_eq'] += 1
        return None
    gv = gold_num(row.get('ans') or '')
    if gv is None:
        stats['bad_ans'] += 1
        return None
    if abs(v - gv) > 1e-4 * max(1.0, abs(gv)):
        stats['bad_ans'] += 1
        return None
    return {'q': q, 'ast': ast, 'expr': infix(ast),
            'eq_raw': eq, 'gv': gv, 'ans': cn.fmt(v),
            'ops': ops_count(ast), 'len': len(infix(ast))}


def collect_valid(rows, cap, seen, stats, src):
    """按 seed 打乱后取 cap 条通过校验的题（含分数中间量等）。"""
    out = []
    for row in rows:
        if len(out) >= cap:
            break
        rec = parse_row(row, stats)
        if rec is None:
            continue
        s = sig(rec['q'])
        if s in seen:
            stats['dup'] += 1
            continue
        seen.add(s)
        rec['src'] = src
        out.append(rec)
    return out


def final_text(ans, rng):
    if rng.random() < 0.5:
        return f"所以：x = {ans}\n#### {ans}"
    return f"计算完成，答案是：{ans}。\n#### {ans}"


def make_direct(rec, rng):
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": rec['q'] + CH_INSTRUCTION},
            {"role": "assistant", "content": final_text(rec['ans'], rng)}]
    return {'conversations': conv, 'kind': 'A'}


def make_single_tool(rec, rng, with_plan):
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": rec['q'] + CH_INSTRUCTION}]
    if with_plan:
        plan = f"解题计划：x = {rec['expr']}"
        conv += tool_msg('scratchpad_write', {'content': plan}, {'ok': True})
    conv += tool_msg('calculate', {'expression': rec['expr']},
                     {'result': rec['ans']})
    conv.append({'role': 'assistant', 'content': final_text(rec['ans'], rng)})
    return {'conversations': conv, 'kind': 'B' + ('+D' if with_plan else '')}


def make_step_tool(rec, rng, with_plan):
    conv = [{"role": "system", "content": SYSTEM, "tools": TOOL_JSON},
            {"role": "user", "content": rec['q'] + CH_INSTRUCTION}]
    if with_plan:
        conv += tool_msg('scratchpad_write',
                         {'content': f"解题计划：x = {rec['expr']}"}, {'ok': True})
    nodes = postorder(rec['ast'])

    def val(node):
        if node[0] == 'NUM':
            return leaf_text(node)
        else:
            return cn.fmt(eval_node(node))

    for nd in nodes:
        lt = val(nd[2])
        rt = val(nd[3])
        expr = f'{lt}{nd[1]}{rt}'
        res = cn.fmt(eval_node(nd))
        conv += tool_msg('calculate', {'expression': expr}, {'result': res})
    conv.append({'role': 'assistant', 'content': final_text(rec['ans'], rng)})
    return {'conversations': conv, 'kind': 'C' + ('+D' if with_plan else '')}


def make_cot(rec):
    content = cn.build_trace(rec['eq_raw'])
    if content is None:
        return None
    v, lines = content
    body = '\n'.join(lines)
    conv = [{'role': 'user', 'content': rec['q'] + CH_INSTRUCTION}]
    conv.append({'role': 'assistant',
                 'content': f"{body}\n所以：x = {rec['ans']}\n#### {rec['ans']}"})
    return {'conversations': conv, 'kind': 'CoT'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ape_cap', type=int, default=15000)
    ap.add_argument('--m23k_cap', type=int, default=6000)
    ap.add_argument('--cot_n', type=int, default=3150)
    ap.add_argument('--ratio_a', type=float, default=0.15)
    ap.add_argument('--ratio_b', type=float, default=0.55)
    ap.add_argument('--ratio_c', type=float, default=0.30)
    ap.add_argument('--plan_ratio', type=float, default=0.10, help='B/C 叠草稿箱比例')
    ap.add_argument('--max_tok', type=int, default=1024,
                    help='渲染后 token 上限，超长毒株(循环/递推题)直接丢弃')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dst', default=f'{ROOT}/dataset/cnmath_tool_sft_v1.jsonl')
    ap.add_argument('--eval_out', default=f'{ROOT}/dataset/cnmath_eval300.jsonl')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/model')

    stats = {'bad_q': 0, 'bad_eq': 0, 'bad_ans': 0, 'dup': 0, 'ape_read': 0,
             'm23_read': 0, 'too_long': 0}
    seen = set()
    ape_rows = cn.read_ape(f'{ROOT}/dataset/ape210k/train.ape.json')
    m23_rows = cn.read_m23k(f'{ROOT}/dataset/math23k/math23k_train.json')
    rng.shuffle(ape_rows)
    rng.shuffle(m23_rows)
    ape = collect_valid(ape_rows, args.ape_cap, seen, stats, 'ape')
    m23 = collect_valid(m23_rows, args.m23k_cap, seen, stats, 'm23k')
    pool = ape + m23
    rng.shuffle(pool)
    stats['ape_valid'] = len(ape)
    stats['m23_valid'] = len(m23)

    n_main = int(args.ape_cap + args.m23k_cap)
    q_a = int(n_main * args.ratio_a)
    q_c = int(n_main * args.ratio_c)
    q_b = n_main - q_a - q_c
    quota = {'A': q_a, 'B': q_b, 'C': q_c}
    made = {'A': 0, 'B': 0, 'C': 0}
    out = []
    used = set()

    for rec in pool:
        if all(made[k] >= quota[k] for k in made):
            break
        s = sig(rec['q'])
        easy = rec['ops'] <= 1 and float(rec['gv']).is_integer()
        complex_ = rec['ops'] >= 5 or rec['len'] > 70
        typ = None
        if made['A'] < quota['A'] and easy:
            typ = 'A'
        elif made['C'] < quota['C'] and complex_:
            typ = 'C'
        elif made['B'] < quota['B']:
            typ = 'B'
        elif made['C'] < quota['C']:
            typ = 'C'
        elif made['A'] < quota['A']:
            typ = 'A'
        if typ is None:
            continue
        if typ == 'A':
            item = make_direct(rec, rng)
        elif typ == 'C':
            item = make_step_tool(rec, rng, rng.random() < args.plan_ratio)
        else:
            item = make_single_tool(rec, rng, rng.random() < args.plan_ratio)
        if rendered_len(tokenizer, item['conversations']) > args.max_tok:
            stats['too_long'] += 1
            continue
        out.append(item)
        made[typ] += 1
        used.add(s)

    # CoT 变体：优先没用过的题；不够就从主轨迹里补
    extra_pool = [r for r in pool if sig(r['q']) not in used]
    if len(extra_pool) < args.cot_n:
        extra_pool += [r for r in pool if sig(r['q']) in used][:args.cot_n - len(extra_pool)]
    rng.shuffle(extra_pool)
    cot_made = 0
    for rec in extra_pool:
        if cot_made >= args.cot_n:
            break
        item = make_cot(rec)
        if item is None:
            continue
        if rendered_len(tokenizer, item['conversations']) > args.max_tok:
            stats['too_long'] += 1
            continue
        out.append(item)
        cot_made += 1

    print('quota', quota, 'made', made, 'cot', cot_made)
    print('stats', stats)

    rng.shuffle(out)
    with open(args.dst, 'w', encoding='utf-8') as f:
        for x in out:
            f.write(json.dumps({'conversations': x['conversations']}, ensure_ascii=False) + '\n')
    print('rows', len(out), '->', args.dst)
    for x in out[:4]:
        print('---')
        for m in x['conversations']:
            c = str(m.get('content', ''))[:110].replace('\n', ' | ')
            tc = m.get('tool_calls', '')
            print(f"  [{m['role']}] content={c!r} tools={str(tc)[:80]!r}")

    # ---- 固定评测子集 300：ape 100 + m23k 100 + cmath 100 ----
    def norm_line(q, a, src):
        return {'question': q.strip(), 'answer': a.strip(), 'source': src}

    eval_rows = []
    ape_test = cn.read_ape(f'{ROOT}/dataset/ape210k/test.ape.json')
    m23_test = cn.read_m23k(f'{ROOT}/dataset/math23k/math23k_test.json')
    for r in rng.sample([r for r in ape_test if (r.get('original_text') or '').strip()
                         and gold_num(r.get('ans')) is not None], 100):
        eval_rows.append(norm_line(r['original_text'], r['ans'], 'ape'))
    for r in rng.sample([r for r in m23_test if (r.get('original_text') or '').strip()
                         and gold_num(r.get('ans')) is not None], 100):
        eval_rows.append(norm_line(r['original_text'], r['ans'], 'm23k'))
    cmath = [json.loads(l) for l in open(f'{ROOT}/dataset/cmath/cmath_dev.jsonl', encoding='utf-8')]
    for r in rng.sample([r for r in cmath if (r.get('input') or '').strip()
                         and gold_num(r.get('golden')) is not None], 100):
        eval_rows.append(norm_line(r['input'], r['golden'], 'cmath'))
    with open(args.eval_out, 'w', encoding='utf-8') as f:
        for x in eval_rows:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('eval rows', len(eval_rows), '->', args.eval_out)


if __name__ == '__main__':
    main()
