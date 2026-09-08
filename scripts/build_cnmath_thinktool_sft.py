"""中文数学 think+工具 冷启动 SFT v1：APE210K + Math23K
    -> user 题面 + assistant `<think>列式</think> + tool_call(calculate...)
       + tool 回包 + #### 终答`。

与 v1 工具 SFT（cnmath_tool_sft_v1）同源同配额同 seed，差别只在：
1) 首个 assistant 回合内容 = "<think>列式：x = <expr></think>"（计划进 think）；
2) easy 直答行（A）= think 后直接 ####（不调工具，但 system 仍带 tools）；
3) 纯 CoT 行取消（目标是"工具调用基础上的 think/answer"）。

对齐点：
- chat template 会把 assistant 内容里的 <think>...</think> 规范渲染成
  `<think>\n<推理>\n</think>\n\n<正文>`，正文后再拼 <tool_call>；
- RLVR/GRPO reward 以 `#### <number>` 为终答信号（有工具版按 train_tool_grpo
  的 calculate_reward 口径再定 think 加分，先保证 SFT 轨迹合法）。
"""
import argparse
import json
import random
import sys

from transformers import AutoTokenizer

ROOT = '/root/autodl-tmp/minimind'
sys.path.insert(0, ROOT + '/scripts')
import build_cnmath_tool_sft as T  # noqa: E402

cn = T.cn
CH_INSTRUCTION = T.CH_INSTRUCTION


def zh(expr):
    """ASCII 表达式 -> 中文习惯 × ÷。"""
    return expr.replace('*', '×').replace('/', '÷')


def think_content(rec):
    return '列式：x = ' + zh(rec['expr'])


def make_tool_call_turn(content, name, args):
    return {'role': 'assistant', 'content': content,
            'tool_calls': json.dumps([{'name': name, 'arguments': args}],
                                     ensure_ascii=False)}


def tool_result(name, args, result):
    return [make_tool_call_turn('', name, args),
            {'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)}]


def make_item(rec, typ, rng, with_plan):
    conv = [{'role': 'system', 'content': T.SYSTEM, 'tools': T.TOOL_JSON},
            {'role': 'user', 'content': rec['q'] + CH_INSTRUCTION}]
    tk = think_content(rec)
    if typ == 'A':
        # easy 直答：think（列式）后直接给终答，不调工具
        conv.append({'role': 'assistant',
                     'content': '<think>\n' + tk + '\n</think>\n'
                                 + T.final_text(rec['ans'], rng)})
        return conv, 'A'
    if with_plan:
        conv += tool_result('scratchpad_write',
                            {'content': '解题计划：x = ' + zh(rec['expr'])},
                            {'ok': True})
    first_call = {'expression': rec['expr']}
    if typ == 'B':
        conv.append(make_tool_call_turn(
            '<think>\n' + tk + '\n</think>', 'calculate', first_call))
        conv.append({'role': 'tool',
                     'content': json.dumps({'result': rec['ans']},
                                           ensure_ascii=False)})
        kind = 'B'
    else:  # C：按 AST 后序拆多步 calculate；首回合 think 与第一步同回合
        nodes = T.postorder(rec['ast'])

        def val(node):
            if node[0] == 'NUM':
                return T.leaf_text(node)
            return cn.fmt(T.eval_node(node))

        first = True
        for nd in nodes:
            expr = val(nd[2]) + nd[1] + val(nd[3])
            res = cn.fmt(T.eval_node(nd))
            if first:
                conv.append(make_tool_call_turn(
                    '<think>\n' + tk + '\n</think>', 'calculate',
                    {'expression': expr}))
                conv.append({'role': 'tool',
                             'content': json.dumps({'result': res},
                                                   ensure_ascii=False)})
                first = False
            else:
                conv += tool_result('calculate', {'expression': expr},
                                    {'result': res})
        kind = 'C'
    conv.append({'role': 'assistant', 'content': T.final_text(rec['ans'], rng)})
    return conv, kind + ('+D' if with_plan else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ape_cap', type=int, default=15000)
    ap.add_argument('--m23k_cap', type=int, default=6000)
    ap.add_argument('--ratio_a', type=float, default=0.15)
    ap.add_argument('--ratio_b', type=float, default=0.55)
    ap.add_argument('--ratio_c', type=float, default=0.30)
    ap.add_argument('--plan_ratio', type=float, default=0.10)
    ap.add_argument('--max_tok', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dst', default=f'{ROOT}/dataset/cnmath_thinktool_v1.jsonl')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/model')
    stats = {'bad_q': 0, 'bad_eq': 0, 'bad_ans': 0, 'dup': 0,
             'ape_read': 0, 'm23_read': 0, 'too_long': 0}
    seen = set()
    ape_rows = cn.read_ape(f'{ROOT}/dataset/ape210k/train.ape.json')
    m23_rows = cn.read_m23k(f'{ROOT}/dataset/math23k/math23k_train.json')
    stats['ape_read'] = len(ape_rows)
    stats['m23_read'] = len(m23_rows)
    rng.shuffle(ape_rows)
    rng.shuffle(m23_rows)
    ape = T.collect_valid(ape_rows, args.ape_cap, seen, stats, 'ape')
    m23 = T.collect_valid(m23_rows, args.m23k_cap, seen, stats, 'm23k')
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
    kind_count = {}
    out = []
    for rec in pool:
        if all(made[k] >= quota[k] for k in made):
            break
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
        conv, kind = make_item(rec, typ, rng,
                               typ != 'A' and rng.random() < args.plan_ratio)
        item = {'conversations': conv}
        if T.rendered_len(tokenizer, item['conversations']) > args.max_tok:
            stats['too_long'] += 1
            continue
        out.append(item)
        made[typ] += 1
        kind_count[kind] = kind_count.get(kind, 0) + 1

    rng.shuffle(out)
    with open(args.dst, 'w', encoding='utf-8') as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('quota', quota, 'made', made)
    print('kinds', kind_count)
    print('stats', stats)
    print('rows', len(out), '->', args.dst)
    for x in out[:2]:
        print('---')
        for m in x['conversations']:
            c = str(m.get('content', ''))[:150].replace('\n', ' | ')
            tc = str(m.get('tool_calls', ''))[:90]
            print(f"  [{m['role']}] content={c!r} tools={tc!r}")


if __name__ == '__main__':
    main()
