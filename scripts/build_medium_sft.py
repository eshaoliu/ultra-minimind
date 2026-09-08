"""中等题 SFT 数据：与 ladder_medium 同模板，但 assistant 输出"计划→分步执行"的标注式解答。

动机：rlvr_grpo2 面对 3~4 步应用题时退化成 "Compute 前两个数 +，再 -2" 的
捷径模式 —— 它没学过"先理解题意、规划运算链"。本数据用带实体标注的分步
解答教规划（每步引述题面对象），数字范围控制在已学会的运算区间内。

Usage: python scripts/build_medium_sft.py [--n 8000] [--dst dataset/medium_sft.jsonl]
"""
import argparse
import json
import random

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'
NAMES = ['Tom', 'Lisa', 'Jack', 'Mary', 'David', 'Anna', 'Sam', 'Nina', 'Leo', 'Lucy', 'Henry', 'Zoe']
CN_NAMES = ['小明', '小红', '小华', '小丽', '小刚', '小芳', '小军', '小燕']
DISTRACT_EN = ["The shop also has {d} notebooks in stock.",
               "There are also {d} students in the classroom.",
               "The box originally had {d} stickers inside.",
               "His neighbor has {d} chickens on the farm."]
DISTRACT_CN = ["商店里还有{d}本笔记本。", "教室里还有{d}名学生。", "盒子里原来还有{d}张贴纸。"]


def line(cn, label, a, op, b, r):
    sym = { '+': '+', '-': '-', 'x': 'x', '/': '/' }[op]
    if cn:
        return f"{label}：{a} {sym} {b} = {r}。"
    return f"{label}: {a} {sym} {b} = {r}."


def t_shop_change(rng, cn):
    n = rng.choice(CN_NAMES if cn else NAMES)
    p1, q1 = rng.randint(3, 19), rng.randint(2, 9)
    p2, q2 = rng.randint(4, 29), rng.randint(2, 8)
    cost = p1 * q1 + p2 * q2
    pay = cost + rng.randint(5, 200)
    if cn:
        q = f"{n}买了{q1}支单价{p1}元的钢笔和{q2}本单价{p2}元的书，付给收银员{pay}元，应找回多少元？"
        steps = [line(cn, f"钢笔总价", p1, 'x', q1, p1*q1),
                 line(cn, f"书总价", p2, 'x', q2, p2*q2),
                 line(cn, "一共花费", p1*q1, '+', p2*q2, cost),
                 line(cn, "应找回", pay, '-', cost, pay-cost)]
    else:
        q = f"{n} buys {q1} pens at ${p1} each and {q2} books at ${p2} each. He pays with ${pay}. How much change should he get?"
        steps = [line(cn, "Cost of pens", p1, 'x', q1, p1*q1),
                 line(cn, "Cost of books", p2, 'x', q2, p2*q2),
                 line(cn, "Total cost", p1*q1, '+', p2*q2, cost),
                 line(cn, "Change", pay, '-', cost, pay-cost)]
    return q, steps, pay - cost


def t_entity_total(rng, cn):
    n1, n2, n3 = rng.sample(CN_NAMES if cn else NAMES, 3)
    a = rng.randint(8, 60)
    b = rng.randint(3, 25)
    c = rng.randint(2, 20)
    b_has = 2 * a + b
    c_has = b_has - c
    if cn:
        q = f"{n1}有{a}颗弹珠，{n2}的弹珠比{n1}的2倍还多{b}颗，{n3}比{n2}少{c}颗。三人一共有多少颗弹珠？"
        steps = [line(cn, f"{n2}的弹珠", a, 'x', 2, 2*a),
                 line(cn, f"{n2}还要多{b}颗", 2*a, '+', b, b_has),
                 line(cn, f"{n3}的弹珠", b_has, '-', c, c_has),
                 line(cn, "三人总共", a + b_has, '+', c_has, a + b_has + c_has)]
    else:
        q = f"{n1} has {a} marbles. {n2} has twice as many as {n1} plus {b} more. {n3} has {c} fewer than {n2}. How many marbles do they have in total?"
        steps = [line(cn, f"{n2}'s marbles", a, 'x', 2, 2*a),
                 line(cn, f"{n2} plus {b} more", 2*a, '+', b, b_has),
                 line(cn, f"{n3}'s marbles", b_has, '-', c, c_has),
                 line(cn, "Total", a + b_has, '+', c_has, a + b_has + c_has)]
    return q, steps, a + b_has + c_has


def t_earn_spend(rng, cn):
    n = rng.choice(CN_NAMES if cn else NAMES)
    a, b = rng.randint(60, 300), rng.randint(60, 300)
    c = rng.randint(20, 150)
    d = rng.randint(10, 90)
    if cn:
        q = f"{n}周一赚了{a}元，周二赚了{b}元。他买食物花了{c}元，坐车花了{d}元，剩下的全部存起来。{n}存了多少钱？"
        steps = [line(cn, "两天共赚", a, '+', b, a+b),
                 line(cn, "一共花费", c, '+', d, c+d),
                 line(cn, "存下", a+b, '-', c+d, a+b-c-d)]
    else:
        q = f"{n} earned ${a} on Monday and ${b} on Tuesday. He spent ${c} on food and ${d} on transport. He saved the rest. How much did he save?"
        steps = [line(cn, "Total earned", a, '+', b, a+b),
                 line(cn, "Total spent", c, '+', d, c+d),
                 line(cn, "Saved", a+b, '-', c+d, a+b-c-d)]
    return q, steps, a + b - c - d


def t_machine(rng, cn):
    k = rng.randint(6, 25)
    h = rng.randint(3, 9)
    j = rng.randint(5, 20)
    h2 = rng.randint(2, 8)
    m = rng.randint(10, 60)
    if cn:
        q = f"甲机器每小时生产{k}个零件，运行了{h}小时；乙机器每小时生产{j}个零件，运行了{h2}小时。之后工人又手工做了{m}个。一共有多少个零件？"
        steps = [line(cn, "甲机器产量", k, 'x', h, k*h),
                 line(cn, "乙机器产量", j, 'x', h2, j*h2),
                 line(cn, "机器合计", k*h, '+', j*h2, k*h + j*h2),
                 line(cn, "加上手工", k*h + j*h2, '+', m, k*h + j*h2 + m)]
    else:
        q = f"Machine A produces {k} parts per hour and runs for {h} hours. Machine B produces {j} parts per hour and runs for {h2} hours. Then workers make {m} more by hand. How many parts are there in total?"
        steps = [line(cn, "Machine A output", k, 'x', h, k*h),
                 line(cn, "Machine B output", j, 'x', h2, j*h2),
                 line(cn, "Machine subtotal", k*h, '+', j*h2, k*h + j*h2),
                 line(cn, "Plus handmade", k*h + j*h2, '+', m, k*h + j*h2 + m)]
    return q, steps, k*h + j*h2 + m


def t_split_rest(rng, cn):
    n = rng.choice(CN_NAMES if cn else NAMES)
    bags = rng.randint(3, 9)
    per = rng.randint(6, 25)
    rest = bags * per
    a = rng.randint(3, 40)
    c = rng.randint(2, 30)
    total = rest + a + c
    if cn:
        q = f"有{total}块糖，先分给{n}{a}块，又分给小丽{c}块，剩下的平均装进{bags}个袋子。每个袋子装多少块？"
        steps = [line(cn, "分出去", a, '+', c, a+c),
                 line(cn, "剩下", total, '-', a+c, rest),
                 line(cn, "每袋", rest, '/', bags, per)]
    else:
        q = f"There are {total} candies. {n} gets {a} candies and Lisa gets {c} candies. The rest are packed equally into {bags} bags. How many candies are in each bag?"
        steps = [line(cn, "Given away", a, '+', c, a+c),
                 line(cn, "Remaining", total, '-', a+c, rest),
                 line(cn, "Per bag", rest, '/', bags, per)]
    return q, steps, per


def t_compare_spend(rng, cn):
    n1, n2 = rng.sample(CN_NAMES if cn else NAMES, 2)
    p, qn = rng.randint(3, 15), rng.randint(3, 12)
    q1, q2 = rng.randint(2, 9), rng.randint(2, 9)
    s1, s2 = p * q1, qn * q2
    if s1 == s2:
        q1 += 1
        s1 = p * q1
    if cn:
        q = f"{n1}买了{q1}个单价{p}元的苹果，{n2}买了{q2}个单价{qn}元的橙子。谁花得多？多多少钱？"
        steps = [line(cn, f"{n1}花费", p, 'x', q1, s1),
                 line(cn, f"{n2}花费", qn, 'x', q2, s2),
                 line(cn, "相差", max(s1, s2), '-', min(s1, s2), abs(s1-s2))]
    else:
        q = f"{n1} buys {q1} apples at ${p} each. {n2} buys {q2} oranges at ${qn} each. Who spends more, and by how much?"
        steps = [line(cn, f"{n1}'s spending", p, 'x', q1, s1),
                 line(cn, f"{n2}'s spending", qn, 'x', q2, s2),
                 line(cn, "Difference", max(s1, s2), '-', min(s1, s2), abs(s1-s2))]
    return q, steps, abs(s1 - s2)


def t_age(rng, cn):
    n1, n2 = rng.sample(CN_NAMES if cn else NAMES, 2)
    s = rng.randint(4, 12)
    k = rng.randint(3, 5)
    y = rng.randint(2, 10)
    if cn:
        q = f"{n2}今年{s}岁，{n1}的年龄是{n2}的{k}倍。再过{y}年，两人的年龄和是多少岁？"
        steps = [line(cn, f"{n1}今年", s, 'x', k, k*s),
                 line(cn, "今年年龄和", s, '+', k*s, s + k*s),
                 line(cn, f"再过{y}年两人共长", 2, 'x', y, 2*y),
                 line(cn, "那时年龄和", s + k*s, '+', 2*y, s + k*s + 2*y)]
    else:
        q = f"{n2} is {s} years old. {n1} is {k} times as old as {n2}. In {y} years, what will the sum of their ages be?"
        steps = [line(cn, f"{n1}'s age now", s, 'x', k, k*s),
                 line(cn, "Sum of ages now", s, '+', k*s, s + k*s),
                 line(cn, f"Both gain in {y} years", 2, 'x', y, 2*y),
                 line(cn, "Sum then", s + k*s, '+', 2*y, s + k*s + 2*y)]
    return q, steps, s + k*s + 2*y


def t_pages_left(rng, cn):
    n = rng.choice(CN_NAMES if cn else NAMES)
    a = rng.randint(15, 80)
    total = a + 2 * a + rng.randint(10, 120)
    if cn:
        q = f"一本书共{total}页。{n}第一天读了{a}页，第二天读的是第一天的2倍。还剩多少页没读？"
        steps = [line(cn, "第二天读", a, 'x', 2, 2*a),
                 line(cn, "两天共读", a, '+', 2*a, 3*a),
                 line(cn, "还剩", total, '-', 3*a, total-3*a)]
    else:
        q = f"A book has {total} pages. {n} reads {a} pages on day one, and twice as many on day two. How many pages are left unread?"
        steps = [line(cn, "Day two reading", a, 'x', 2, 2*a),
                 line(cn, "Total read", a, '+', 2*a, 3*a),
                 line(cn, "Pages left", total, '-', 3*a, total-3*a)]
    return q, steps, total - 3*a


GENS = [t_shop_change, t_entity_total, t_earn_spend, t_machine, t_split_rest, t_compare_spend, t_age, t_pages_left]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--dst', default='dataset/medium_sft.jsonl')
    parser.add_argument('--seed', type=int, default=321)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    out = []
    while len(out) < args.n:
        gen = rng.choice(GENS)
        cn = rng.random() < 0.5
        q, steps, ans = gen(rng, cn)
        if q in seen:
            continue
        seen.add(q)
        if rng.random() < 0.5:
            tpl = rng.choice(DISTRACT_CN if cn else DISTRACT_EN)
            q += ' ' + tpl.format(d=rng.randint(10, 400))
        ans_text = "\n".join(steps) + f"\n#### {ans}\n"
        out.append({'conversations': [
            {'role': 'user', 'content': q + INSTRUCTION},
            {'role': 'assistant', 'content': ans_text},
        ]})

    rng.shuffle(out)
    with open(args.dst, 'w') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'写出 {len(out)} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
