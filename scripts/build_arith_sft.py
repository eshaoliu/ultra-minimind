"""构造算术能力 SFT 数据：显式位运算过程的加减乘除 + 包装应用题。

背景：warmup 模型的 CoT 结构正确但算术全错（317-76 算成 48），
GSM8K gold CoT 是"心算跳步"风格，64M 模型学不会。本数据用
位数对齐（carry/borrow/partial product）的显式过程教算术。

Usage: python scripts/build_arith_sft.py [--n 20000] [--dst dataset/arith_sft.jsonl]
"""
import argparse
import json
import random

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'
NAMES = ['Tom', 'Lisa', 'Jack', 'Mary', 'David', 'Anna', 'Sam', 'Nina', 'Leo', 'Lucy']
CN_NAMES = ['小明', '小红', '小华', '小丽', '小刚', '小芳']


def digits(n):
    return [int(c) for c in str(n)]


def add_steps(a, b, cn=False):
    """位数对齐加法过程。返回 (过程文本, 结果)。"""
    da, db = digits(a), digits(b)
    L = max(len(da), len(db))
    da = [0] * (L - len(da)) + da
    db = [0] * (L - len(db)) + db
    labels_en = ['Ones', 'Tens', 'Hundreds', 'Thousands']
    labels_cn = ['个位', '十位', '百位', '千位']
    labels = labels_cn if cn else labels_en
    lines, carry, parts = [], 0, []
    for i in range(L - 1, -1, -1):
        s = da[i] + db[i] + carry
        d, new_carry = s % 10, s // 10
        prev = f" + 进位{carry}" if (carry and cn) else (f" + carry {carry}" if carry else "")
        if s >= 10:
            txt = f"{s}，写{d}进{new_carry}" if cn else f"{s}, write down {d} and carry {new_carry}"
        else:
            txt = f"{s}，写{d}" if cn else f"{s}, write down {d}"
        lines.append(f"{labels[L-1-i]}：{da[i]} + {db[i]}{prev} = {txt}" if cn
                     else f"{labels[i]}: {da[i]} + {db[i]}{prev} = {txt}")
        parts.append(str(d))
        carry = new_carry
    if carry:
        lines.append(f"进位{carry}到更高位" if cn else f"Carry {carry} to the next digit")
        parts.append(str(carry))
    return lines, int(''.join(reversed(parts)))


def sub_steps(a, b, cn=False):
    """位数对齐减法（a>b）。返回 (过程文本, 结果)。"""
    assert a >= b
    da, db = digits(a), digits(b)
    L = len(da)
    db = [0] * (L - len(db)) + db
    labels_en = ['Ones', 'Tens', 'Hundreds', 'Thousands']
    labels_cn = ['个位', '十位', '百位', '千位']
    labels = labels_cn if cn else labels_en
    lines, res = [], []
    for i in range(L - 1, -1, -1):
        x = da[i]
        if x < db[i]:
            x += 10
            j = i - 1
            while da[j] == 0:
                da[j] = 9
                j -= 1
            da[j] -= 1
            if cn:
                lines.append(f"{labels[L-1-i]}：{x - 10}不够减{db[i]}，借1当10：{x} - {db[i]} = {x - db[i]}")
            else:
                lines.append(f"{labels[L-1-i]}: {x - 10} < {db[i]}, borrow 1: {x} - {db[i]} = {x - db[i]}")
        else:
            if cn:
                lines.append(f"{labels[L-1-i]}：{x} - {db[i]} = {x - db[i]}")
            else:
                lines.append(f"{labels[L-1-i]}: {x} - {db[i]} = {x - db[i]}")
        res.append(str(x - db[i]))
    return lines, int(''.join(reversed(res)))


def mul_steps(a, b, cn=False):
    """一位乘数逐位展开；整十乘数用移位。保证每个子步骤都是可学的一位数运算。"""
    if b % 10 == 0 and b > 10:
        base = b // 10
        p = a * base
        if cn:
            return [f"{a} x {b} = {a} x {base} x 10 = {p} x 10 = {a * b}"], a * b
        return [f"{a} x {b} = {a} x {base} x 10 = {p} x 10 = {a * b}"], a * b
    assert b < 10
    da = digits(a)
    lines, carry, out = [], 0, []
    labels_en = ['Ones', 'Tens', 'Hundreds', 'Thousands']
    labels_cn = ['个位', '十位', '百位', '千位']
    labels = labels_cn if cn else labels_en
    for i in range(len(da) - 1, -1, -1):
        s = da[i] * b + carry
        d, new_carry = s % 10, s // 10
        pos = len(da) - 1 - i
        prev_cn = f" + 进位{carry}" if carry else ""
        prev_en = f" + carry {carry}" if carry else ""
        if s >= 10:
            txt_cn, txt_en = f"{s}，写{d}进{new_carry}", f"{s}, write down {d} and carry {new_carry}"
        else:
            txt_cn, txt_en = f"{s}，写{d}", f"{s}, write down {d}"
        if cn:
            lines.append(f"{labels[pos]}：{da[i]} x {b}{prev_cn} = {txt_cn}")
        else:
            lines.append(f"{labels[pos]}: {da[i]} x {b}{prev_en} = {txt_en}")
        out.append(str(d))
        carry = new_carry
    if carry:
        lines.append(f"最后进位{carry}" if cn else f"Final carry: {carry}")
        out.append(str(carry))
    return lines, int(''.join(reversed(out)))


def div_steps(a, b, cn=False):
    """整除：试商 + 乘验算。"""
    q = a // b
    lines = [f"Try {q}: {q} x {b} = {q*b}, which equals {a}. So {a} / {b} = {q}."
             if not cn else f"试商{q}：{q} x {b} = {q*b}，正好等于{a}。所以 {a} / {b} = {q}。"]
    return lines, q


def fmt_compute(a, op, b, cn=False):
    lines, res = {'+': add_steps, '-': sub_steps, 'x': mul_steps, '/': div_steps}[op](a, b, cn)
    sym = f"{a} {op} {b}"
    head = f"计算 {sym}：" if cn else f"Compute {sym}:"
    tail = f"所以 {sym} = {res}。" if cn else f"So {sym} = {res}."
    return head + "\n" + "\n".join(lines) + "\n" + tail, res


def drill(rng):
    cn = rng.random() < 0.5
    op = rng.choice(['+', '-', 'x', '/'])
    if op == '+':
        a, b = rng.randint(11, 999), rng.randint(11, 999)
    elif op == '-':
        a, b = rng.randint(50, 999), rng.randint(11, 499)
        if b > a:
            a, b = b, a
    elif op == 'x':
        if rng.random() < 0.25:
            a, b = rng.randint(3, 99), rng.choice([20, 30, 40, 50])
        else:
            a, b = rng.randint(3, 999), rng.randint(2, 9)
    else:
        b = rng.randint(2, 12)
        q = rng.randint(3, 60)
        a = b * q
    q_text = f"{a} {op} {b} = ?" + ("" if not cn else "")
    ans_text, res = fmt_compute(a, op, b, cn)
    ans_text += f"\n#### {res}"
    return q_text, ans_text


EN_WORD = {
    '+': ("{n} has {a} apples. He buys {b} more. How many apples does {n} have now?", "{a} + {b}"),
    '-': ("{n} had {a} candies. He gave {b} candies to his friends. How many candies does {n} have left?", "{a} - {b}"),
    'x': ("A box contains {a} pencils. How many pencils are there in {b} boxes?", "{a} x {b}"),
    '/': ("There are {a} students. Each team has {b} students. How many teams can be formed?", "{a} / {b}"),
}
CN_WORD = {
    '+': ("{n}有{a}个苹果，又买了{b}个。{n}现在有多少个苹果？", "{a} + {b}"),
    '-': ("{n}有{a}颗糖，送给朋友{b}颗。{n}还剩多少颗糖？", "{a} - {b}"),
    'x': ("每个盒子装{a}支铅笔，{b}个盒子一共装多少支铅笔？", "{a} x {b}"),
    '/': ("有{a}名学生，每{b}人分成一队，可以分成多少队？", "{a} / {b}"),
}


def word_one(rng):
    cn = rng.random() < 0.5
    op = rng.choice(['+', '-', 'x', '/'])
    name = rng.choice(CN_NAMES if cn else NAMES)
    if op == '+':
        a, b = rng.randint(11, 999), rng.randint(11, 999)
    elif op == '-':
        a, b = rng.randint(50, 999), rng.randint(11, 499)
        if b > a:
            a, b = b, a
    elif op == 'x':
        a, b = rng.randint(3, 99), rng.randint(2, 9)
    else:
        b = rng.randint(2, 12)
        q = rng.randint(3, 40)
        a = b * q
    tpl, expr = (CN_WORD if cn else EN_WORD)[op]
    q_text = tpl.format(n=name, a=a, b=b)
    calc, res = fmt_compute(a, op, b, cn)
    lead_cn = f"需要计算 {a} {op} {b}。"
    lead_en = f"We need to compute {a} {op} {b}."
    ans_text = (lead_cn if cn else lead_en) + "\n" + calc
    ans_text += f"\n#### {res}\n"
    return q_text, ans_text


def word_two(rng):
    cn = rng.random() < 0.5
    name = rng.choice(CN_NAMES if cn else NAMES)
    kind = rng.choice(['add_sub', 'mul_sub', 'sub_sub', 'add_add'])
    if kind == 'add_sub':
        a, b = rng.randint(20, 400), rng.randint(10, 300)
        s = a + b
        c = rng.randint(5, s - 5)
        if cn:
            q_text = f"{name}有{a}张邮票，又买了{b}张，然后送给朋友{c}张。{name}现在有多少张邮票？"
        else:
            q_text = f"{name} has {a} stamps. He buys {b} more, then gives {c} stamps to his friend. How many stamps does {name} have now?"
        c1, r1 = fmt_compute(a, '+', b, cn)
        c2, r2 = fmt_compute(r1, '-', c, cn)
    elif kind == 'mul_sub':
        a, b = rng.randint(4, 15), rng.randint(3, 9)
        c = rng.randint(1, a * b - 1)
        if cn:
            q_text = f"商店每盒装{a}个鸡蛋，{name}买了{b}盒，用掉{c}个做蛋糕。还剩多少个鸡蛋？"
        else:
            q_text = f"A shop packs {a} eggs per box. {name} buys {b} boxes and uses {c} eggs for a cake. How many eggs are left?"
        c1, r1 = fmt_compute(a, 'x', b, cn)
        c2, r2 = fmt_compute(r1, '-', c, cn)
    elif kind == 'sub_sub':
        a = rng.randint(300, 999)
        b = rng.randint(30, 200)
        r1 = a - b
        c = rng.randint(10, min(150, r1 - 1))
        if cn:
            q_text = f"{name}有{a}元，买玩具花了{b}元，买书花了{c}元。{name}还剩多少元？"
        else:
            q_text = f"{name} had {a} dollars. He spent {b} dollars on a toy and {c} dollars on a book. How much money does {name} have left?"
        c1, r1 = fmt_compute(a, '-', b, cn)
        c2, r2 = fmt_compute(r1, '-', c, cn)
    else:
        a, b, c = rng.randint(15, 300), rng.randint(15, 300), rng.randint(15, 300)
        if cn:
            q_text = f"{name}周一读了{a}页，周二读了{b}页，周三读了{c}页。{name}一共读了多少页？"
        else:
            q_text = f"{name} read {a} pages on Monday, {b} pages on Tuesday, and {c} pages on Wednesday. How many pages did {name} read in total?"
        c1, r1 = fmt_compute(a, '+', b, cn)
        c2, r2 = fmt_compute(r1, '+', c, cn)
    ans_text = c1 + "\n" + c2
    ans_text += f"\n#### {r2}\n"
    return q_text, ans_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--dst', default='dataset/arith_sft.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    out = []
    # 配比：口算题 55%，单步应用题 25%，两步应用题 20%
    quota = [(drill, int(args.n * 0.55)), (word_one, int(args.n * 0.25)), (word_two, int(args.n * 0.20))]
    for gen, n in quota:
        made = 0
        while made < n:
            q, a = gen(rng)
            if q in seen:
                continue
            seen.add(q)
            out.append({'conversations': [
                {'role': 'user', 'content': q + INSTRUCTION},
                {'role': 'assistant', 'content': a},
            ]})
            made += 1
    rng.shuffle(out)
    with open(args.dst, 'w') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'写出 {len(out)} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
