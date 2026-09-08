"""构造降难度 RLVR 数据：一步/两步算术应用题（答案程序验证），抬 GRPO 起点正确率。

GSM8K 随机题上 warmup 模型 temp0.8 正确率仅 ~0.24%，96% zero-variance 组。
混合本脚本生成的一步/两步题（预期正确率 20%~80%，GRPO 信号区间），
让组内 advantage 非零的比例大幅提升。

Usage: python scripts/build_easy_rlvr.py [--n1 3000] [--n2 4500] [--dst dataset/easy_math.jsonl]
"""
import argparse
import json
import random

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'

NAMES = ['Tom', 'Lisa', 'Jack', 'Mary', 'David', 'Anna', 'Sam', 'Nina', 'Leo', 'Lucy']
CN_NAMES = ['小明', '小红', '小华', '小丽', '小刚', '小芳']
CN = {
    'add': "{n}有{a}个苹果，又买了{b}个。{n}现在有多少个苹果？",
    'sub': "{n}有{a}颗糖，送给朋友{b}颗。{n}还剩多少颗糖？",
    'mul': "每个盒子装{a}支铅笔，{b}个盒子一共装多少支铅笔？",
    'div': "有{a}名学生，每{b}人分成一队，可以分成多少队？",
    'add_sub': "{n}有{a}张邮票，又买了{b}张，然后送给朋友{c}张。{n}现在有多少张邮票？",
    'mul_sub': "商店每盒装{a}个鸡蛋，{n}买了{b}盒，用掉{c}个做蛋糕。还剩多少个鸡蛋？",
    'mul_div': "{n}有{b}袋弹珠，每袋{cp}个，平均装进{c}个盒子，每个盒子装多少个？",
    'sub_sub': "{n}有{a}元，买玩具花了{b}元，买书花了{c}元。{n}还剩多少元？",
    'add_add': "{n}周一读了{a}页，周二读了{b}页，周三读了{c}页。{n}一共读了多少页？",
}
EN = {
    'add': "{n} has {a} apples. He buys {b} more. How many apples does {n} have now?",
    'sub': "{n} had {a} candies. He gave {b} candies to his friends. How many candies does {n} have left?",
    'mul': "A box contains {a} pencils. How many pencils are there in {b} boxes?",
    'div': "There are {a} students. Each team has {b} students. How many teams can be formed?",
    'add_sub': "{n} has {a} stamps. He buys {b} more, then gives {c} stamps to his friend. How many stamps does {n} have now?",
    'mul_sub': "A shop packs {a} eggs per box. {n} buys {b} boxes and uses {c} eggs for a cake. How many eggs are left?",
    'mul_div': "{n} has {b} bags of {cp} marbles. He splits them equally into {c} boxes. How many marbles are in each box?",
    'sub_sub': "{n} had {a} dollars. He spent {b} dollars on a toy and {c} dollars on a book. How much money does {n} have left?",
    'add_add': "{n} read {a} pages on Monday, {b} pages on Tuesday, and {c} pages on Wednesday. How many pages did {n} read in total?",
}


def _render(lang, kind, **kw):
    tpl = (CN if lang == 'cn' else EN)[kind]
    return tpl.format(**kw)


def one_step(rng):
    kind = rng.choice(['add', 'sub', 'mul', 'div'])
    lang = rng.choice(['en', 'cn'])
    name = rng.choice(CN_NAMES if lang == 'cn' else NAMES)
    if kind == 'add':
        a, b = rng.randint(11, 499), rng.randint(11, 499)
        kw = dict(n=name, a=a, b=b)
        ans = a + b
    elif kind == 'sub':
        a, b = rng.randint(50, 999), rng.randint(11, 499)
        if b > a:
            a, b = b + rng.randint(1, 100), a
        kw = dict(n=name, a=a, b=b)
        ans = a - b
    elif kind == 'mul':
        a, b = rng.randint(3, 19), rng.randint(3, 15)
        kw = dict(n=name, a=a, b=b)
        ans = a * b
    else:
        b = rng.randint(3, 12)
        ans = rng.randint(3, 30)
        a = b * ans
        kw = dict(n=name, a=a, b=b)
    return _render(lang, kind, **kw), ans


def two_step(rng):
    kind = rng.choice(['add_sub', 'mul_sub', 'mul_div', 'sub_sub', 'add_add'])
    lang = rng.choice(['en', 'cn'])
    name = rng.choice(CN_NAMES if lang == 'cn' else NAMES)
    if kind == 'add_sub':
        a, b = rng.randint(20, 300), rng.randint(10, 200)
        ans = a + b
        c = rng.randint(5, ans - 5)
        kw = dict(n=name, a=a, b=b, c=c)
        ans -= c
    elif kind == 'mul_sub':
        a, b = rng.randint(4, 15), rng.randint(4, 12)
        c = rng.randint(1, a * b - 1)
        kw = dict(n=name, a=a, b=b, c=c)
        ans = a * b - c
    elif kind == 'mul_div':
        c = rng.randint(2, 9)
        per = rng.randint(3, 15)
        b = rng.randint(2, 8)
        kw = dict(n=name, b=b, c=c, p=per, cp=c*per)
        ans = per * b
    elif kind == 'sub_sub':
        a = rng.randint(200, 999)
        b = rng.randint(30, 150)
        c = rng.randint(10, 100)
        kw = dict(n=name, a=a, b=b, c=c)
        ans = a - b - c
    else:
        a, b, c = rng.randint(15, 200), rng.randint(15, 200), rng.randint(15, 200)
        kw = dict(n=name, a=a, b=b, c=c)
        ans = a + b + c
    return _render(lang, kind, **kw), ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n1', type=int, default=3000, help='一步题数量')
    parser.add_argument('--n2', type=int, default=4500, help='两步题数量')
    parser.add_argument('--dst', default='dataset/easy_math.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    out = []
    for gen, n in ((one_step, args.n1), (two_step, args.n2)):
        made = 0
        while made < n:
            q, ans = gen(rng)
            if q in seen:
                continue
            seen.add(q)
            out.append({'conversations': [
                {'role': 'user', 'content': q + INSTRUCTION},
            ], 'answer': str(ans)})
            made += 1

    rng.shuffle(out)
    with open(args.dst, 'w') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'写出 {len(out)} 条（一步 {args.n1}，两步 {args.n2}）-> {args.dst}')


if __name__ == '__main__':
    main()
