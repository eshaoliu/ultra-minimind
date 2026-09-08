"""APE210K + Math23K -> 中文数学逐步算式 SFT（等式自动展开 + #### 终答，可自动校验）。

Usage: python scripts/build_cnmath_sft.py [--n_ape 25000] [--dst dataset/cnmath_sft_v1.jsonl]
"""
import argparse
import json
import random
import re

INSTRUCTION = "\n请一步步计算，最后在 #### 后给出最终数字答案（例如：#### 42）。"
OP_SYM = {"*": "×", "/": "÷", "+": "+", "-": "-", "^": "^"}


def read_ape(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def read_m23k(path):
    txt = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder()
    items, i = [], 0
    while i < len(txt):
        while i < len(txt) and txt[i] in " \r\n\t":
            i += 1
        if i >= len(txt):
            break
        o, i = dec.raw_decode(txt, i)
        items.append(o)
    return items


def tokenize(s):
    s = s.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
    s = s.replace("[", "(").replace("]", ")").replace("^", "^")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(^|\()(?=-)", r"\g<1>0-", s)  # unary minus -> 0-
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            num = s[i:j]
            if j < len(s) and s[j] == "%":
                toks.append(("NUM", float(num) / 100.0, num + "%"))
                j += 1
            else:
                toks.append(("NUM", float(num), num))
            i = j
        elif c in "()+-*/^":
            toks.append((c, None, None))
            i += 1
        else:
            return None
    return toks


def parse(toks):
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None, None)

    def expr():
        nonlocal pos
        node = term()
        while pos < len(toks) and peek()[0] in ("+", "-"):
            op = peek()[0]
            pos += 1
            node = ("OP", op, node, term())
        return node

    def term():
        nonlocal pos
        node = pow_()
        while pos < len(toks) and peek()[0] in ("*", "/"):
            op = peek()[0]
            pos += 1
            node = ("OP", op, node, pow_())
        return node

    def pow_():
        nonlocal pos
        node = atom()
        if pos < len(toks) and peek()[0] == "^":
            pos += 1
            node = ("OP", "^", node, pow_())
        return node

    def atom():
        nonlocal pos
        t = peek()
        if t[0] == "NUM":
            pos += 1
            return ("NUM", t[1], t[2])
        if t[0] == "(":
            pos += 1
            n = expr()
            if pos < len(toks) and peek()[0] == ")":
                pos += 1
            return n
        return None

    node = expr()
    return node if pos == len(toks) else None


def fmt(v):
    if v is None:
        return "?"
    v = round(v + 0.0, 10)
    if abs(v) < 1e-9:
        return "0"
    if abs(v) < 1e15 and float(v).is_integer():
        return str(int(v))
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def walk(node, lines):
    if node[0] == "NUM":
        return node[1], node[2]
    _, op, left, right = node
    lv, ll = walk(left, lines)
    rv, rl = walk(right, lines)
    if op == "+":
        res = lv + rv
    elif op == "-":
        res = lv - rv
    elif op == "*":
        res = lv * rv
    elif op == "/":
        res = lv / rv
    elif op == "^":
        res = lv ** rv
    else:
        return None, None
    lines.append(f"{ll} {OP_SYM[op]} {rl} = {fmt(res)}")
    return res, fmt(res)


def build_trace(equation):
    if "=" in equation:
        equation = equation.split("=", 1)[1]
    toks = tokenize(equation)
    if not toks:
        return None
    ast = parse(toks)
    if ast is None:
        return None
    lines = []
    try:
        v, _ = walk(ast, lines)
    except Exception:
        return None
    return v, lines


def gold_value(ans):
    a = ans.strip().replace("％", "%")
    if a.endswith("%"):
        return None  # 先跳过以百分比为最终答案的行
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", a)
    if not m:
        return None
    return float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ape", type=int, default=25000)
    ap.add_argument("--dst", default="dataset/cnmath_sft_v1.jsonl")
    args = ap.parse_args()

    ape = read_ape("dataset/ape210k/train.ape.json")
    m23 = read_m23k("dataset/math23k/math23k_train.json")
    random.Random(42).shuffle(ape)

    seen = set()
    stats = {"ape_ok": 0, "m23_ok": 0, "dup": 0, "bad_eq": 0, "bad_ans": 0}
    out = []
    rows = [("m23k", x) for x in m23] + [("ape210k", x) for x in ape[: args.n_ape]]
    for src, row in rows:
        text = (row.get("original_text") or "").strip()
        if not text or text in seen or len(text) > 280:
            stats["dup"] += 1
            continue
        eq = (row.get("equation") or "").strip()
        got = build_trace(eq)
        if got is None:
            stats["bad_eq"] += 1
            continue
        v, lines = got
        gv = gold_value(row.get("ans") or "")
        if gv is None or abs(v - gv) > 1e-5 * max(1.0, abs(gv)):
            stats["bad_ans"] += 1
            continue
        ans = fmt(v)
        content = "\n".join(lines)
        if content:
            content += "\n"
        content += f"所以：x = {ans}\n#### {ans}"
        out.append({
            "conversations": [
                {"role": "user", "content": text + INSTRUCTION},
                {"role": "assistant", "content": content},
            ],
            "source": src,
        })
        seen.add(text)
        if src == "m23k":
            stats["m23_ok"] += 1
        else:
            stats["ape_ok"] += 1

    with open(args.dst, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("stats", stats, "total", len(out), "->", args.dst)
    for x in out[:6]:
        print("---")
        print("Q:", x["conversations"][0]["content"][:120].replace("\n", " "))
        print("A:", x["conversations"][1]["content"].replace("\n", " | "))


if __name__ == "__main__":
    main()
